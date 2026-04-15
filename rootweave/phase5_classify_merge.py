"""Phase 5: classify each lateral-stopped branch and merge split artifacts.

Phase 4 produces one ``BranchPath`` per seed.  Each path stops with a
``parent`` tag that says how it ended:

  * ``"taproot"``  — the branch reached the taproot directly.
  * ``"lateral"``  — the branch stopped on a previously-finished lateral.
  * ``"unknown"``  — direction collapse or step limit hit.

For every ``"lateral"``-stopped branch we analyze the **attachment point
on the parent** and choose one of four outcomes:

  1. **REDUNDANT** — the child's path is too short to be meaningful; the
     parent already covers the same physical region.  Drop the child.

  2. **SPLIT-EXTENSION** — the attachment sits near the PARENT'S TIP.
     The child is really the outer portion of a single biological root
     that was split by over-seeding or a phase-4 break.  Merge the two
     paths into one (child tip → child-to-parent-attach → parent's
     remainder → taproot), inherit the parent's order / parent_label,
     and absorb the parent's identity.

  3. **SIBLING-SECONDARY** — the attachment sits near the PARENT'S
     TAPROOT END.  The child is a sibling secondary that happened to
     arrive at the parent instead of the taproot.  Extend the child's
     path along the parent's remaining segment so it reaches the
     taproot, and keep both as independent secondaries.

  4. **TERTIARY** — the attachment is mid-shaft on the parent.  This
     IS a genuine branching.  Keep the child's path unchanged (tip to
     attachment only) and record ``parent_label`` pointing to the
     parent's current identity.  The child's ``order`` is
     ``parent.order + 1``.

The classification thresholds are config-driven:
``split_min_child_length``, ``split_tip_zone_ratio``,
``split_base_zone_ratio``.

Single-pass design: one flat list of ``ClassifiedBranch`` objects that
directly encodes the root-order tree (secondary / tertiary / …).  No
separate follow-up classification step is required.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import PipelineConfig
from .phase4_inward_tracking import BranchPath


# ---------------------------------------------------------------------------
# Output data model
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedBranch:
    """A single lateral root after classification + merging.

    Attributes
    ----------
    label
        Primary identifier.  Matches the originating seed's label.  If
        this branch absorbed another (SPLIT-EXTENSION), only the
        surviving label is kept here; ``absorbed_labels`` records the
        others.
    order
        Root order: 2 = secondary (direct to taproot), 3 = tertiary (on
        a secondary), 4 = quaternary, etc.  ``0`` flags an incomplete
        branch (``unknown`` stop) that couldn't be classified.
    path
        3D path points.  For secondaries the path runs tip → taproot.
        For tertiaries (and higher) the path runs tip → attachment on
        the parent (it does NOT continue along the parent's shaft).
    parent_label
        ``None`` for secondaries (parent is the taproot).  For
        tertiaries this is the parent secondary's ``label``.
    attachment_point
        3D location of the attachment (last node of ``path``) for
        non-taproot branches.  ``None`` for pure taproot-direct
        branches without any attachment info.
    attachment_index_on_parent
        Index into the parent's path at which this branch attaches.
        ``None`` for secondaries attached to the taproot.
    classification
        Which rule produced this entry: ``"taproot-direct"``,
        ``"split-extension"``, ``"sibling-secondary"``, ``"tertiary"``,
        or ``"unknown"``.
    absorbed_labels
        Labels of other ``BranchPath``s that were merged into this one
        via SPLIT-EXTENSION.  Empty for most branches.
    """

    label: int
    order: int
    path: List[np.ndarray]
    parent_label: Optional[int] = None
    attachment_point: Optional[np.ndarray] = None
    attachment_index_on_parent: Optional[int] = None
    classification: str = "unknown"
    absorbed_labels: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_and_merge(
    branches: List[BranchPath],
    config: PipelineConfig,
) -> List[ClassifiedBranch]:
    """Classify every phase-4 branch at its attachment point and merge
    split artifacts into their parents.

    Processes in ascending label order so parents (longer roots, seeded
    earlier under longest-first ordering) are classified before their
    children.  Children reference their parent via ``parent_label``; if
    a SPLIT-EXTENSION absorbed that parent, a redirect table ensures
    later children find the surviving entry.
    """
    # `classified` holds the current output entries, keyed by the label
    # of the BRANCH that currently owns each entry (may change when a
    # SPLIT-EXTENSION absorbs a parent).
    classified: Dict[int, ClassifiedBranch] = {}
    # Redirect: original (absorbed) label → current owning label.
    redirect: Dict[int, int] = {}

    n_redundant = 0
    n_split = 0
    n_sibling = 0
    n_tertiary = 0
    n_unknown = 0

    for b in sorted(branches, key=lambda x: x.label):

        # --- "unknown" stop: incomplete path, record as-is ---
        if b.parent == "unknown":
            classified[b.label] = ClassifiedBranch(
                label=b.label,
                order=0,
                path=[np.asarray(p) for p in b.path],
                classification="unknown",
            )
            n_unknown += 1
            continue

        # --- "taproot" stop: secondary directly on the taproot ---
        if b.parent == "taproot":
            attach_pt = (
                np.asarray(b.path[-1]) if len(b.path) > 0 else None
            )
            classified[b.label] = ClassifiedBranch(
                label=b.label,
                order=2,
                path=[np.asarray(p) for p in b.path],
                parent_label=None,
                attachment_point=attach_pt,
                attachment_index_on_parent=None,
                classification="taproot-direct",
            )
            continue

        # --- "lateral" stop: classify against the parent ---
        if b.parent_label is None:
            # Shouldn't happen, but guard.
            classified[b.label] = ClassifiedBranch(
                label=b.label, order=0,
                path=[np.asarray(p) for p in b.path],
                classification="unknown",
            )
            n_unknown += 1
            continue

        parent_current_label = _resolve(redirect, b.parent_label)
        parent_entry = classified.get(parent_current_label)

        if parent_entry is None or len(parent_entry.path) == 0:
            # Parent missing or empty: treat as unknown.
            classified[b.label] = ClassifiedBranch(
                label=b.label, order=0,
                path=[np.asarray(p) for p in b.path],
                classification="unknown",
            )
            n_unknown += 1
            continue

        # Find the attachment index on the parent's CURRENT path
        # (the parent's path may already be a merged form after an
        # earlier SPLIT-EXTENSION absorbed its own parent; using the
        # current path is correct).
        attach_pt = np.asarray(b.path[-1])
        parent_arr = np.asarray(parent_entry.path)
        dists = np.linalg.norm(parent_arr - attach_pt, axis=1)
        attach_idx = int(np.argmin(dists))

        parent_len = max(len(parent_entry.path) - 1, 1)
        attach_ratio = attach_idx / parent_len
        child_len = len(b.path)

        # --- REDUNDANT: very short child, parent already has this region ---
        if child_len < config.split_min_child_length:
            # Don't emit; the parent entry already covers it.
            n_redundant += 1
            # Redirect future children of this label back to the parent
            redirect[b.label] = parent_current_label
            continue

        # --- SPLIT-EXTENSION: attach near the parent's TIP ---
        if attach_ratio < config.split_tip_zone_ratio:
            merged_path = _merge_tip_extension(
                b.path, parent_entry.path, attach_idx,
            )
            # The merged branch takes over the parent's role (same order,
            # same grandparent, same attachment on grandparent).
            new_entry = ClassifiedBranch(
                label=b.label,  # child label (it carries the true tip)
                order=parent_entry.order,
                path=merged_path,
                parent_label=parent_entry.parent_label,
                attachment_point=parent_entry.attachment_point,
                attachment_index_on_parent=parent_entry.attachment_index_on_parent,
                classification="split-extension",
                absorbed_labels=(
                    [parent_entry.label] + list(parent_entry.absorbed_labels)
                ),
            )
            # Swap: remove parent entry, insert merged entry under child's label
            del classified[parent_current_label]
            classified[b.label] = new_entry
            # Future lookups for the old parent label should find the merged entry
            redirect[parent_current_label] = b.label
            if parent_entry.label != parent_current_label:
                redirect[parent_entry.label] = b.label
            n_split += 1
            continue

        # --- SIBLING-SECONDARY: attach near the parent's BASE ---
        if attach_ratio > config.split_base_zone_ratio:
            extended = (
                [np.asarray(p) for p in b.path]
                + [np.asarray(p) for p in parent_entry.path[attach_idx + 1:]]
            )
            classified[b.label] = ClassifiedBranch(
                label=b.label,
                order=2,                    # sibling of parent → still a secondary
                path=extended,
                parent_label=None,          # attached to taproot
                attachment_point=(
                    np.asarray(extended[-1]) if extended else None
                ),
                attachment_index_on_parent=None,
                classification="sibling-secondary",
            )
            n_sibling += 1
            continue

        # --- TERTIARY: mid-shaft branching ---
        classified[b.label] = ClassifiedBranch(
            label=b.label,
            order=parent_entry.order + 1,
            path=[np.asarray(p) for p in b.path],
            parent_label=parent_entry.label,
            attachment_point=attach_pt.copy(),
            attachment_index_on_parent=attach_idx,
            classification="tertiary",
        )
        n_tertiary += 1

    print(
        f"Classify+merge: {len(classified)} classified branches "
        f"({n_tertiary} tertiary, {n_sibling} sibling-secondary, "
        f"{n_split} split-extension, {n_redundant} redundant dropped, "
        f"{n_unknown} unknown)"
    )
    return list(classified.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(redirect: Dict[int, int], label: int) -> int:
    """Follow the redirect chain (absorbed → survivor) to the terminal label."""
    seen = set()
    while label in redirect and label not in seen:
        seen.add(label)
        label = redirect[label]
    return label


def _merge_tip_extension(
    child_path,
    parent_path,
    attach_idx: int,
    dedupe_tol: float = 1e-9,
) -> List[np.ndarray]:
    """Concatenate ``child_path`` with the part of ``parent_path`` that
    lies beyond the attachment, skipping the duplicated attach node.

    The result runs child_tip → child's second-to-last node →
    parent[attach_idx] (possibly dropped) → parent[attach_idx+1] → … →
    parent[-1].
    """
    out = [np.asarray(p) for p in child_path]
    if len(parent_path) == 0:
        return out

    parent_remainder = [np.asarray(p) for p in parent_path[attach_idx:]]
    # Drop the very first point of parent_remainder if it duplicates the
    # child's last point (they represent the same physical attachment).
    if out and parent_remainder:
        if np.linalg.norm(out[-1] - parent_remainder[0]) <= dedupe_tol:
            parent_remainder = parent_remainder[1:]
    out.extend(parent_remainder)
    return out
