"""rootweave: inward-only root segmentation with shrinking-shell tip seeding.

Five phases:
  1. phase1_main_path        — taproot centerline (Dijkstra + iterative centering).
  2. phase2_main_volume      — R(t) radius profile + main root volume mask.
  3. phase3_tip_seeds        — shrinking-tube tip seed detection with KNN-graph
                               classification.
  4. phase4_inward_tracking  — inward-only tracker, longest seed first, with
                               arrival at the taproot or previously finished laterals.
  5. phase5_classify_merge   — classify each attachment point (REDUNDANT /
                               SPLIT-EXTENSION / SIBLING-SECONDARY / TERTIARY)
                               and merge split artifacts, yielding a flat
                               root-order tree (ClassifiedBranch list).

Entry points:
  - ``rootweave.run``: full pipeline
  - ``rootweave.run_phase1_only``: load + taproot extraction only
  - ``rootweave.PipelineConfig``: all tunable parameters
"""

from .config import PipelineConfig
from .pipeline import run, run_phase1_only
