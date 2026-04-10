"""Branch origin data structure.

The actual detection logic has moved to phase2_analysis.py (combined
cross-section sweep). This module is kept so that phase4_branch_growth
can import BranchOrigin without a circular dependency.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BranchOrigin:
    label: int
    points: np.ndarray        # (N, 3) seed points near the branch base
    direction: np.ndarray     # (3,) unit initial growth direction
    distance_to_root: float   # distance from seed to main root centerline
