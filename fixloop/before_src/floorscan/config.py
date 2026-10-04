"""All tunable parameters in one place (nothing is silently hard-coded in the geometry code)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- drift handling (ablation switches) -----------------------------------
    local_planar: bool = True  # plane-anchored per-room pose correction
    door_graph: bool = True  # door-anchored room pose graph (loop closure)
    planar_iters: int = 3
    # --- geometry --------------------------------------------------------------
    grid_res: float = 0.05
    persistence: float = 0.12  # m, watershed persistence (room vs doorway)
    min_room_area: float = 1.5
    wall_window: float = 0.45
    wall_min_frac: float = 0.12
    default_wall_thickness: float = 0.12
    chunk_frames: int = 6
    # --- misc ---------------------------------------------------------------------
    seed: int = 0
    damage: bool = True
    # --- tier-specific --------------------------------------------------------------
    scale_card_mm: float = 100.0
