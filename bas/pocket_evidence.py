from __future__ import annotations


# Low-frame-rate video can lose a ball between the last table-side sample and
# the pocket centre. Crossing this terminal corridor starts retractable
# evidence only; reappearance, outward motion, and lip occupancy still veto it.
VISUAL_TERMINAL_DEPTH_DIAMETERS = -0.75
VISUAL_TERMINAL_ADVANCE_DIAMETERS = 0.25


__all__ = [
    "VISUAL_TERMINAL_ADVANCE_DIAMETERS",
    "VISUAL_TERMINAL_DEPTH_DIAMETERS",
]
