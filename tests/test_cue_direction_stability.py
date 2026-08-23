from __future__ import annotations

import numpy as np

from bas.config import PlannerConfig
from bas.planning.cue_direction_resolver import CueDirectionResolver
from bas.planning.cue_direction_stability import CueDirectionStabilizer


def test_one_frame_orthogonal_jump_waits_for_confirmation() -> None:
    stabilizer = CueDirectionStabilizer(
        PlannerConfig(
            cue_sector_angle_deg=15.0,
            cue_sector_edge_margin_deg=1.0,
            cue_sector_switch_confirm_frames=3,
        )
    )
    stabilizer.stabilize(
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    decision = stabilizer.stabilize(
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    )

    np.testing.assert_allclose(decision.direction_px, (1.0, 0.0), atol=1.0e-6)
    assert decision.status == "switch_pending:1/3"


def test_one_pixel_body_imbalance_remains_ambiguous() -> None:
    resolved = CueDirectionResolver().resolve(
        cue_center_px=np.asarray([500.0, 250.0], dtype=np.float32),
        cue_radius_px=15.0,
        p1_px=np.asarray([450.0, 250.0], dtype=np.float32),
        p2_px=np.asarray([551.0, 250.0], dtype=np.float32),
    )

    assert resolved is not None
    assert resolved.status == "body_side_ambiguous"
