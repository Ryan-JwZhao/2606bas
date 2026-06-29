from __future__ import annotations

import numpy as np

from bas.planning.cue_sector import CueSectorDebugView
from bas.ui.cue_sector_preview import draw_cue_sector_candidate_box


def test_cue_sector_candidate_box_draws_when_enabled() -> None:
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    debug_view = CueSectorDebugView(
        cue_center_px=(120.0, 160.0),
        direction_px=(1.0, 0.0),
        half_width_px=48.0,
        candidate_centers_px=((260.0, 160.0), (360.0, 188.0)),
        candidate_track_ids=(2, 3),
        status="own_group",
    )

    drawn = draw_cue_sector_candidate_box(image, debug_view, enabled=True)

    assert drawn == 3
    assert int(np.count_nonzero(image)) > 0


def test_cue_sector_candidate_box_skips_when_disabled() -> None:
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    debug_view = CueSectorDebugView(
        cue_center_px=(120.0, 160.0),
        direction_px=(1.0, 0.0),
        half_width_px=48.0,
        candidate_centers_px=((260.0, 160.0),),
        candidate_track_ids=(2,),
        status="own_group",
    )

    drawn = draw_cue_sector_candidate_box(image, debug_view, enabled=False)

    assert drawn == 0
    assert int(np.count_nonzero(image)) == 0
