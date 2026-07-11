from __future__ import annotations

from bas.operator_controls import RuntimeControlState, toggled_object_group
from bas.schemas import Event


def test_runtime_control_state_clears_single_turn_overrides_after_turn_resolve() -> None:
    state = RuntimeControlState(free_shot_active=True, black_shot_active=True)

    state.advance_from_events([Event(name="TURN_RESOLVE", ts_cam_ns=1, frame_id=1)])

    assert state.free_shot_active is False
    assert state.black_shot_active is False


def test_toggle_object_group_only_flips_a_known_object_group() -> None:
    assert toggled_object_group(None) is None
    assert toggled_object_group("solid") == "stripe"
    assert toggled_object_group("stripe") == "solid"
    assert toggled_object_group("black") is None
