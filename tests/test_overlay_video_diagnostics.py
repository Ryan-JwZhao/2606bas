from bas.video_diagnostics.overlay_video import (
    TripletFlashDetector,
    _apply_target_group_override,
    build_parser,
)


def test_parser_defaults_to_state_machine_target_group() -> None:
    args = build_parser().parse_args(["--video", "sample.mp4"])

    assert args.target_group is None


def test_parser_keeps_explicit_target_group_override() -> None:
    args = build_parser().parse_args(
        ["--video", "sample.mp4", "--target-group", "stripe"]
    )

    assert args.target_group == "stripe"


class _RecordingStateMachine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def set_turn_target_group(self, group: str, *, reason: str) -> None:
        self.calls.append((group, reason))


def test_default_target_group_does_not_mutate_state_machine() -> None:
    state_machine = _RecordingStateMachine()

    applied = _apply_target_group_override(state_machine, None)

    assert applied is False
    assert state_machine.calls == []


def test_explicit_target_group_still_mutates_state_machine() -> None:
    state_machine = _RecordingStateMachine()

    applied = _apply_target_group_override(state_machine, "solid")

    assert applied is True
    assert state_machine.calls == [("solid", "diagnostic_video_replay")]


def test_triplet_detector_flags_stable_one_frame_flash() -> None:
    detector = TripletFlashDetector("route_presence")

    detector.push(10, True, stable=True)
    detector.push(11, False, stable=True)
    detector.push(12, True, stable=True)

    assert detector.total == 1
    assert detector.events[0]["frame"] == 11


def test_triplet_detector_ignores_motion_transition_and_persistent_change() -> None:
    detector = TripletFlashDetector("route_presence")

    detector.push(20, True, stable=True)
    detector.push(21, False, stable=False)
    detector.push(22, True, stable=True)
    detector.push(23, False, stable=True)
    detector.push(24, False, stable=True)

    assert detector.total == 0
