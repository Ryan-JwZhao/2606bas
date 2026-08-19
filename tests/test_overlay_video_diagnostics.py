from bas.video_diagnostics.overlay_video import TripletFlashDetector


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
