from __future__ import annotations

from collections.abc import Callable

from ..config import DetectorConfig
from ..training import RULES_MODE, TRAINING_MODE, normalize_operating_mode
from .detector import Detector, create_detector
from .service import DetectService


class ModeAwareDetectService:
    """Lazily owns one detection service per operating mode."""

    def __init__(
        self,
        rule_config: DetectorConfig,
        training_config: DetectorConfig,
        *,
        initial_mode: str = RULES_MODE,
        detector_factory: Callable[[DetectorConfig], Detector] = create_detector,
    ):
        self._configs = {RULES_MODE: rule_config, TRAINING_MODE: training_config}
        self._services: dict[str, DetectService] = {}
        self._detector_factory = detector_factory
        self._mode = RULES_MODE
        self.activate(initial_mode)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def detector(self) -> Detector:
        return self._services[self._mode].detector

    def activate(self, mode: str) -> str:
        normalized = normalize_operating_mode(mode)
        if normalized not in self._services:
            config = self._configs[normalized]
            self._services[normalized] = DetectService(
                self._detector_factory(config),
                detect_interval_frames=config.detect_interval_frames,
                detect_fps_limit_hz=config.detect_fps_limit_hz,
            )
        self._mode = normalized
        self._services[normalized].reset_cache()
        return normalized

    def process(self, *args, **kwargs):
        return self._services[self._mode].process(*args, **kwargs)

    def reset_cache(self) -> None:
        self._services[self._mode].reset_cache()
