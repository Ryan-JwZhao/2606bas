from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_dir: str | Path = "logs", level: str = "INFO") -> Path:
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / "bas.log"

    fmt = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    rotating = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    rotating.setFormatter(fmt)

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(rotating)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return file_path

