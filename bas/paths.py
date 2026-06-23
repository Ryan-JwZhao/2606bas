from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def resolve_path(value: str | Path | None, *, base: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return ((base or PROJECT_ROOT) / p).resolve()
