from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator


class ReplayReader:
    def __init__(self, path: str | Path):
        p = Path(path)
        if p.is_dir():
            p = p / "events.jsonl"
        self.path = p

    def iter_events(self) -> Iterator[Dict[str, object]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

