from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import summarize_samples


RL_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect BAS learning samples.")
    parser.add_argument("--samples", default=str(RL_ROOT / "data" / "samples"), help="Sample JSONL file or directory.")
    args = parser.parse_args()
    summary = summarize_samples(args.samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
