from __future__ import annotations

from typing import List

from ..schemas import ShotCandidate


class DisabledLearningRanker:
    """Placeholder for the future independent learning system.

    The online program deliberately does not collect training samples and does
    not apply learned ranking. It only preserves the interface where a frozen
    ranker can be plugged in later.
    """

    version = "learning_disabled"

    def rerank(self, candidates: List[ShotCandidate]) -> List[ShotCandidate]:
        return sorted(candidates, key=lambda c: c.score, reverse=True)

