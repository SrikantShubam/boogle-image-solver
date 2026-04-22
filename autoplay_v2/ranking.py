from __future__ import annotations

from typing import Iterable, List

from autoplay_v2.models import SolvedWord


def rank_solved_words(words: Iterable[SolvedWord]) -> List[SolvedWord]:
    return sorted(
        words,
        key=lambda item: (
            0 if item.length == 3 else 1 if item.length == 4 else 2,
            item.length,
            -item.score,
            len(item.path),
            item.word,
            tuple(item.path),
        ),
    )
