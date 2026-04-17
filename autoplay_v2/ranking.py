from __future__ import annotations

from typing import Iterable, List

from autoplay_v2.models import SolvedWord


def rank_solved_words(words: Iterable[SolvedWord]) -> List[SolvedWord]:
    return sorted(
        words,
        key=lambda item: (
            -item.score,
            -item.length,
            len(item.path),
            item.word,
            tuple(item.path),
        ),
    )
