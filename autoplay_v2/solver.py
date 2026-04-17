from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from autoplay_v2.models import SolvedWord


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


def build_trie(words: Iterable[str]) -> TrieNode:
    root = TrieNode()
    for word in words:
        node = root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True
    return root


def score_word(word: str) -> float:
    return float(max(0, len(word) - 2))


def flatten_grid(grid: List[List[str]]) -> List[str]:
    return [token for row in grid for token in row]


def build_neighbors(size: int) -> List[List[int]]:
    neighbors = [[] for _ in range(size * size)]
    for row in range(size):
        for col in range(size):
            idx = row * size + col
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = row + dr
                    nc = col + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        neighbors[idx].append(nr * size + nc)
    return neighbors


def build_solver_resources(words: Set[str], min_word_len: int = 3) -> Dict[str, object]:
    normalized_words = {word.strip().upper() for word in words if word.strip()}
    max_len = max((len(word) for word in normalized_words), default=0)
    return {
        "trie_root": build_trie(normalized_words),
        "max_word_len": max_len,
        "min_word_len": min_word_len,
        "neighbors_by_size": {},
    }


def _trie_advance(node: TrieNode, token: str) -> Optional[TrieNode]:
    cur = node
    for ch in token:
        cur = cur.children.get(ch)
        if cur is None:
            return None
    return cur


def solve_board_with_paths(
    grid: List[List[str]],
    resources: Dict[str, object],
    deadline: Optional[float] = None,
) -> List[SolvedWord]:
    if not grid or not grid[0]:
        return []

    size = len(grid)
    tiles = flatten_grid(grid)
    trie_root: TrieNode = resources["trie_root"]  # type: ignore[assignment]
    max_word_len = int(resources["max_word_len"])
    min_word_len = int(resources["min_word_len"])
    neighbors_by_size: Dict[int, List[List[int]]] = resources["neighbors_by_size"]  # type: ignore[assignment]
    neighbors = neighbors_by_size.setdefault(size, build_neighbors(size))

    found: Dict[str, SolvedWord] = {}
    check_interval = 128
    steps = 0

    def path_has_unique_tiles(path: List[int]) -> bool:
        return len(path) == len(set(path))

    def dfs(idx: int, node: TrieNode, mask: int, word: str, path: List[int]) -> None:
        nonlocal steps
        steps += 1
        if deadline is not None and steps % check_interval == 0 and time.perf_counter() >= deadline:
            return

        token = tiles[idx]
        next_node = _trie_advance(node, token)
        if next_node is None:
            return

        next_word = word + token
        next_path = path + [idx]
        if (
            next_node.is_end
            and len(next_word) >= min_word_len
            and next_word not in found
            and path_has_unique_tiles(next_path)
        ):
            found[next_word] = SolvedWord(
                word=next_word,
                path=next_path,
                score=score_word(next_word),
                length=len(next_word),
                token_count=len(next_path),
            )

        if len(next_word) >= max_word_len:
            return

        next_mask = mask | (1 << idx)
        for nb in neighbors[idx]:
            if next_mask & (1 << nb):
                continue
            if _trie_advance(next_node, tiles[nb]) is None:
                continue
            dfs(nb, next_node, next_mask, next_word, next_path)

    for start in range(len(tiles)):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if _trie_advance(trie_root, tiles[start]) is None:
            continue
        dfs(start, trie_root, 0, "", [])

    return sorted(
        found.values(),
        key=lambda item: (-item.score, -item.length, item.word, tuple(item.path)),
    )
