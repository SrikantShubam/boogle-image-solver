# BOGGLE SOLVER - WINNING VERSION
# Generated automatically by boggle_solver_benchmark.ipynb

import time
import random
import string
import heapq
import urllib.request
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

BOARD_SIZE = 5
MIN_WORD_LEN = 3
TIME_BUDGET_SECONDS = 10.0
RNG_SEED = 1337
MAX_SPECIAL_NODES = 2
BONUS_NODE_PROBABILITY = 1.0
DIGRAPH_TOKENS = ['QU', 'TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ON', 'AT', 'EN']
WORDS_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
WORDS_PATH = Path("words.txt")
LETTER_WEIGHT = {'E': 0.2, 'T': 0.3, 'A': 0.3, 'O': 0.3, 'I': 0.4, 'N': 0.4, 'S': 0.5, 'H': 0.5, 'R': 0.5, 'D': 0.7, 'L': 0.7, 'C': 0.8, 'U': 0.8, 'M': 0.9, 'W': 1.0, 'F': 1.1, 'G': 1.1, 'Y': 1.2, 'P': 1.2, 'B': 1.3, 'V': 1.4, 'K': 1.5, 'J': 2.0, 'X': 2.1, 'Q': 2.2, 'Z': 2.3}

def ensure_word_list(path=WORDS_PATH, url=WORDS_URL):
    if not path.exists():
        print(f"Downloading word list to {path} ...")
        urllib.request.urlretrieve(url, path)
    return path

def load_words_from_file(path, min_len=MIN_WORD_LEN):
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            word = raw.strip().upper()
            if len(word) >= min_len and word.isalpha():
                words.add(word)
    return words

def build_prefixes(words):
    prefixes = set()
    for w in words:
        for i in range(1, len(w) + 1):
            prefixes.add(w[:i])
    return prefixes

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

def build_trie(words):
    trie = Trie()
    for w in words:
        trie.insert(w)
    return trie

def prepare_solver_resources(words):
    return {
        "all_words": words,
        "prefixes": build_prefixes(words),
        "trie": build_trie(words),
        "max_word_len": max(len(w) for w in words),
        "letter_weight": LETTER_WEIGHT,
    }

def score_word(word):
    return max(0, len(word) - 2)

def _token_weight(token, letter_weight):
    if not token:
        return 1.0
    return sum(letter_weight.get(ch, 1.0) for ch in token) / len(token)

def flatten_grid(grid):
    return [ch for row in grid for ch in row]

def serialize_grid(grid):
    return "|".join(",".join(row) for row in grid)

def serialize_board(board):
    grid = board["grid"]
    digraph_indices = set(board.get("digraph_indices", set()))
    bonus_index = board.get("bonus_index")
    size = len(grid)
    rows = []
    for r in range(size):
        cells = []
        for c in range(size):
            idx = r * size + c
            token = grid[r][c]
            suffix = ""
            if idx in digraph_indices:
                suffix += "~"
            if bonus_index is not None and idx == bonus_index:
                suffix += "*"
            cells.append(f"{token}{suffix}")
        rows.append(",".join(cells))
    return "|".join(rows)

def build_neighbors(size):
    neighbors = [[] for _ in range(size * size)]
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        neighbors[idx].append(nr * size + nc)
    return neighbors

def generate_random_grid(size=BOARD_SIZE, rng=None):
    rng = rng or random
    letters = string.ascii_uppercase
    return [[rng.choice(letters) for _ in range(size)] for _ in range(size)]

def generate_random_board(
    size=BOARD_SIZE,
    rng=None,
    max_special_nodes=MAX_SPECIAL_NODES,
    digraph_tokens=DIGRAPH_TOKENS,
    bonus_probability=BONUS_NODE_PROBABILITY,
):
    rng = rng or random
    grid = generate_random_grid(size=size, rng=rng)
    n = size * size

    special_count = rng.randint(0, max_special_nodes)
    special_indices = set(rng.sample(range(n), k=special_count)) if special_count else set()
    for idx in special_indices:
        r, c = divmod(idx, size)
        grid[r][c] = rng.choice(digraph_tokens)

    bonus_index = rng.randrange(n) if rng.random() <= bonus_probability else None
    return {
        "grid": grid,
        "digraph_indices": special_indices,
        "bonus_index": bonus_index,
    }

def make_board_word_matcher(grid, neighbors, digraph_indices=None, bonus_index=None):
    tokens = flatten_grid(grid)
    n = len(tokens)
    digraph_set = set(digraph_indices or set())

    starts_by_char = defaultdict(list)
    for i, tok in enumerate(tokens):
        if tok:
            starts_by_char[tok[0]].append(i)

    def tile_bonus(idx):
        b = 0.0
        if idx in digraph_set:
            b += 0.5
        if bonus_index is not None and idx == bonus_index:
            b += 1.0
        return b

    @lru_cache(maxsize=None)
    def best_bonus(word):
        if not word:
            return None

        @lru_cache(maxsize=None)
        def dfs(last_idx, pos, mask):
            if pos == len(word):
                return 0.0

            best_val = None
            for nb in neighbors[last_idx]:
                bit = 1 << nb
                if mask & bit:
                    continue
                tok = tokens[nb]
                if not word.startswith(tok, pos):
                    continue
                nxt = dfs(nb, pos + len(tok), mask | bit)
                if nxt is None:
                    continue
                cand = tile_bonus(nb) + nxt
                if best_val is None or cand > best_val:
                    best_val = cand
            return best_val

        best_val = None
        for start in starts_by_char.get(word[0], []):
            tok = tokens[start]
            if not word.startswith(tok, 0):
                continue

            consumed = len(tok)
            base = tile_bonus(start)
            if consumed == len(word):
                cand = base
            else:
                nxt = dfs(start, consumed, 1 << start)
                if nxt is None:
                    continue
                cand = base + nxt

            if best_val is None or cand > best_val:
                best_val = cand

        return best_val

    return best_bonus

def validate_candidates(candidates, grid, all_words, neighbors, digraph_indices=None, bonus_index=None):
    raw = []
    for w in candidates:
        if isinstance(w, str):
            s = w.strip().upper()
            if s:
                raw.append(s)

    unique = set(raw)
    matcher = make_board_word_matcher(
        grid,
        neighbors,
        digraph_indices=digraph_indices,
        bonus_index=bonus_index,
    )

    valid = set()
    invalid_rejected = 0
    bonus_points_total = 0.0
    score_total = 0.0
    for word in unique:
        if len(word) < MIN_WORD_LEN:
            invalid_rejected += 1
            continue
        if word not in all_words:
            invalid_rejected += 1
            continue

        bonus = matcher(word)
        if bonus is None:
            invalid_rejected += 1
            continue

        valid.add(word)
        bonus_points_total += bonus
        score_total += score_word(word) + bonus

    duplicate_rejected = max(0, len(raw) - len(unique))
    return {
        "valid_words": valid,
        "score": score_total,
        "bonus_points": bonus_points_total,
        "raw_count": len(raw),
        "valid_count": len(valid),
        "invalid_rejected": invalid_rejected,
        "duplicate_rejected": duplicate_rejected,
    }

def _trie_advance(node, token):
    cur = node
    for ch in token:
        cur = cur.children.get(ch)
        if cur is None:
            return None
    return cur

def solve_exact_trie_dfs(grid, deadline, resources):
    tiles = flatten_grid(grid)
    neighbors = resources["neighbors"]
    trie_root = resources["trie"].root
    max_word_len = resources["max_word_len"]

    found = set()
    steps = 0

    def dfs(idx, node, mask, path):
        nonlocal steps
        steps += 1
        if steps % 128 == 0 and time.perf_counter() >= deadline:
            return

        token = tiles[idx]
        next_node = _trie_advance(node, token)
        if next_node is None:
            return

        word = path + token
        if next_node.is_end and len(word) >= MIN_WORD_LEN:
            found.add(word)

        if len(word) >= max_word_len:
            return

        new_mask = mask | (1 << idx)
        for nb in neighbors[idx]:
            if new_mask & (1 << nb):
                continue
            if _trie_advance(next_node, tiles[nb]) is not None:
                dfs(nb, next_node, new_mask, word)

    for start in range(len(tiles)):
        if time.perf_counter() >= deadline:
            break
        if _trie_advance(trie_root, tiles[start]) is not None:
            dfs(start, trie_root, 0, "")

    return found

WINNER_NAME = 'Exact_Trie_DFS'
WINNER_SOLVER = solve_exact_trie_dfs

def solve_board(board, words_path=WORDS_PATH, time_budget=TIME_BUDGET_SECONDS):
    resources = prepare_solver_resources(load_words_from_file(ensure_word_list(words_path)))
    if isinstance(board, dict):
        grid = board['grid']
        digraph_indices = board.get('digraph_indices', set())
        bonus_index = board.get('bonus_index')
    else:
        grid = board
        digraph_indices = set()
        bonus_index = None
    resources['neighbors'] = build_neighbors(len(grid))
    deadline = time.perf_counter() + time_budget
    candidates = WINNER_SOLVER(grid, deadline, resources)
    return validate_candidates(candidates, grid, resources['all_words'], resources['neighbors'], digraph_indices=digraph_indices, bonus_index=bonus_index)

if __name__ == '__main__':
    b = generate_random_board()
    result = solve_board(b)
    print('Board:')
    for row in b['grid']:
        print(' '.join(row))
    print('Board key:', serialize_board(b))
    print('Winner solver:', WINNER_NAME)
    print('Valid unique words:', result['valid_count'])
    print('Bonus points:', result['bonus_points'])
    print('Score:', result['score'])
