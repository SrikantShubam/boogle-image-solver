from autoplay_v2.models import SolvedWord
from autoplay_v2.ranking import rank_solved_words


def test_rank_solved_words_score_then_length_then_lexical():
    items = [
        SolvedWord(word="DOG", path=[0, 1, 2], score=1.0, length=3, token_count=3),
        SolvedWord(word="DRAGON", path=[0, 1, 2, 3, 4, 5], score=4.0, length=6, token_count=6),
        SolvedWord(word="DRAIN", path=[0, 1, 2, 3, 4], score=3.0, length=5, token_count=5),
        SolvedWord(word="DROPS", path=[0, 4, 8, 9, 10], score=3.0, length=5, token_count=5),
    ]
    ranked = rank_solved_words(items)
    assert [item.word for item in ranked] == ["DRAGON", "DRAIN", "DROPS", "DOG"]


def test_rank_solved_words_prefers_shorter_path_on_equal_score_and_length():
    items = [
        SolvedWord(word="HEAT", path=[0, 1, 2, 3], score=2.0, length=4, token_count=4),
        SolvedWord(word="HEAL", path=[0, 1, 5, 9], score=2.0, length=4, token_count=4),
    ]
    ranked = rank_solved_words(items)
    assert [item.word for item in ranked] == ["HEAL", "HEAT"]
