from autoplay_v2.solver import build_solver_resources, solve_board_with_paths


def test_solver_returns_paths_and_handles_digraph_tokens():
    grid = [
        ["TH", "E"],
        ["N", "A"],
    ]
    words = {"THEN", "THE", "HEN", "THENA"}
    resources = build_solver_resources(words)
    solved = solve_board_with_paths(grid, resources)
    solved_map = {item.word: item for item in solved}

    assert "THEN" in solved_map
    assert solved_map["THEN"].path == [0, 1, 2]
    assert solved_map["THEN"].token_count == 3
    assert solved_map["THEN"].length == 4


def test_solver_does_not_reuse_tiles_for_single_word():
    grid = [
        ["A", "B"],
        ["C", "D"],
    ]
    words = {"ABA", "AB", "ABCD"}
    resources = build_solver_resources(words)
    solved = solve_board_with_paths(grid, resources)
    solved_words = {item.word for item in solved}
    assert "ABA" not in solved_words
    for item in solved:
        assert len(item.path) == len(set(item.path))


def test_solver_has_deterministic_ordering_for_equal_score():
    grid = [
        ["C", "A"],
        ["R", "T"],
    ]
    words = {"CAR", "CAT"}
    resources = build_solver_resources(words)
    solved = solve_board_with_paths(grid, resources)
    assert [item.word for item in solved] == ["CAR", "CAT"]
