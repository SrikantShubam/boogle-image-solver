from autoplay_v2.feedback import append_feedback_entry, read_recent_feedback
from autoplay_v2.config import repo_root


def test_append_feedback_entries_and_read_recent():
    feedback_path = repo_root() / "runs" / "_pytest_dictionary_feedback.jsonl"
    append_feedback_entry(
        feedback_path=feedback_path,
        word="THEN",
        status="accepted",
        board_signature="ABCD|EFGH|IJKL|MNOP",
    )
    append_feedback_entry(
        feedback_path=feedback_path,
        word="THANE",
        status="rejected",
        board_signature="ABCD|EFGH|IJKL|MNOP",
    )
    append_feedback_entry(
        feedback_path=feedback_path,
        word="QUEL",
        status="unknown",
        board_signature="ABCD|EFGH|IJKL|MNOP",
    )

    entries = read_recent_feedback(feedback_path=feedback_path, limit=2)
    assert len(entries) == 2
    assert entries[0]["word"] == "THANE"
    assert entries[1]["word"] == "QUEL"
    assert entries[0]["status"] == "rejected"
    assert entries[1]["status"] == "unknown"
    feedback_path.unlink(missing_ok=True)
