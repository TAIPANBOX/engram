"""Tests for the Engram CLI (engram.cli)."""

from __future__ import annotations

import pytest

from engram import Engram, StubLLMAdapter
from engram.cli import main


@pytest.fixture()
def store_path(tmp_path):
    """Return a path to a pre-populated .engram file."""
    path = str(tmp_path / "test.engram")
    stub = StubLLMAdapter(
        facts=[
            {"subject": "Ivan", "predicate": "works_at", "object": "Globex", "confidence": 0.9},
            {"subject": "Ivan", "predicate": "lives_in", "object": "Berlin", "confidence": 0.8},
        ]
    )
    with Engram(path=path, llm=stub) as mem:
        mem.observe("Ivan joined Globex last week", actors=["Ivan"], tags=["career"])
        mem.observe("Alice presented the roadmap", actors=["Alice"], tags=["work"])
        mem.observe("The team shipped v2 of the payment service", tags=["shipping"])
        mem.assert_fact("Alice", "role", "CTO", confidence=0.95)
        mem.reflect()
    return path


@pytest.fixture()
def empty_path(tmp_path):
    path = str(tmp_path / "empty.engram")
    with Engram(path=path):
        pass
    return path


# ------------------------------------------------------------------
# inspect
# ------------------------------------------------------------------


def test_inspect_shows_episode_count(store_path, capsys):
    main(["inspect", store_path])
    out = capsys.readouterr().out
    assert "Episodes:" in out
    assert "3" in out


def test_inspect_shows_facts(store_path, capsys):
    main(["inspect", store_path])
    out = capsys.readouterr().out
    assert "Facts:" in out


def test_inspect_shows_reflections(store_path, capsys):
    main(["inspect", store_path])
    out = capsys.readouterr().out
    assert "Reflections:" in out


def test_inspect_empty_store(empty_path, capsys):
    main(["inspect", empty_path])
    out = capsys.readouterr().out
    assert "Episodes:" in out
    assert "never run" in out


def test_inspect_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["inspect", str(tmp_path / "nonexistent.engram")])
    assert exc.value.code == 1


# ------------------------------------------------------------------
# recall
# ------------------------------------------------------------------


def test_recall_returns_results(store_path, capsys):
    main(["recall", store_path, "Ivan career"])
    out = capsys.readouterr().out
    assert "Ivan" in out
    assert "Recalling" in out


def test_recall_respects_k(store_path, capsys):
    main(["recall", store_path, "team work", "--k", "1"])
    out = capsys.readouterr().out
    # At most 1 result → only one result line (numbered "1.")
    assert "  1." in out
    assert "  2." not in out


def test_recall_spreading_mode(store_path, capsys):
    main(["recall", store_path, "Ivan", "--mode", "spreading"])
    out = capsys.readouterr().out
    assert "mode=spreading" in out


def test_recall_empty_store_shows_no_results(empty_path, capsys):
    main(["recall", empty_path, "anything"])
    out = capsys.readouterr().out
    assert "no results" in out


def test_recall_as_of_flag(store_path, capsys):
    main(["recall", store_path, "Ivan", "--as-of", "2099-01-01"])
    out = capsys.readouterr().out
    assert "as-of=2099-01-01" in out


def test_recall_invalid_as_of_exits(store_path):
    with pytest.raises(SystemExit) as exc:
        main(["recall", store_path, "query", "--as-of", "not-a-date"])
    assert exc.value.code == 1


# ------------------------------------------------------------------
# timeline
# ------------------------------------------------------------------


def test_timeline_shows_facts(store_path, capsys):
    main(["timeline", store_path, "Ivan"])
    out = capsys.readouterr().out
    assert "Ivan" in out
    assert "works_at" in out
    assert "Globex" in out


def test_timeline_unknown_entity_shows_no_facts(store_path, capsys):
    main(["timeline", store_path, "Nobody"])
    out = capsys.readouterr().out
    assert "no facts" in out


def test_timeline_shows_all_facts_for_entity(store_path, capsys):
    main(["timeline", store_path, "Ivan"])
    out = capsys.readouterr().out
    assert "lives_in" in out


# ------------------------------------------------------------------
# observe
# ------------------------------------------------------------------


def test_observe_adds_episode(tmp_path, capsys):
    path = str(tmp_path / "obs.engram")
    with Engram(path=path):
        pass
    main(["observe", path, "Maria reviewed the code"])
    out = capsys.readouterr().out
    assert "Observed:" in out
    # Verify it's actually stored
    with Engram(path=path) as mem:
        results = mem.recall("Maria code", k=3)
    assert any("Maria" in r.episode.content for r in results)


def test_observe_with_actors_and_tags(tmp_path, capsys):
    path = str(tmp_path / "obs.engram")
    with Engram(path=path):
        pass
    main(["observe", path, "Bob fixed the bug", "--actors", "Bob", "--tags", "bugfix", "backend"])
    out = capsys.readouterr().out
    assert "actors: Bob" in out
    assert "tags: bugfix, backend" in out


def test_observe_salience_and_valence(tmp_path, capsys):
    path = str(tmp_path / "obs.engram")
    with Engram(path=path):
        pass
    main(["observe", path, "Great news", "--salience", "0.9", "--valence", "0.8"])
    out = capsys.readouterr().out
    assert "Observed:" in out


# ------------------------------------------------------------------
# reflect
# ------------------------------------------------------------------


def test_reflect_without_llm_runs_decay(store_path, capsys):
    main(["reflect", store_path])
    out = capsys.readouterr().out
    assert "Done in" in out
    assert "Episodes processed:" in out


def test_reflect_with_stub_via_internal_api(tmp_path, capsys):
    """Verify reflect writes a reflection run record."""
    path = str(tmp_path / "ref.engram")
    with Engram(path=path) as mem:
        mem.observe("Ivan works at Globex", actors=["Ivan"])
    main(["reflect", path])
    with Engram(path=path) as mem:
        run = mem._store.get_last_reflection()
    assert run is not None
    assert run.finished_at is not None


# ------------------------------------------------------------------
# forget
# ------------------------------------------------------------------


def test_forget_episode_via_cli(store_path, capsys):
    with Engram(path=store_path) as mem:
        ep_id = mem.observe("Temporary sensitive info")
    main(["forget", store_path, "--episode", ep_id])
    out = capsys.readouterr().out
    assert "Erased episode" in out
    assert ep_id in out
    # Confirm gone
    with Engram(path=store_path) as mem:
        assert mem._store.get_episode(ep_id) is None


def test_forget_episode_not_found_exits(store_path):
    with pytest.raises(SystemExit) as exc:
        main(["forget", store_path, "--episode", "nonexistent-uuid"])
    assert exc.value.code == 1


def test_forget_entity_via_cli(store_path, capsys):
    main(["forget", store_path, "--entity", "Ivan"])
    out = capsys.readouterr().out
    assert 'Erased entity "Ivan"' in out
    assert "Episodes deleted:" in out
    assert "Facts deleted:" in out


def test_forget_both_flags_exits(store_path):
    with pytest.raises(SystemExit) as exc:
        main(["forget", store_path, "--episode", "abc", "--entity", "Ivan"])
    assert exc.value.code == 1


def test_forget_no_flags_exits(store_path):
    with pytest.raises(SystemExit) as exc:
        main(["forget", store_path])
    assert exc.value.code == 1


# ------------------------------------------------------------------
# version and help
# ------------------------------------------------------------------


def test_version_flag(capsys):
    from engram import __version__

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_no_command_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1


def test_help_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "inspect" in out
    assert "recall" in out
    assert "timeline" in out
    assert "reflect" in out
    assert "forget" in out
