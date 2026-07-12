"""Tests for v1.3 multi-agent shared memory."""

from __future__ import annotations

import pytest

from engram import Engram, StubLLMAdapter
from engram.cli import main

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def shared_path(tmp_path) -> str:
    return str(tmp_path / "shared.engram")


@pytest.fixture()
def two_agents(shared_path):
    """Return (agent_a, agent_b) writing to the same file."""
    a = Engram(path=shared_path, agent_id="agent-a")
    b = Engram(path=shared_path, agent_id="agent-b")
    yield a, b
    a.close()
    b.close()


# ------------------------------------------------------------------
# Basic isolation
# ------------------------------------------------------------------


def test_agents_write_to_same_file(two_agents, shared_path):
    a, b = two_agents
    a.observe("Alpha event", actors=["Alice"])
    b.observe("Beta event", actors=["Bob"])

    with Engram(path=shared_path) as global_mem:
        assert global_mem._store.episode_count() == 2


def test_recall_scoped_to_own_agent(two_agents):
    a, b = two_agents
    a.observe("Alice joined Globex", actors=["Alice"])
    b.observe("Bob left Acme", actors=["Bob"])

    a_results = a.recall("Alice Globex", k=10)
    assert all("Alice" in r.episode.content for r in a_results)
    assert not any("Bob" in r.episode.content for r in a_results)


def test_recall_does_not_see_other_agents_episodes(two_agents):
    a, b = two_agents
    a.observe("Top secret A information")
    b.observe("Top secret B information")

    a_results = a.recall("secret information", k=10)
    b_results = b.recall("secret information", k=10)

    a_ids = {r.episode.id for r in a_results}
    b_ids = {r.episode.id for r in b_results}
    assert a_ids.isdisjoint(b_ids)


def test_episode_count_scoped_per_agent(two_agents):
    a, b = two_agents
    a.observe("A1")
    a.observe("A2")
    b.observe("B1")

    assert a._store.episode_count() == 2
    assert b._store.episode_count() == 1


def test_episode_agent_id_stored_on_model(two_agents):
    a, _ = two_agents
    ep_id = a.observe("Agent A event")
    ep = a._store.get_episode(ep_id)
    assert ep is not None
    assert ep.agent_id == "agent-a"


# ------------------------------------------------------------------
# Cross-agent recall
# ------------------------------------------------------------------


def test_cross_agent_recall_sees_all_episodes(two_agents):
    a, b = two_agents
    a.observe("Alpha alpha alpha unique-a")
    b.observe("Beta beta beta unique-b")

    cross_results = a.recall("unique", k=10, cross_agent=True)
    contents = [r.episode.content for r in cross_results]
    assert any("unique-a" in c for c in contents)
    assert any("unique-b" in c for c in contents)


def test_cross_agent_false_is_default(two_agents):
    a, b = two_agents
    a.observe("Alpha alpha alpha distinct-a")
    b.observe("Beta beta beta distinct-b")

    normal_results = a.recall("distinct", k=10)
    contents = [r.episode.content for r in normal_results]
    assert not any("distinct-b" in c for c in contents)


def test_no_agent_id_sees_all_episodes(shared_path):
    with Engram(path=shared_path, agent_id="agent-a") as a:
        a.observe("Agent A says hi")
    with Engram(path=shared_path, agent_id="agent-b") as b:
        b.observe("Agent B says hello")

    with Engram(path=shared_path) as global_mem:
        results = global_mem.recall("says", k=10)
        assert len(results) == 2


# ------------------------------------------------------------------
# Shared facts
# ------------------------------------------------------------------


def test_facts_are_shared_across_agents(two_agents):
    a, b = two_agents
    a.assert_fact("Alice", "role", "CTO")

    # agent-b can see the fact via timeline (facts are global)
    facts = b.timeline("Alice")
    assert len(facts) == 1
    assert facts[0].object == "CTO"


def test_reflect_processes_only_own_episodes(shared_path):
    stub_a = StubLLMAdapter(
        facts=[{"subject": "Alice", "predicate": "role", "object": "CTO", "confidence": 0.9}]
    )
    stub_b = StubLLMAdapter(
        facts=[{"subject": "Bob", "predicate": "role", "object": "VP", "confidence": 0.9}]
    )
    with Engram(path=shared_path, agent_id="agent-a", llm=stub_a) as a:
        a.observe("Alice is the CTO", actors=["Alice"])
        run_a = a.reflect()
        assert run_a.episodes_processed == 1

    with Engram(path=shared_path, agent_id="agent-b", llm=stub_b) as b:
        b.observe("Bob is the VP", actors=["Bob"])
        run_b = b.reflect()
        # Only processes agent-b's new episode, not agent-a's
        assert run_b.episodes_processed == 1


def test_reflect_incremental_per_agent(shared_path):
    """Second reflect for agent-a should not re-process agent-b's episodes."""
    stub = StubLLMAdapter(facts=[])
    with Engram(path=shared_path, agent_id="agent-a", llm=stub) as a:
        a.observe("A1")
        run1 = a.reflect()
        assert run1.episodes_processed == 1

        a.observe("A2")
        run2 = a.reflect()
        assert run2.episodes_processed == 1  # only the new one

    with Engram(path=shared_path, agent_id="agent-b", llm=stub) as b:
        b.observe("B1")
        run_b = b.reflect()
        assert run_b.episodes_processed == 1  # only B1


# ------------------------------------------------------------------
# Graph edges are private per agent; entities stay shared
# ------------------------------------------------------------------


def test_entities_shared_but_edges_scoped_per_agent(shared_path):
    """Two agents share entity rows (semantic memory) but not episode edges, so
    spreading activation can't hop through another agent's episodes."""
    from datetime import UTC, datetime

    a = Engram(path=shared_path, agent_id="agent-a")
    b = Engram(path=shared_path, agent_id="agent-b")
    now = datetime.now(tz=UTC)

    ent_a = a._store.find_or_create_entity("Ivan", "person", now)
    ent_b = b._store.find_or_create_entity("Ivan", "person", now)
    assert ent_a.id == ent_b.id  # entity is shared across agents

    a._store.insert_edge("ep-a", ent_a.id, "mentions", weight=1.0, created_at=now)
    b._store.insert_edge("ep-b", ent_b.id, "mentions", weight=1.0, created_at=now)

    # Neighbors of the shared entity are scoped to the querying agent's edges.
    neighbors_a = {n for n, _ in a._store.get_neighbors(ent_a.id)}
    neighbors_b = {n for n, _ in b._store.get_neighbors(ent_b.id)}
    assert neighbors_a == {"ep-a"}
    assert neighbors_b == {"ep-b"}

    # An unscoped store still sees every edge.
    with Engram(path=shared_path) as global_mem:
        assert {n for n, _ in global_mem._store.get_neighbors(ent_a.id)} == {"ep-a", "ep-b"}

    a.close()
    b.close()


# ------------------------------------------------------------------
# Decay is scoped to the calling agent
# ------------------------------------------------------------------


def test_decay_does_not_touch_other_agents_scores(shared_path):
    """One agent's decay must not recompute another agent's importance scores."""
    with (
        Engram(path=shared_path, agent_id="agent-a") as a,
        Engram(path=shared_path, agent_id="agent-b") as b,
    ):
        a.observe("agent-a event")
        b.observe("agent-b event")
        b_ep_id = b._store.get_episodes_since(None)[0].id
        b._store.update_importance(b_ep_id, 0.42)  # sentinel value

        updated = a.decay()
        assert updated == 1  # only agent-a's single episode

        refreshed = b._store.get_episode(b_ep_id)
        assert refreshed is not None
        assert refreshed.importance_score == 0.42  # untouched by agent-a's decay


# ------------------------------------------------------------------
# list_agents()
# ------------------------------------------------------------------


def test_list_agents_returns_all_agent_ids(two_agents, shared_path):
    a, b = two_agents
    a.observe("A episode")
    b.observe("B episode")

    with Engram(path=shared_path) as global_mem:
        agents = global_mem.list_agents()

    assert "agent-a" in agents
    assert "agent-b" in agents


def test_list_agents_empty_when_no_scoped_episodes(shared_path):
    with Engram(path=shared_path) as mem:
        mem.observe("Unscoped episode")
        assert mem.list_agents() == []


def test_list_agents_excludes_unscoped(shared_path):
    with Engram(path=shared_path) as unscoped:
        unscoped.observe("Unscoped")
    with Engram(path=shared_path, agent_id="agent-x") as scoped:
        scoped.observe("Scoped")

    with Engram(path=shared_path) as mem:
        agents = mem.list_agents()
    assert agents == ["agent-x"]


# ------------------------------------------------------------------
# forget() respects agent scope
# ------------------------------------------------------------------


def test_prune_episodes_does_not_touch_other_agents(shared_path):
    """Reflection-driven prune in agent-a must not delete agent-b's low-importance episodes."""
    with Engram(path=shared_path, agent_id="agent-a") as a:
        a.observe("agent-a low-importance event")
        a._store.update_importance(a._store.get_episodes_since(None)[0].id, 0.01)
    with Engram(path=shared_path, agent_id="agent-b") as b:
        b.observe("agent-b low-importance event")
        b._store.update_importance(b._store.get_episodes_since(None)[0].id, 0.01)

    with Engram(path=shared_path, agent_id="agent-a") as a:
        pruned = a._store.prune_episodes(threshold=0.1)
        assert pruned == 1  # only agent-a's

    with Engram(path=shared_path, agent_id="agent-b") as b:
        assert b._store.episode_count() == 1  # agent-b's still there


def test_prune_episodes_cleans_fts_index(shared_path):
    """prune_episodes must remove pruned rows from the FTS index so they don't surface in hybrid recall."""
    with Engram(path=shared_path, agent_id="agent-a") as a:
        ep_id = a.observe("globally unique fts marker xyzzy")
        a._store.update_importance(ep_id, 0.01)
        pruned = a._store.prune_episodes(threshold=0.1)
        assert pruned == 1
        # Hybrid recall on the unique term must return nothing.
        results = a.recall("xyzzy", k=5, mode="hybrid")
        assert results == []


def test_forget_entity_erases_across_all_agents(two_agents):
    a, b = two_agents
    a.observe("Alice joined Globex", actors=["Alice"])
    b.observe("Alice reviewed the PR", actors=["Alice"])

    result = a.forget_entity("Alice")
    assert result.episodes_deleted == 2  # both agents' episodes gone


def test_forget_cannot_delete_another_agents_episode(two_agents):
    """Regression test for the cross-agent forget() bug: delete_episode()
    had no agent_id filter, so agent-a could permanently erase an episode
    that belongs to agent-b just by knowing/guessing its id. agent-a's
    forget() must now treat agent-b's episode_id as not found, and agent-b's
    episode must survive untouched.
    """
    a, b = two_agents
    b_ep_id = b.observe("agent-b private episode")

    with pytest.raises(KeyError):
        a.forget(b_ep_id)

    assert b._store.episode_count() == 1  # agent-b's episode still exists
    assert b._store.get_episode(b_ep_id) is not None


def test_delete_episode_scoped_to_agent_returns_false_for_other_agent(two_agents):
    """Store-level contract behind the test above: a scoped delete attempt
    against another agent's episode id reports "not found" (False) rather
    than deleting it or raising.
    """
    a, b = two_agents
    b_ep_id = b.observe("agent-b private episode")

    assert a._store.delete_episode(b_ep_id, agent_id="agent-a") is False
    assert b._store.get_episode(b_ep_id) is not None


def test_forget_own_episode_still_succeeds(two_agents):
    """Companion to the two tests above: scoping forget() to the caller's
    own agent must not prevent an agent from forgetting its OWN episodes."""
    _, b = two_agents
    b_ep_id = b.observe("agent-b episode to erase")

    b.forget(b_ep_id)  # must not raise

    assert b._store.episode_count() == 0
    assert b._store.get_episode(b_ep_id) is None


def test_get_episode_scoped_to_agent_returns_none_for_other_agent(two_agents):
    """Read-path counterpart: a scoped get_episode() must not return another
    agent's episode either (backs the MCP why() tool's agent scoping)."""
    a, b = two_agents
    b_ep_id = b.observe("agent-b private episode")

    assert a._store.get_episode(b_ep_id, agent_id="agent-a") is None
    assert b._store.get_episode(b_ep_id, agent_id="agent-b") is not None


# ------------------------------------------------------------------
# Backward compatibility — no agent_id
# ------------------------------------------------------------------


def test_no_agent_id_backward_compatible(tmp_path):
    path = str(tmp_path / "compat.engram")
    with Engram(path=path) as mem:
        ep_id = mem.observe("legacy observation")
        results = mem.recall("legacy", k=3)
        assert any(r.episode.id == ep_id for r in results)
        ep = mem._store.get_episode(ep_id)
        assert ep is not None
        assert ep.agent_id is None


def test_existing_store_migrates_without_error(tmp_path):
    """Opening a pre-v1.3 store (no agent_id column) must not raise."""
    import sqlite3

    path = str(tmp_path / "old.engram")
    # Simulate a pre-v1.3 store: create without agent_id column
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE episodes (id TEXT PRIMARY KEY, content TEXT NOT NULL, "
        "timestamp DATETIME NOT NULL, actors JSON DEFAULT '[]', tags JSON DEFAULT '[]', "
        "salience REAL DEFAULT 0.5, emotional_valence REAL DEFAULT 0.0, "
        "summary_of JSON DEFAULT '[]', importance_score REAL NOT NULL DEFAULT 1.0)"
    )
    conn.commit()
    conn.close()

    # Opening with current Engram must succeed (migration adds the column)
    with Engram(path=path) as mem:
        ep_id = mem.observe("Post-migration episode")
        results = mem.recall("migration", k=3)
        assert any(r.episode.id == ep_id for r in results)


# ------------------------------------------------------------------
# CLI: list-agents and --agent-id
# ------------------------------------------------------------------


def test_cli_list_agents(shared_path, capsys):
    with Engram(path=shared_path, agent_id="planner") as a:
        a.observe("Plan the roadmap")
    with Engram(path=shared_path, agent_id="coder") as b:
        b.observe("Write the code")

    main(["list-agents", shared_path])
    out = capsys.readouterr().out
    assert "planner" in out
    assert "coder" in out


def test_cli_list_agents_empty(shared_path, capsys):
    with Engram(path=shared_path) as mem:
        mem.observe("no agent")
    main(["list-agents", shared_path])
    out = capsys.readouterr().out
    assert "no agents found" in out


def test_cli_observe_with_agent_id(shared_path, capsys):
    main(["observe", shared_path, "Agent wrote this", "--agent-id", "bot-1"])
    capsys.readouterr()
    with Engram(path=shared_path) as mem:
        agents = mem.list_agents()
    assert "bot-1" in agents


def test_cli_recall_with_agent_id(shared_path, capsys):
    with Engram(path=shared_path, agent_id="bot-a") as a:
        a.observe("Bot A secret mission")
    with Engram(path=shared_path, agent_id="bot-b") as b:
        b.observe("Bot B classified report")

    main(["recall", shared_path, "mission report", "--agent-id", "bot-a"])
    out = capsys.readouterr().out
    assert "Bot A" in out
    assert "Bot B" not in out


def test_cli_recall_cross_agent(shared_path, capsys):
    with Engram(path=shared_path, agent_id="bot-a") as a:
        a.observe("Alpha unique observation")
    with Engram(path=shared_path, agent_id="bot-b") as b:
        b.observe("Beta unique observation")

    main(["recall", shared_path, "unique observation", "--cross-agent"])
    out = capsys.readouterr().out
    assert "Alpha" in out
    assert "Beta" in out
