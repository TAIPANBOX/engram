"""CLI entry point for Engram.

Usage::

    engram inspect     <path> [--agent-id ID]
    engram recall      <path> <query> [--k K] [--mode MODE] [--as-of DATE]
                              [--agent-id ID] [--cross-agent]
    engram timeline    <path> <entity>
    engram observe     <path> <content> [--actors NAME...] [--tags TAG...]
                              [--salience F] [--valence F] [--agent-id ID]
    engram reflect     <path> [--llm anthropic|openai] [--model MODEL]
                              [--base-url URL] [--agent-id ID]
    engram forget      <path> (--episode ID | --entity NAME) [--agent-id ID]
    engram list-agents <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _open(path: str, agent_id: str | None = None, must_exist: bool = True) -> Any:
    """Open an Engram store, exiting on failure."""
    from engram import Engram

    p = Path(path)
    if must_exist and path != ":memory:" and not p.exists():
        _die(f"file not found: {path}")
    return Engram(path=path, agent_id=agent_id)


def _fmt_date(dt: Any) -> str:
    if dt is None:
        return "now"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _fmt_actors(actors: list[str]) -> str:
    return ", ".join(actors) if actors else "—"


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------


def _cmd_inspect(args: argparse.Namespace) -> None:
    import os

    with _open(args.path, agent_id=getattr(args, "agent_id", None)) as mem:
        store = mem._store
        ep = store.episode_count()
        vec = store.vec_count()
        facts_total = store.fact_count()
        facts_active = store.active_fact_count()
        entities = store.entity_count()
        runs = store.reflection_count()
        last_run = store.get_last_reflection()

    size_str = ""
    p = Path(args.path)
    if p.exists():
        size_bytes = os.path.getsize(p)
        if size_bytes >= 1_048_576:
            size_str = f"  ({size_bytes / 1_048_576:.1f} MB)"
        else:
            size_str = f"  ({size_bytes / 1024:.0f} KB)"

    print(f"\nStore: {args.path}{size_str}\n")
    print(f"  Episodes:       {ep:>6}   (vec index: {vec})")
    print(
        f"  Facts:          {facts_total:>6}   (active: {facts_active}, superseded: {facts_total - facts_active})"
    )
    print(f"  Entities:       {entities:>6}")
    if runs == 0:
        print(f"  Reflections:    {runs:>6}   (never run)")
    else:
        last_str = _fmt_date(last_run.started_at) if last_run else "—"
        print(f"  Reflections:    {runs:>6}   (last: {last_str})")
    print()


def _cmd_recall(args: argparse.Namespace) -> None:
    from datetime import UTC, datetime

    as_of = None
    if args.as_of:
        try:
            as_of = datetime.fromisoformat(args.as_of).replace(tzinfo=UTC)
        except ValueError:
            _die(f"invalid --as-of date: {args.as_of!r}  (expected ISO format, e.g. 2024-03-01)")

    with _open(args.path, agent_id=getattr(args, "agent_id", None)) as mem:
        results = mem.recall(
            args.query,
            k=args.k,
            mode=args.mode,
            as_of=as_of,
            cross_agent=getattr(args, "cross_agent", False),
        )

    mode_label = f"mode={args.mode}"
    if as_of:
        mode_label += f", as-of={args.as_of}"
    print(f'\nRecalling "{args.query}"  (k={args.k}, {mode_label})\n')

    if not results:
        print("  (no results)\n")
        return

    for i, r in enumerate(results, 1):
        ep = r.episode
        ts = _fmt_date(ep.timestamp)
        actors_str = f"actors: {_fmt_actors(ep.actors)}  |  " if ep.actors else ""
        print(f"  {i}. [{r.score:.3f}] {ep.content}")
        print(f"       {actors_str}{ts}  |  id: {ep.id}")
        if ep.tags:
            print(f"       tags: {', '.join(ep.tags)}")
        print()


def _cmd_timeline(args: argparse.Namespace) -> None:
    with _open(args.path, agent_id=getattr(args, "agent_id", None)) as mem:
        facts = mem.timeline(args.entity)

    print(f'\nFact timeline for "{args.entity}"\n')

    if not facts:
        print("  (no facts found)\n")
        return

    for f in facts:
        start = f.valid_from.strftime("%Y-%m-%d")
        end = f.valid_to.strftime("%Y-%m-%d") if f.valid_to else "now    "
        status = " ← superseded" if f.valid_to is not None else ""
        print(
            f"  [{start} → {end}]  "
            f"{f.subject} {f.predicate} {f.object}"
            f"  (conf: {f.confidence:.2f}){status}"
        )
    print()


def _cmd_observe(args: argparse.Namespace) -> None:
    with _open(args.path, agent_id=getattr(args, "agent_id", None), must_exist=False) as mem:
        ep_id = mem.observe(
            args.content,
            actors=args.actors or [],
            tags=args.tags or [],
            salience=args.salience,
            emotional_valence=args.valence,
        )

    print(f"\nObserved: {ep_id}")
    print(f'  "{args.content}"')
    if args.actors:
        print(f"  actors: {', '.join(args.actors)}")
    if args.tags:
        print(f"  tags: {', '.join(args.tags)}")
    print()


def _cmd_reflect(args: argparse.Namespace) -> None:
    import time

    llm = None
    if args.llm == "anthropic":
        from engram import AnthropicAdapter

        llm = AnthropicAdapter(model=args.model or "claude-haiku-4-5-20251001")
    elif args.llm == "openai":
        from engram import OpenAIAdapter

        kwargs: dict[str, Any] = {"model": args.model or "gpt-4o-mini"}
        if args.base_url:
            kwargs["base_url"] = args.base_url
        llm = OpenAIAdapter(**kwargs)
    elif args.llm:
        _die(f"unknown --llm value: {args.llm!r}  (choose: anthropic, openai)")

    print(f"\nRunning reflection on {args.path}…", flush=True)
    t0 = time.monotonic()

    with _open(args.path, agent_id=getattr(args, "agent_id", None)) as mem:
        if llm is not None:
            mem._llm = llm
        run = mem.reflect()

    elapsed = time.monotonic() - t0
    print(f"  Episodes processed:   {run.episodes_processed}")
    print(f"  Facts extracted:      {run.facts_extracted}")
    print(f"  Contradictions:       {run.contradictions_resolved}")
    if run.cost_tokens:
        print(f"  Tokens used:          {run.cost_tokens:,}")
    print(f"  Done in {elapsed:.1f}s\n")


def _cmd_list_agents(args: argparse.Namespace) -> None:
    with _open(args.path) as mem:
        agents = mem.list_agents()
    if not agents:
        print("\n  (no agents found — store has no agent-scoped episodes)\n")
        return
    print(f"\nAgents in {args.path}\n")
    for agent in agents:
        print(f"  {agent}")
    print()


def _cmd_forget(args: argparse.Namespace) -> None:
    if not args.episode and not args.entity:
        _die("specify --episode <id> or --entity <name>")
    if args.episode and args.entity:
        _die("--episode and --entity are mutually exclusive")

    with _open(args.path, agent_id=getattr(args, "agent_id", None)) as mem:
        if args.episode:
            try:
                mem.forget(args.episode)
            except KeyError:
                _die(f"episode not found: {args.episode!r}")
            print(f"\nErased episode {args.episode}\n")
        else:
            result = mem.forget_entity(args.entity)
            print(f'\nErased entity "{args.entity}"')
            print(f"  Episodes deleted: {result.episodes_deleted}")
            print(f"  Facts deleted:    {result.facts_deleted}\n")


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engram",
        description="Engram — cognitive memory layer for AI agents",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    def _add_agent_id(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--agent-id",
            default=None,
            metavar="ID",
            dest="agent_id",
            help=(
                "scope this operation to a named agent. Any string works as a "
                "scoping key; use an agent://<trust-domain>/<name> identifier "
                "if this store also writes an Agent Passport event log, whose "
                "schema requires that shape"
            ),
        )

    # inspect
    p_inspect = sub.add_parser("inspect", help="show store statistics")
    p_inspect.add_argument("path", help="path to .engram file")
    _add_agent_id(p_inspect)

    # recall
    p_recall = sub.add_parser("recall", help="search episodes by semantic query")
    p_recall.add_argument("path", help="path to .engram file")
    p_recall.add_argument("query", help="search query")
    p_recall.add_argument("--k", type=int, default=5, metavar="K", help="max results (default 5)")
    p_recall.add_argument(
        "--mode",
        default="hybrid",
        choices=["cosine", "spreading", "hybrid"],
        help="retrieval mode (default hybrid)",
    )
    p_recall.add_argument(
        "--as-of",
        default=None,
        metavar="DATE",
        help="time-travel: only episodes before this ISO date (e.g. 2024-03-01)",
    )
    _add_agent_id(p_recall)
    p_recall.add_argument(
        "--cross-agent",
        action="store_true",
        default=False,
        dest="cross_agent",
        help="search all agents' episodes (overrides --agent-id scope)",
    )

    # timeline
    p_tl = sub.add_parser("timeline", help="show fact history for an entity")
    p_tl.add_argument("path", help="path to .engram file")
    p_tl.add_argument("entity", help="entity name (e.g. Ivan)")

    # observe
    p_obs = sub.add_parser("observe", help="record a new observation")
    p_obs.add_argument("path", help="path to .engram file")
    p_obs.add_argument("content", help="observation text")
    p_obs.add_argument(
        "--actors",
        nargs="+",
        metavar="NAME",
        default=None,
        help="named entities involved",
    )
    p_obs.add_argument(
        "--tags",
        nargs="+",
        metavar="TAG",
        default=None,
        help="categorical labels",
    )
    p_obs.add_argument(
        "--salience",
        type=float,
        default=0.5,
        metavar="F",
        help="importance at encoding, 0-1 (default 0.5)",
    )
    p_obs.add_argument(
        "--valence",
        type=float,
        default=0.0,
        metavar="F",
        help="emotional valence, -1 to +1 (default 0.0)",
    )
    _add_agent_id(p_obs)

    # reflect
    p_ref = sub.add_parser("reflect", help="run the reflection loop")
    p_ref.add_argument("path", help="path to .engram file")
    p_ref.add_argument(
        "--llm",
        default=None,
        metavar="PROVIDER",
        help="LLM provider for fact extraction: anthropic, openai",
    )
    p_ref.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="model id (e.g. claude-haiku-4-5-20251001, gpt-4o-mini)",
    )
    p_ref.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="custom base URL for OpenAI-compatible endpoints (Ollama, etc.)",
    )
    _add_agent_id(p_ref)

    # forget
    p_forget = sub.add_parser("forget", help="erase an episode or entity")
    p_forget.add_argument("path", help="path to .engram file")
    p_forget.add_argument("--episode", default=None, metavar="ID", help="episode id to erase")
    p_forget.add_argument("--entity", default=None, metavar="NAME", help="entity name to erase")
    _add_agent_id(p_forget)

    # list-agents
    p_la = sub.add_parser("list-agents", help="list all agent ids in the store")
    p_la.add_argument("path", help="path to .engram file")

    return parser


def _get_version() -> str:
    try:
        from engram import __version__

        return __version__
    except Exception:
        return "unknown"


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

_HANDLERS = {
    "inspect": _cmd_inspect,
    "recall": _cmd_recall,
    "timeline": _cmd_timeline,
    "observe": _cmd_observe,
    "reflect": _cmd_reflect,
    "forget": _cmd_forget,
    "list-agents": _cmd_list_agents,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    _HANDLERS[args.cmd](args)


if __name__ == "__main__":
    main()
