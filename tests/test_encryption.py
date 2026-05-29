"""Encryption-at-rest tests using SQLCipher.

Skipped automatically when sqlcipher3 is not installed.
Install prerequisites:
  macOS:  brew install sqlcipher && pip install 'engdbram[encryption]'
  Linux:  apt install libsqlcipher-dev && pip install 'engdbram[encryption]'
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlcipher3", reason="sqlcipher3 not installed")

from pathlib import Path

from engram import Engram


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "enc.engram")


def test_create_and_reopen(db_path: str) -> None:
    with Engram(path=db_path, key="secret") as mem:
        mem.observe("Alice is a data scientist")

    with Engram(path=db_path, key="secret") as mem:
        results = mem.recall("Alice")
    assert len(results) > 0


def test_wrong_key_raises(db_path: str) -> None:
    with Engram(path=db_path, key="correct") as mem:
        mem.observe("Bob works at Acme")

    with pytest.raises(Exception), Engram(path=db_path, key="wrong") as mem:  # noqa: B017
        mem.recall("Bob")


def test_plain_db_unchanged(tmp_path: Path) -> None:
    path = str(tmp_path / "plain.engram")
    with Engram(path=path) as mem:
        mem.observe("plain text episode")

    with Engram(path=path) as mem:
        results = mem.recall("plain")
    assert len(results) > 0


def test_backup_encrypted(tmp_path: Path) -> None:
    src = str(tmp_path / "src.engram")
    dst = str(tmp_path / "dst.engram")

    with Engram(path=src, key="pw") as mem:
        mem.observe("Charlie is CTO")
        mem.backup(dst)

    with Engram(path=dst, key="pw") as mem:
        results = mem.recall("Charlie")
    assert len(results) > 0


def test_rekey(db_path: str) -> None:
    with Engram(path=db_path, key="old") as mem:
        mem.observe("Dave is an engineer")
        mem.rekey("new")

    with Engram(path=db_path, key="new") as mem:
        results = mem.recall("Dave")
    assert len(results) > 0

    with pytest.raises(Exception), Engram(path=db_path, key="old") as mem:  # noqa: B017
        mem.recall("Dave")


def test_rekey_on_plain_raises(tmp_path: Path) -> None:
    path = str(tmp_path / "plain.engram")
    with Engram(path=path) as mem, pytest.raises(ValueError, match="rekey"):
        mem.rekey("newkey")


def test_memory_store_with_key() -> None:
    with Engram(path=":memory:", key="memkey") as mem:
        mem.observe("in-memory encrypted episode")
        results = mem.recall("in-memory")
    assert len(results) > 0
