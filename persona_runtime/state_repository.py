from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiosqlite

from .models import SessionState, UserState


class StateRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: aiosqlite.Connection | None = None

    async def init_db(self):
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_states (
                scope_key TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                state_version INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL
            )
            """
        )
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_states (
                scope_key TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                state_version INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL
            )
            """
        )
        await self.conn.commit()

    async def close(self):
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    async def get_user_state(self, scope_key: str) -> UserState:
        assert self.conn is not None
        cursor = await self.conn.execute("SELECT data_json FROM user_states WHERE scope_key = ?", (scope_key,))
        row = await cursor.fetchone()
        await cursor.close()
        if not row:
            return UserState(scope_key=scope_key)
        data = json.loads(row[0])
        return UserState(**data)

    async def get_session_state(self, scope_key: str) -> SessionState:
        assert self.conn is not None
        cursor = await self.conn.execute("SELECT data_json FROM session_states WHERE scope_key = ?", (scope_key,))
        row = await cursor.fetchone()
        await cursor.close()
        if not row:
            return SessionState(scope_key=scope_key)
        data = json.loads(row[0])
        return SessionState(**data)

    async def upsert_user_state(self, state: UserState):
        assert self.conn is not None
        state.updated_at = int(time.time())
        await self.conn.execute(
            """
            INSERT INTO user_states(scope_key, data_json, state_version, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope_key) DO UPDATE SET
              data_json=excluded.data_json,
              state_version=excluded.state_version,
              updated_at=excluded.updated_at
            """,
            (state.scope_key, json.dumps(asdict(state), ensure_ascii=False), state.state_version, state.updated_at),
        )
        await self.conn.commit()

    async def upsert_session_state(self, state: SessionState):
        assert self.conn is not None
        state.updated_at = int(time.time())
        await self.conn.execute(
            """
            INSERT INTO session_states(scope_key, data_json, state_version, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(scope_key) DO UPDATE SET
              data_json=excluded.data_json,
              state_version=excluded.state_version,
              updated_at=excluded.updated_at
            """,
            (state.scope_key, json.dumps(asdict(state), ensure_ascii=False), state.state_version, state.updated_at),
        )
        await self.conn.commit()

    async def final_commit_states(
        self,
        user_state: UserState,
        session_state: SessionState,
        user_updates: dict[str, Any],
        session_updates: dict[str, Any],
    ):
        assert self.conn is not None
        await self.conn.execute("BEGIN IMMEDIATE")
        try:
            db_user = await self.get_user_state(user_state.scope_key)
            db_session = await self.get_session_state(session_state.scope_key)

            if db_user.state_version != user_state.state_version:
                raise RuntimeError("user_state_version_conflict")
            if db_session.state_version != session_state.state_version:
                raise RuntimeError("session_state_version_conflict")

            for k, v in user_updates.items():
                setattr(user_state, k, v)
            for k, v in session_updates.items():
                setattr(session_state, k, v)

            user_state.state_version += 1
            session_state.state_version += 1
            user_state.updated_at = int(time.time())
            session_state.updated_at = int(time.time())

            await self.conn.execute(
                "REPLACE INTO user_states(scope_key, data_json, state_version, updated_at) VALUES (?, ?, ?, ?)",
                (user_state.scope_key, json.dumps(asdict(user_state), ensure_ascii=False), user_state.state_version, user_state.updated_at),
            )
            await self.conn.execute(
                "REPLACE INTO session_states(scope_key, data_json, state_version, updated_at) VALUES (?, ?, ?, ?)",
                (session_state.scope_key, json.dumps(asdict(session_state), ensure_ascii=False), session_state.state_version, session_state.updated_at),
            )
            await self.conn.commit()
        except Exception:
            await self.conn.rollback()
            raise
