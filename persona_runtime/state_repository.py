from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiosqlite

from .models import ExampleRecord, LearningBufferItem, RawTurnRecord, SessionState, TurnTraceRecord, UserState, WeightPatchRecord


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
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS examples (
                example_id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id TEXT NOT NULL,
                scene TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_reply TEXT NOT NULL,
                source TEXT NOT NULL,
                quality_score REAL NOT NULL DEFAULT 1.0,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL
            )
            """
        )
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_turns (
                turn_id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                user_message TEXT NOT NULL,
                assistant_reply TEXT NOT NULL,
                selected_policy_json TEXT NOT NULL,
                final_committed INTEGER NOT NULL
            )
            """
        )
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turn_traces (
                turn_id TEXT PRIMARY KEY,
                scene TEXT NOT NULL,
                selected_behavior TEXT NOT NULL,
                behavior_probabilities_json TEXT NOT NULL,
                feedback_label TEXT NULL,
                scene_scores_json TEXT NOT NULL,
                selected_example_ids_json TEXT NOT NULL,
                selected_lore_ids_json TEXT NOT NULL
            )
            """
        )
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_buffer (
                buffer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id TEXT NOT NULL,
                user_excerpt TEXT NOT NULL,
                assistant_excerpt TEXT NOT NULL,
                scene TEXT NOT NULL,
                selected_behavior TEXT NOT NULL,
                feedback_label TEXT NULL,
                content_mode TEXT NOT NULL,
                learning_eligible INTEGER NOT NULL,
                analyzed_at INTEGER NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        await self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weight_patches (
                patch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patch_type TEXT NOT NULL,
                scene TEXT NOT NULL,
                target_key TEXT NOT NULL,
                delta REAL NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                decay_factor REAL NOT NULL DEFAULT 1.0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                reason TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL
            )
            """
        )
        await self._ensure_column("learning_buffer", "analyzed_at", "INTEGER NULL")
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

    async def create_example(self, record: ExampleRecord) -> ExampleRecord:
        assert self.conn is not None
        cursor = await self.conn.execute(
            """
            INSERT INTO examples(
                turn_id, scene, tags_json, user_message, assistant_reply, source, quality_score, enabled, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.turn_id,
                record.scene,
                json.dumps(record.tags, ensure_ascii=False),
                record.user_message,
                record.assistant_reply,
                record.source,
                record.quality_score,
                1 if record.enabled else 0,
                record.created_at,
            ),
        )
        await self.conn.commit()
        record.example_id = cursor.lastrowid
        await cursor.close()
        return record

    async def create_raw_turn(self, record: RawTurnRecord):
        assert self.conn is not None
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO raw_turns(turn_id, timestamp, user_message, assistant_reply, selected_policy_json, final_committed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.turn_id,
                record.timestamp,
                record.user_message,
                record.assistant_reply,
                json.dumps(record.selected_policy, ensure_ascii=False),
                1 if record.final_committed else 0,
            ),
        )
        await self.conn.commit()

    async def create_turn_trace(self, record: TurnTraceRecord):
        assert self.conn is not None
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO turn_traces(
                turn_id, scene, selected_behavior, behavior_probabilities_json, feedback_label,
                scene_scores_json, selected_example_ids_json, selected_lore_ids_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.turn_id,
                record.scene,
                record.selected_behavior,
                json.dumps(record.behavior_probabilities, ensure_ascii=False),
                record.feedback_label,
                json.dumps(record.scene_scores, ensure_ascii=False),
                json.dumps(record.selected_example_ids, ensure_ascii=False),
                json.dumps(record.selected_lore_ids, ensure_ascii=False),
            ),
        )
        await self.conn.commit()

    async def create_learning_buffer_item(self, item: LearningBufferItem):
        assert self.conn is not None
        cursor = await self.conn.execute(
            """
            INSERT INTO learning_buffer(
                turn_id, user_excerpt, assistant_excerpt, scene, selected_behavior,
                feedback_label, content_mode, learning_eligible, analyzed_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.turn_id,
                item.user_excerpt,
                item.assistant_excerpt,
                item.scene,
                item.selected_behavior,
                item.feedback_label,
                item.content_mode,
                1 if item.learning_eligible else 0,
                item.analyzed_at,
                item.created_at,
            ),
        )
        await self.conn.commit()
        item.buffer_id = cursor.lastrowid
        await cursor.close()

    async def create_weight_patch(self, record: WeightPatchRecord) -> WeightPatchRecord:
        assert self.conn is not None
        cursor = await self.conn.execute(
            """
            INSERT INTO weight_patches(
                patch_type, scene, target_key, delta, min_value, max_value,
                decay_factor, evidence_count, reason, metadata_json, active, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.patch_type,
                record.scene,
                record.target_key,
                record.delta,
                record.min_value,
                record.max_value,
                record.decay_factor,
                record.evidence_count,
                record.reason,
                json.dumps(record.metadata, ensure_ascii=False),
                1 if record.active else 0,
                record.created_at,
            ),
        )
        await self.conn.commit()
        record.patch_id = cursor.lastrowid
        await cursor.close()
        return record

    async def list_examples(self, enabled_only: bool = False, limit: int = 50) -> list[ExampleRecord]:
        assert self.conn is not None
        query = """
            SELECT example_id, turn_id, scene, tags_json, user_message, assistant_reply, source, quality_score, enabled, created_at
            FROM examples
        """
        params: list[Any] = []
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY example_id DESC LIMIT ?"
        params.append(limit)
        cursor = await self.conn.execute(query, tuple(params))
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_example(row) for row in rows]

    async def get_example(self, example_id: int) -> ExampleRecord | None:
        assert self.conn is not None
        cursor = await self.conn.execute(
            """
            SELECT example_id, turn_id, scene, tags_json, user_message, assistant_reply, source, quality_score, enabled, created_at
            FROM examples
            WHERE example_id = ?
            """,
            (example_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if not row:
            return None
        return self._row_to_example(row)

    async def update_example_metadata(self, example_id: int, scene: str | None = None, tags: list[str] | None = None) -> ExampleRecord | None:
        existing = await self.get_example(example_id)
        if existing is None:
            return None
        if scene is not None:
            existing.scene = scene
        if tags is not None:
            existing.tags = tags
        assert self.conn is not None
        await self.conn.execute(
            "UPDATE examples SET scene = ?, tags_json = ? WHERE example_id = ?",
            (existing.scene, json.dumps(existing.tags, ensure_ascii=False), example_id),
        )
        await self.conn.commit()
        return existing

    async def set_example_enabled(self, example_id: int, enabled: bool) -> ExampleRecord | None:
        existing = await self.get_example(example_id)
        if existing is None:
            return None
        existing.enabled = enabled
        assert self.conn is not None
        await self.conn.execute(
            "UPDATE examples SET enabled = ? WHERE example_id = ?",
            (1 if enabled else 0, example_id),
        )
        await self.conn.commit()
        return existing

    async def delete_example(self, example_id: int) -> bool:
        assert self.conn is not None
        cursor = await self.conn.execute("DELETE FROM examples WHERE example_id = ?", (example_id,))
        await self.conn.commit()
        changed = cursor.rowcount > 0
        await cursor.close()
        return changed

    async def get_raw_turn(self, turn_id: str) -> RawTurnRecord | None:
        assert self.conn is not None
        cursor = await self.conn.execute(
            """
            SELECT turn_id, timestamp, user_message, assistant_reply, selected_policy_json, final_committed
            FROM raw_turns
            WHERE turn_id = ?
            """,
            (turn_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if not row:
            return None
        return RawTurnRecord(
            turn_id=str(row[0]),
            timestamp=int(row[1]),
            user_message=str(row[2]),
            assistant_reply=str(row[3]),
            selected_policy=json.loads(row[4]) if row[4] else {},
            final_committed=bool(row[5]),
        )

    async def get_turn_trace(self, turn_id: str) -> TurnTraceRecord | None:
        assert self.conn is not None
        cursor = await self.conn.execute(
            """
            SELECT turn_id, scene, selected_behavior, behavior_probabilities_json, feedback_label,
                   scene_scores_json, selected_example_ids_json, selected_lore_ids_json
            FROM turn_traces
            WHERE turn_id = ?
            """,
            (turn_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if not row:
            return None
        return TurnTraceRecord(
            turn_id=str(row[0]),
            scene=str(row[1]),
            selected_behavior=str(row[2]),
            behavior_probabilities=json.loads(row[3]) if row[3] else {},
            feedback_label=row[4],
            scene_scores=json.loads(row[5]) if row[5] else {},
            selected_example_ids=json.loads(row[6]) if row[6] else [],
            selected_lore_ids=json.loads(row[7]) if row[7] else [],
        )

    async def list_learning_buffer(self, limit: int = 50) -> list[LearningBufferItem]:
        assert self.conn is not None
        cursor = await self.conn.execute(
            """
            SELECT buffer_id, turn_id, user_excerpt, assistant_excerpt, scene, selected_behavior, feedback_label, content_mode, learning_eligible, analyzed_at, created_at
            FROM learning_buffer
            ORDER BY buffer_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            LearningBufferItem(
                turn_id=str(row[1]),
                user_excerpt=str(row[2]),
                assistant_excerpt=str(row[3]),
                scene=str(row[4]),
                selected_behavior=str(row[5]),
                feedback_label=row[6],
                content_mode=str(row[7]),
                learning_eligible=bool(row[8]),
                analyzed_at=int(row[9]) if row[9] is not None else None,
                created_at=int(row[10]),
                buffer_id=int(row[0]),
            )
            for row in rows
        ]

    async def list_pending_learning_buffer(self, limit: int = 50) -> list[LearningBufferItem]:
        assert self.conn is not None
        cursor = await self.conn.execute(
            """
            SELECT buffer_id, turn_id, user_excerpt, assistant_excerpt, scene, selected_behavior, feedback_label, content_mode, learning_eligible, analyzed_at, created_at
            FROM learning_buffer
            WHERE learning_eligible = 1 AND analyzed_at IS NULL
            ORDER BY buffer_id ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            LearningBufferItem(
                turn_id=str(row[1]),
                user_excerpt=str(row[2]),
                assistant_excerpt=str(row[3]),
                scene=str(row[4]),
                selected_behavior=str(row[5]),
                feedback_label=row[6],
                content_mode=str(row[7]),
                learning_eligible=bool(row[8]),
                analyzed_at=int(row[9]) if row[9] is not None else None,
                created_at=int(row[10]),
                buffer_id=int(row[0]),
            )
            for row in rows
        ]

    async def mark_learning_buffer_analyzed(self, buffer_ids: list[int], analyzed_at: int | None = None):
        assert self.conn is not None
        ids = [int(buffer_id) for buffer_id in buffer_ids if buffer_id is not None]
        if not ids:
            return
        timestamp = analyzed_at if analyzed_at is not None else int(time.time())
        placeholders = ", ".join("?" for _ in ids)
        await self.conn.execute(
            f"UPDATE learning_buffer SET analyzed_at = ? WHERE buffer_id IN ({placeholders})",
            (timestamp, *ids),
        )
        await self.conn.commit()

    async def list_weight_patches(
        self,
        *,
        patch_type: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[WeightPatchRecord]:
        assert self.conn is not None
        query = """
            SELECT patch_id, patch_type, scene, target_key, delta, min_value, max_value,
                   decay_factor, evidence_count, reason, metadata_json, active, created_at
            FROM weight_patches
        """
        clauses: list[str] = []
        params: list[Any] = []
        if patch_type is not None:
            clauses.append("patch_type = ?")
            params.append(patch_type)
        if active_only:
            clauses.append("active = 1")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY patch_id DESC LIMIT ?"
        params.append(limit)
        cursor = await self.conn.execute(query, tuple(params))
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_weight_patch(row) for row in rows]

    async def list_examples_by_ids(self, example_ids: list[int]) -> list[ExampleRecord]:
        assert self.conn is not None
        ids = [int(example_id) for example_id in example_ids]
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        cursor = await self.conn.execute(
            f"""
            SELECT example_id, turn_id, scene, tags_json, user_message, assistant_reply, source, quality_score, enabled, created_at
            FROM examples
            WHERE example_id IN ({placeholders})
            """,
            tuple(ids),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_example(row) for row in rows]

    @staticmethod
    def _row_to_example(row: tuple[Any, ...]) -> ExampleRecord:
        return ExampleRecord(
            example_id=int(row[0]),
            turn_id=str(row[1]),
            scene=str(row[2]),
            tags=json.loads(row[3]) if row[3] else [],
            user_message=str(row[4]),
            assistant_reply=str(row[5]),
            source=str(row[6]),
            quality_score=float(row[7]),
            enabled=bool(row[8]),
            created_at=int(row[9]),
        )

    @staticmethod
    def _row_to_weight_patch(row: tuple[Any, ...]) -> WeightPatchRecord:
        return WeightPatchRecord(
            patch_id=int(row[0]),
            patch_type=str(row[1]),
            scene=str(row[2]),
            target_key=str(row[3]),
            delta=float(row[4]),
            min_value=float(row[5]),
            max_value=float(row[6]),
            decay_factor=float(row[7]),
            evidence_count=int(row[8]),
            reason=str(row[9]),
            metadata=json.loads(row[10]) if row[10] else {},
            active=bool(row[11]),
            created_at=int(row[12]),
        )

    async def _ensure_column(self, table_name: str, column_name: str, column_definition: str):
        assert self.conn is not None
        cursor = await self.conn.execute(f"PRAGMA table_info({table_name})")
        rows = await cursor.fetchall()
        await cursor.close()
        existing = {str(row[1]) for row in rows}
        if column_name in existing:
            return
        await self.conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
