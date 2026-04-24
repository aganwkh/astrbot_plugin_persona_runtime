from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from persona_runtime.models import (
    ExampleRecord,
    LearningBufferItem,
    RawTurnRecord,
    SessionState,
    TurnTraceRecord,
    UserState,
    WeightPatchRecord,
)
from persona_runtime.state_repository import StateRepository


class StateRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = StateRepository(Path(self.tmp.name) / "state.db")
        await self.repo.init_db()

    async def asyncTearDown(self):
        await self.repo.close()
        self.tmp.cleanup()

    async def test_initializes_database_and_returns_default_states(self):
        user = await self.repo.get_user_state("user:1")
        session = await self.repo.get_session_state("session:1")

        self.assertEqual(user.scope_key, "user:1")
        self.assertEqual(session.scope_key, "session:1")
        self.assertEqual(user.state_version, 0)
        self.assertEqual(session.state_version, 0)

    async def test_reads_and_writes_user_and_session_state(self):
        await self.repo.upsert_user_state(UserState(scope_key="user:1", affinity=3))
        await self.repo.upsert_session_state(SessionState(scope_key="session:1", turn_counter=7))

        user = await self.repo.get_user_state("user:1")
        session = await self.repo.get_session_state("session:1")

        self.assertEqual(user.affinity, 3)
        self.assertEqual(session.turn_counter, 7)

    async def test_compare_and_swap_success(self):
        await self.repo.upsert_user_state(UserState(scope_key="user:1"))
        await self.repo.upsert_session_state(SessionState(scope_key="session:1"))
        user = await self.repo.get_user_state("user:1")
        session = await self.repo.get_session_state("session:1")

        await self.repo.final_commit_states(
            user_state=user,
            session_state=session,
            user_updates={"trust": 2},
            session_updates={"last_sent_success": True},
        )

        stored_user = await self.repo.get_user_state("user:1")
        stored_session = await self.repo.get_session_state("session:1")
        self.assertEqual(stored_user.trust, 2)
        self.assertTrue(stored_session.last_sent_success)
        self.assertEqual(stored_user.state_version, 1)
        self.assertEqual(stored_session.state_version, 1)

    async def test_compare_and_swap_failure_rolls_back_transaction(self):
        await self.repo.upsert_user_state(UserState(scope_key="user:1"))
        await self.repo.upsert_session_state(SessionState(scope_key="session:1"))
        stale_user = await self.repo.get_user_state("user:1")
        stale_session = await self.repo.get_session_state("session:1")

        fresh_user = await self.repo.get_user_state("user:1")
        fresh_session = await self.repo.get_session_state("session:1")
        await self.repo.final_commit_states(
            user_state=fresh_user,
            session_state=fresh_session,
            user_updates={"affinity": 5},
            session_updates={"active_topic": "first"},
        )

        with self.assertRaises(RuntimeError):
            await self.repo.final_commit_states(
                user_state=stale_user,
                session_state=stale_session,
                user_updates={"affinity": 9},
                session_updates={"active_topic": "stale"},
            )

        stored_user = await self.repo.get_user_state("user:1")
        stored_session = await self.repo.get_session_state("session:1")
        self.assertEqual(stored_user.affinity, 5)
        self.assertEqual(stored_session.active_topic, "first")
        self.assertEqual(stored_user.state_version, 1)
        self.assertEqual(stored_session.state_version, 1)

    async def test_dual_scope_commit_does_not_half_succeed(self):
        await self.repo.upsert_user_state(UserState(scope_key="user:1"))
        await self.repo.upsert_session_state(SessionState(scope_key="session:1"))
        user = await self.repo.get_user_state("user:1")
        session = await self.repo.get_session_state("session:1")

        with self.assertRaises(TypeError):
            await self.repo.final_commit_states(
                user_state=user,
                session_state=session,
                user_updates={"affinity": 4},
                session_updates={"active_topic": object()},
            )

        stored_user = await self.repo.get_user_state("user:1")
        stored_session = await self.repo.get_session_state("session:1")
        self.assertEqual(stored_user.affinity, 0)
        self.assertEqual(stored_user.state_version, 0)
        self.assertEqual(stored_session.active_topic, "")
        self.assertEqual(stored_session.state_version, 0)

    async def test_create_and_manage_examples(self):
        created = await self.repo.create_example(
            ExampleRecord(
                turn_id="turn-1",
                scene="status_share",
                tags=["brief", "low_pressure"],
                user_message="just finished lunch",
                assistant_reply="nice, sit for a bit first.",
            )
        )
        self.assertIsNotNone(created.example_id)

        listed = await self.repo.list_examples(enabled_only=True, limit=10)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].scene, "status_share")

        updated = await self.repo.update_example_metadata(int(created.example_id), scene="casual_chat", tags=["natural"])
        self.assertIsNotNone(updated)
        self.assertEqual(updated.scene, "casual_chat")
        self.assertEqual(updated.tags, ["natural"])

        disabled = await self.repo.set_example_enabled(int(created.example_id), enabled=False)
        self.assertIsNotNone(disabled)
        self.assertFalse(disabled.enabled)
        enabled_only = await self.repo.list_examples(enabled_only=True, limit=10)
        self.assertEqual(enabled_only, [])

        deleted = await self.repo.delete_example(int(created.example_id))
        self.assertTrue(deleted)
        self.assertIsNone(await self.repo.get_example(int(created.example_id)))

    async def test_persists_raw_turn_trace_and_learning_buffer(self):
        await self.repo.create_raw_turn(
            RawTurnRecord(
                turn_id="turn-1",
                user_message="just finished lunch",
                assistant_reply="nice, sit for a bit first.",
                selected_policy={"selected_behavior": "short_ack"},
            )
        )
        await self.repo.create_turn_trace(
            TurnTraceRecord(
                turn_id="turn-1",
                scene="status_share",
                selected_behavior="short_ack",
                behavior_probabilities={"short_ack": 0.82},
                scene_scores={"status_share": 0.99},
                selected_example_ids=[1],
                selected_lore_ids=["mygo_base"],
            )
        )
        await self.repo.create_learning_buffer_item(
            LearningBufferItem(
                turn_id="turn-1",
                user_excerpt="just finished lunch",
                assistant_excerpt="nice, sit for a bit first.",
                scene="status_share",
                selected_behavior="short_ack",
            )
        )

        raw_turn = await self.repo.get_raw_turn("turn-1")
        trace = await self.repo.get_turn_trace("turn-1")
        learning_items = await self.repo.list_learning_buffer(limit=10)

        self.assertIsNotNone(raw_turn)
        self.assertEqual(raw_turn.assistant_reply, "nice, sit for a bit first.")
        self.assertIsNotNone(trace)
        self.assertEqual(trace.scene, "status_share")
        self.assertEqual(trace.selected_example_ids, [1])
        self.assertEqual(len(learning_items), 1)
        self.assertEqual(learning_items[0].turn_id, "turn-1")
        self.assertIsNotNone(learning_items[0].buffer_id)

    async def test_marks_learning_items_analyzed_and_persists_weight_patches(self):
        first = LearningBufferItem(
            turn_id="turn-1",
            user_excerpt="hello",
            assistant_excerpt="hey",
            scene="casual_chat",
            selected_behavior="short_ack",
        )
        second = LearningBufferItem(
            turn_id="turn-2",
            user_excerpt="hello again",
            assistant_excerpt="hey again",
            scene="casual_chat",
            selected_behavior="short_ack",
        )
        await self.repo.create_learning_buffer_item(first)
        await self.repo.create_learning_buffer_item(second)

        pending = await self.repo.list_pending_learning_buffer(limit=10)
        self.assertEqual(len(pending), 2)

        await self.repo.mark_learning_buffer_analyzed([item.buffer_id for item in pending if item.buffer_id is not None], analyzed_at=123456)
        pending_after = await self.repo.list_pending_learning_buffer(limit=10)
        all_items = await self.repo.list_learning_buffer(limit=10)

        self.assertEqual(pending_after, [])
        self.assertTrue(all(item.analyzed_at == 123456 for item in all_items))

        created_patch = await self.repo.create_weight_patch(
            WeightPatchRecord(
                patch_type="behavior_weight_patch",
                scene="casual_chat",
                target_key="short_ack",
                delta=0.04,
                evidence_count=2,
                reason="dominant behavior",
                metadata={"turn_ids": ["turn-1", "turn-2"]},
            )
        )
        listed_patches = await self.repo.list_weight_patches(limit=10)

        self.assertIsNotNone(created_patch.patch_id)
        self.assertEqual(len(listed_patches), 1)
        self.assertEqual(listed_patches[0].target_key, "short_ack")


if __name__ == "__main__":
    unittest.main()
