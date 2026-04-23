from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from persona_runtime.models import SessionState, UserState
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


if __name__ == "__main__":
    unittest.main()
