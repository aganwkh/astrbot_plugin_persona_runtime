from __future__ import annotations

import asyncio
import unittest

from persona_runtime.scope_lock_manager import ScopeLockManager


class ScopeLockManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_guard_creates_locks_in_fixed_user_then_session_order(self):
        manager = ScopeLockManager()

        async with manager.guard("user-1", "session-1"):
            self.assertEqual(list(manager._locks), ["user:user-1", "session:session-1"])

    async def test_concurrent_access_does_not_deadlock(self):
        manager = ScopeLockManager()
        entered: list[str] = []

        async def worker(name: str, user: str, session: str):
            async with manager.guard(user, session):
                entered.append(name)
                await asyncio.sleep(0.01)

        await asyncio.wait_for(
            asyncio.gather(
                worker("same-session-a", "user-a", "room-1"),
                worker("same-session-b", "user-b", "room-1"),
                worker("cross-session-a", "user-c", "room-2"),
                worker("cross-session-b", "user-c", "room-3"),
            ),
            timeout=1,
        )

        self.assertCountEqual(
            entered,
            ["same-session-a", "same-session-b", "cross-session-a", "cross-session-b"],
        )


if __name__ == "__main__":
    unittest.main()
