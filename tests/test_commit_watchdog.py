from __future__ import annotations

import asyncio
import unittest

from persona_runtime.commit_watchdog import CommitWatchdog
from persona_runtime.observability_logger import ObservabilityLogger
from persona_runtime.turn_registry import TurnRegistry


class CommitWatchdogTests(unittest.IsolatedAsyncioTestCase):
    async def test_rolls_back_when_after_message_sent_is_missing(self):
        registry = TurnRegistry(default_timeout_seconds=0)
        rolled_back: list[tuple[str, str]] = []

        async def on_timeout(turn_id: str, reason: str):
            rolled_back.append((turn_id, reason))
            registry.mark_aborted(turn_id, reason)

        turn = registry.create_turn("user:1", "session:1")
        watchdog = CommitWatchdog(
            turn_registry=registry,
            timeout_seconds=0,
            on_timeout=on_timeout,
            logger=ObservabilityLogger(enabled=False),
        )

        await watchdog.start()
        try:
            await asyncio.sleep(1.1)
        finally:
            await watchdog.stop()

        self.assertEqual(rolled_back, [(turn.turn_id, "commit_timeout")])
        self.assertTrue(registry.get(turn.turn_id).aborted)

    async def test_final_committed_turn_is_not_rolled_back_again(self):
        registry = TurnRegistry(default_timeout_seconds=0)
        rolled_back: list[tuple[str, str]] = []

        async def on_timeout(turn_id: str, reason: str):
            rolled_back.append((turn_id, reason))
            registry.mark_aborted(turn_id, reason)

        turn = registry.create_turn("user:1", "session:1")
        registry.mark_final_committed(turn.turn_id, "after_message_sent")
        watchdog = CommitWatchdog(
            turn_registry=registry,
            timeout_seconds=0,
            on_timeout=on_timeout,
            logger=ObservabilityLogger(enabled=False),
        )

        await watchdog.start()
        try:
            await asyncio.sleep(1.1)
        finally:
            await watchdog.stop()

        self.assertEqual(rolled_back, [])
        self.assertTrue(registry.get(turn.turn_id).final_committed)


if __name__ == "__main__":
    unittest.main()
