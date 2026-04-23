from __future__ import annotations

import time
import unittest

from persona_runtime.turn_registry import TurnRegistry


class TurnRegistryTests(unittest.TestCase):
    def test_turn_id_creation(self):
        registry = TurnRegistry(default_timeout_seconds=15)

        first = registry.create_turn("user:1", "session:1")
        second = registry.create_turn("user:1", "session:1")

        self.assertTrue(first.turn_id)
        self.assertTrue(second.turn_id)
        self.assertNotEqual(first.turn_id, second.turn_id)
        self.assertEqual(first.user_scope_key, "user:1")
        self.assertGreaterEqual(first.deadline_at, int(time.time()))

    def test_pre_turn_written_and_final_committed_state_changes(self):
        registry = TurnRegistry()
        turn = registry.create_turn("user:1", "session:1")

        turn.pre_turn_written = True
        registry.mark_candidate(turn.turn_id, "on_llm_response")
        registry.mark_final_committed(turn.turn_id, "after_message_sent")

        stored = registry.get(turn.turn_id)
        self.assertIsNotNone(stored)
        self.assertTrue(stored.pre_turn_written)
        self.assertTrue(stored.final_committed)
        self.assertEqual(stored.commit_stage, "after_message_sent")
        self.assertGreater(stored.finished_at, 0)

    def test_prunes_finished_timeout_turns(self):
        registry = TurnRegistry()
        turn = registry.create_turn("user:1", "session:1")
        registry.mark_aborted(turn.turn_id, "commit_timeout")
        stored = registry.get(turn.turn_id)
        self.assertIsNotNone(stored)
        stored.finished_at = int(time.time()) - 301

        registry.prune_finished(older_than_seconds=300)

        self.assertIsNone(registry.get(turn.turn_id))


if __name__ == "__main__":
    unittest.main()
