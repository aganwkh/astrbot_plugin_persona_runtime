from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dev_harness import FakeEvent, build_plugin


class RecordChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_commit_persists_raw_turn_trace_and_learning_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin, module = build_plugin(Path(tmp), timeout_seconds=1)
            try:
                event = FakeEvent(text="刚吃完饭", user_id="user-1", session_id="room-1")
                req = module.ProviderRequest(system_prompt="")
                await plugin.on_llm_request(event, req)
                await plugin.on_llm_response(event, module.LLMResponse(text="嗯。那先坐一会儿吧。"))
                await plugin.after_message_sent(event)

                turn_id = plugin.turn_registry.find_latest_turn_id(event)
                self.assertIsNotNone(turn_id)

                raw_turn = await plugin.repository.get_raw_turn(turn_id)
                trace = await plugin.repository.get_turn_trace(turn_id)
                learning_items = await plugin.repository.list_learning_buffer(limit=10)

                self.assertIsNotNone(raw_turn)
                self.assertEqual(raw_turn.user_message, "刚吃完饭")
                self.assertEqual(raw_turn.assistant_reply, "嗯。那先坐一会儿吧。")

                self.assertIsNotNone(trace)
                self.assertEqual(trace.scene, "status_share")
                self.assertEqual(trace.selected_behavior, "short_ack")

                self.assertEqual(len(learning_items), 1)
                self.assertEqual(learning_items[0].turn_id, turn_id)
                self.assertEqual(learning_items[0].scene, "status_share")
            finally:
                await plugin.terminate()


if __name__ == "__main__":
    unittest.main()
