from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dev_harness import FakeEvent, build_plugin


class PrexampleCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_can_capture_and_list_last_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin, module = build_plugin(Path(tmp), timeout_seconds=1)
            try:
                event = FakeEvent(text="刚吃完饭", user_id="user-1", session_id="room-1")
                req = module.ProviderRequest(system_prompt="")
                await plugin.on_llm_request(event, req)
                await plugin.on_llm_response(event, module.LLMResponse(text="嗯。那先坐一会儿吧。"))
                await plugin.after_message_sent(event)

                capture_event = FakeEvent(text="/prexample last scene=status_share tags=短句,不追问", user_id="user-1", session_id="room-1")
                capture_output = []
                async for item in plugin.prexample(capture_event):
                    capture_output.append(item)
                capture_text = "\n".join(capture_output)
                self.assertIn("saved_examples:", capture_text)
                self.assertIn("scene=status_share", capture_text)

                list_event = FakeEvent(text="/prexamples", user_id="user-1", session_id="room-1")
                list_output = []
                async for item in plugin.prexamples(list_event):
                    list_output.append(item)
                list_text = "\n".join(list_output)
                self.assertIn("#1", list_text)
                self.assertIn("刚吃完饭", list_text)
                self.assertIn("嗯。那先坐一会儿吧。", list_text)

                second_event = FakeEvent(text="刚吃完饭", user_id="user-1", session_id="room-1")
                second_req = module.ProviderRequest(system_prompt="")
                await plugin.on_llm_request(second_event, second_req)
                await plugin.on_llm_response(second_event, module.LLMResponse(text="好。那先休息一下。"))
                await plugin.after_message_sent(second_event)

                why_event = FakeEvent(text="/prwhy", user_id="user-1", session_id="room-1")
                why_output = []
                async for item in plugin.prwhy(why_event):
                    why_output.append(item)
                why_text = "\n".join(why_output)
                self.assertIn("selected_examples:", why_text)
                self.assertIn("#1", why_text)
            finally:
                await plugin.terminate()


if __name__ == "__main__":
    unittest.main()
