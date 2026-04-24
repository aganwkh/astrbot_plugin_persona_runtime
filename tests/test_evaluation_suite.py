from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dev_harness import FakeEvent, build_plugin
from persona_runtime.models import WeightPatchRecord


class EvaluationSuiteTests(unittest.IsolatedAsyncioTestCase):
    async def test_preval_reports_all_cases_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin, _module = build_plugin(Path(tmp), timeout_seconds=1, extra_config={"deterministic_mode": True})
            try:
                await plugin._ensure_ready()
                await plugin.repository.create_weight_patch(
                    WeightPatchRecord(
                        patch_type="behavior_weight_patch",
                        scene="casual_chat",
                        target_key="followup_question",
                        delta=0.2,
                        evidence_count=3,
                        reason="live patch that should be ignored by eval",
                    )
                )
                event = FakeEvent(text="/preval", user_id="user-eval", session_id="room-eval")
                outputs = []
                async for item in plugin.preval(event):
                    outputs.append(item)
                output = "\n".join(outputs)

                self.assertIn("evaluation_cases: 4", output)
                self.assertIn("passed: 4", output)
                self.assertIn("failed: 0", output)
            finally:
                await plugin.terminate()


if __name__ == "__main__":
    unittest.main()
