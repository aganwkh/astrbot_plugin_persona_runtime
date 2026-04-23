from __future__ import annotations

import unittest

from persona_runtime.tool_context_tracker import ToolContextTracker


class ToolContextTrackerTests(unittest.TestCase):
    def test_tool_chain_is_associated_with_same_turn(self):
        tracker = ToolContextTracker()

        tracker.on_agent_begin("turn-1")
        tracker.on_tool_begin("turn-1", "lookup", {"q": "abc"})
        tracker.on_tool_result("turn-1", "lookup", ["result"])

        trace = tracker.get("turn-1")
        self.assertIsNotNone(trace)
        self.assertTrue(trace.agent_active)
        self.assertTrue(trace.tool_called)
        self.assertEqual(trace.tool_name, "lookup")
        self.assertTrue(trace.tool_success)
        self.assertEqual(trace.tool_call_count, 1)
        self.assertEqual(trace.tool_result_count, 1)
        self.assertEqual(trace.calls[0]["args"], {"q": "abc"})
        self.assertTrue(trace.calls[0]["success"])

    def test_different_turns_do_not_cross_wire(self):
        tracker = ToolContextTracker()

        tracker.on_tool_begin("turn-1", "lookup", {"q": "one"})
        tracker.on_tool_begin("turn-2", "search", {"q": "two"})
        tracker.on_tool_result("turn-2", "search", [])

        first = tracker.get("turn-1")
        second = tracker.get("turn-2")

        self.assertEqual(first.tool_name, "lookup")
        self.assertIsNone(first.tool_success)
        self.assertEqual(second.tool_name, "search")
        self.assertFalse(second.tool_success)
        self.assertTrue(second.tool_result_empty)


if __name__ == "__main__":
    unittest.main()
