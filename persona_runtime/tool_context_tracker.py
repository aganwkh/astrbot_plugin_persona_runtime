from __future__ import annotations

import time
from typing import Any

from .models import ToolTrace


class ToolContextTracker:
    def __init__(self):
        self._traces: dict[str, ToolTrace] = {}

    def on_agent_begin(self, turn_id: str):
        trace = self._traces.setdefault(turn_id, ToolTrace(turn_id=turn_id))
        trace.agent_active = True

    def on_tool_begin(self, turn_id: str, tool_name: str, tool_args: dict[str, Any]):
        trace = self._traces.setdefault(turn_id, ToolTrace(turn_id=turn_id))
        trace.tool_called = True
        trace.tool_name = tool_name
        trace.tool_args = tool_args
        trace.tool_call_count += 1
        trace.calls.append(
            {
                "index": trace.tool_call_count,
                "tool_name": tool_name,
                "args": tool_args,
                "started_at": int(time.time()),
                "success": None,
            }
        )

    def on_tool_result(self, turn_id: str, tool_name: str, tool_result: Any):
        trace = self._traces.setdefault(turn_id, ToolTrace(turn_id=turn_id))
        trace.tool_result_count += 1
        trace.tool_name = tool_name
        success = tool_result is not None
        result_empty = self._is_empty_result(tool_result)
        trace.tool_success = success and not result_empty
        trace.tool_result_empty = result_empty
        trace.tool_error_flag = not success
        if trace.tool_error_flag:
            trace.tool_error_count += 1
        for call in reversed(trace.calls):
            if call.get("tool_name") == tool_name and call.get("success") is None:
                call["finished_at"] = int(time.time())
                call["success"] = trace.tool_success
                call["result_empty"] = result_empty
                break

    def get(self, turn_id: str) -> ToolTrace | None:
        return self._traces.get(turn_id)

    def pop(self, turn_id: str) -> ToolTrace | None:
        return self._traces.pop(turn_id, None)

    @staticmethod
    def _is_empty_result(tool_result: Any) -> bool:
        if tool_result is None:
            return True
        content = getattr(tool_result, "content", None)
        if content is not None:
            return len(content) == 0
        if isinstance(tool_result, (str, bytes, list, tuple, dict, set)):
            return len(tool_result) == 0
        return False
