from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _now_ts() -> int:
    return int(time.time())


@dataclass
class NormalizedInput:
    raw_text: str
    is_empty: bool = False
    is_system: bool = False
    is_command_like: bool = False
    has_attachment_only: bool = False
    is_streaming: bool = False


@dataclass
class BypassDecision:
    bypass: bool
    reason: str = ""


@dataclass
class ScopeKeys:
    user_scope_key: str
    session_scope_key: str


@dataclass
class UserState:
    schema_version: int = 1
    scope_type: str = "user"
    scope_key: str = ""
    created_at: int = field(default_factory=_now_ts)
    updated_at: int = field(default_factory=_now_ts)
    dirty_flag: bool = False
    state_version: int = 0
    affinity: int = 0
    trust: int = 0
    relationship_stage: str = "neutral"
    recent_praise_decay: int = 0
    recent_doubt_decay: int = 0
    tool_style_bias: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class SessionState:
    schema_version: int = 1
    scope_type: str = "session"
    scope_key: str = ""
    created_at: int = field(default_factory=_now_ts)
    updated_at: int = field(default_factory=_now_ts)
    dirty_flag: bool = False
    state_version: int = 0
    task_mode: bool = False
    active_topic: str = ""
    last_tool_used: str = ""
    tool_chain_active: bool = False
    lore_cooldown: int = 0
    last_sent_success: bool = False
    turn_counter: int = 0
    repeat_tail_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ResolutionResult:
    feature_scores: dict[str, float]
    preliminary_mood: str
    preliminary_reply_mode: str
    pre_turn_flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialoguePolicyResult:
    reply_mode: str = "normal"
    allow_followup_question: bool = False
    tone_mode: str = "flat"
    need_lore_injection: bool = False
    need_soften_tool_tone: bool = False
    need_task_mode: bool = False
    max_followup_question_count: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoreItem:
    lore_id: str
    text: str
    score: float
    topic: str = ""


@dataclass
class MergedPromptPayload:
    policy_lines: list[str]
    lore_items: list[LoreItem]
    style_lines: list[str] = field(default_factory=list)


@dataclass
class ToolTrace:
    turn_id: str
    agent_active: bool = False
    tool_called: bool = False
    tool_name: str = ""
    tool_success: bool | None = None
    tool_result_empty: bool | None = None
    tool_error_flag: bool = False
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_call_count: int = 0
    tool_result_count: int = 0
    tool_error_count: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "agent_active": self.agent_active,
            "tool_called": self.tool_called,
            "tool_name": self.tool_name,
            "tool_success": self.tool_success,
            "tool_result_empty": self.tool_result_empty,
            "tool_error_flag": self.tool_error_flag,
            "tool_call_count": self.tool_call_count,
            "tool_result_count": self.tool_result_count,
            "tool_error_count": self.tool_error_count,
            "calls": self.calls,
        }


@dataclass
class TurnRecord:
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    user_scope_key: str = ""
    session_scope_key: str = ""
    created_at: int = field(default_factory=_now_ts)
    deadline_at: int = 0
    pre_turn_written: bool = False
    pre_user_state: Any = None
    pre_session_state: Any = None
    final_committed: bool = False
    commit_stage: str = "none"
    aborted: bool = False
    abort_reason: str = ""
    finished_at: int = 0
    bundle: Any = None


@dataclass
class RuntimeContextBundle:
    turn_id: str
    user_scope_key: str
    session_scope_key: str
    normalized_input: NormalizedInput
    feature_scores: dict[str, float]
    policy: DialoguePolicyResult
    selected_lore_ids: list[str]
    injected_len: int
    unified_msg_origin: str = ""
