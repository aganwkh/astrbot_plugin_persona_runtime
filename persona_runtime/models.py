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
class SceneResolution:
    main_scene: str
    scene_scores: dict[str, float]
    is_feedback: bool = False
    feedback_target: str = "none"
    feature_scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BehaviorProbabilityResult:
    scene: str
    behavior_probabilities: dict[str, float]
    reasons: dict[str, str] = field(default_factory=dict)
    applied_effects: list["LearningEffect"] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DialoguePolicyResult:
    selected_behavior: str = "short_ack"
    reply_length: str = "short"
    followup_intensity: str = "none"
    role_tone_strength: float = 0.65
    allow_followup_question: bool = False
    need_lore_injection: bool = False
    need_soften_tool_tone: bool = False
    need_task_mode: bool = False
    tone_mode: str = "flat"
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
class ExampleRecord:
    example_id: int | None = None
    turn_id: str = ""
    scene: str = "casual_chat"
    tags: list[str] = field(default_factory=list)
    user_message: str = ""
    assistant_reply: str = ""
    source: str = "manual_capture"
    quality_score: float = 1.0
    enabled: bool = True
    created_at: int = field(default_factory=_now_ts)

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "turn_id": self.turn_id,
            "scene": self.scene,
            "tags": self.tags,
            "enabled": self.enabled,
            "quality_score": self.quality_score,
            "user_chars": len(self.user_message),
            "assistant_chars": len(self.assistant_reply),
            "created_at": self.created_at,
        }


@dataclass
class ExampleSelection:
    example_id: int
    scene: str
    tags: list[str]
    user_message: str
    assistant_reply: str
    score: float
    reason: str
    applied_effects: list["LearningEffect"] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RawTurnRecord:
    turn_id: str
    timestamp: int = field(default_factory=_now_ts)
    user_message: str = ""
    assistant_reply: str = ""
    selected_policy: dict[str, Any] = field(default_factory=dict)
    final_committed: bool = True

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnTraceRecord:
    turn_id: str
    scene: str
    selected_behavior: str
    behavior_probabilities: dict[str, float]
    feedback_label: str | None = None
    scene_scores: dict[str, float] = field(default_factory=dict)
    selected_example_ids: list[int] = field(default_factory=list)
    selected_lore_ids: list[str] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LearningBufferItem:
    turn_id: str
    user_excerpt: str
    assistant_excerpt: str
    scene: str
    selected_behavior: str
    feedback_label: str | None = None
    content_mode: str = "full_short"
    learning_eligible: bool = True
    analyzed_at: int | None = None
    created_at: int = field(default_factory=_now_ts)
    buffer_id: int | None = None

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeightPatchRecord:
    patch_id: int | None = None
    patch_type: str = "behavior_weight_patch"
    scene: str = "casual_chat"
    target_key: str = ""
    delta: float = 0.0
    min_value: float = 0.05
    max_value: float = 0.95
    decay_factor: float = 1.0
    evidence_count: int = 0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: int = field(default_factory=_now_ts)

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LearningEffect:
    patch_id: int | None = None
    patch_type: str = ""
    scene: str = ""
    target_key: str = ""
    delta: float = 0.0
    reason: str = ""

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptModuleSelection:
    module_id: str
    title: str
    content: str
    score: float = 1.0
    reason: str = ""

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "title": self.title,
            "score": self.score,
            "reason": self.reason,
            "content_chars": len(self.content or ""),
        }


@dataclass
class TokenBudgetSummary:
    max_runtime_chars: int
    used_chars: int = 0
    trimmed: bool = False

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptPlan:
    main_scene: str
    scene_scores: dict[str, float]
    is_feedback: bool = False
    feedback_target: str = "none"
    behavior_probabilities: dict[str, float] = field(default_factory=dict)
    selected_behavior: str = "short_ack"
    reply_length: str = "short"
    followup_intensity: str = "none"
    role_tone_strength: float = 0.65
    current_scene_text: str = ""
    behavior_tendency_text: str = ""
    selected_examples: list[ExampleSelection] = field(default_factory=list)
    selected_modules: list[PromptModuleSelection] = field(default_factory=list)
    selected_lore_ids: list[str] = field(default_factory=list)
    learning_effects: list[LearningEffect] = field(default_factory=list)
    debug_reasons: list[str] = field(default_factory=list)
    token_budget: TokenBudgetSummary = field(default_factory=lambda: TokenBudgetSummary(max_runtime_chars=0))

    def selected_module_ids(self) -> list[str]:
        return [module.module_id for module in self.selected_modules]

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "main_scene": self.main_scene,
            "scene_scores": self.scene_scores,
            "is_feedback": self.is_feedback,
            "feedback_target": self.feedback_target,
            "behavior_probabilities": self.behavior_probabilities,
            "selected_behavior": self.selected_behavior,
            "reply_length": self.reply_length,
            "followup_intensity": self.followup_intensity,
            "role_tone_strength": self.role_tone_strength,
            "selected_examples": [example.to_log_dict() for example in self.selected_examples],
            "selected_modules": [module.to_log_dict() for module in self.selected_modules],
            "selected_lore_ids": self.selected_lore_ids,
            "learning_effects": [effect.to_log_dict() for effect in self.learning_effects],
            "token_budget": self.token_budget.to_log_dict(),
            "debug_reasons": self.debug_reasons,
        }


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
    scene_resolution: SceneResolution
    behavior_result: BehaviorProbabilityResult
    policy: DialoguePolicyResult
    prompt_plan: PromptPlan
    selected_lore_ids: list[str]
    injected_len: int
    assistant_reply_text: str = ""
    unified_msg_origin: str = ""


# Backward-compatible alias for starter-era references.
ResolutionResult = SceneResolution
