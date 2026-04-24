from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from .persona_runtime.batch_learning_analyzer import BatchLearningAnalyzer
from .persona_runtime.behavior_probability_engine import BehaviorProbabilityEngine
from .persona_runtime.bypass_router import BypassRouter
from .persona_runtime.commit_watchdog import CommitWatchdog
from .persona_runtime.dialogue_policy import DialoguePolicy
from .persona_runtime.eligibility_checker import EligibilityChecker
from .persona_runtime.evaluation_suite import EvaluationSuite
from .persona_runtime.example_selector import ExampleSelector
from .persona_runtime.fallback_controller import FallbackController
from .persona_runtime.input_normalizer import InputNormalizer
from .persona_runtime.learning_filter import LearningFilter
from .persona_runtime.lore_injector import LoreInjector
from .persona_runtime.models import ExampleRecord, NormalizedInput, PromptPlan, RawTurnRecord, RuntimeContextBundle, SessionState, ToolTrace, TurnTraceRecord, UserState
from .persona_runtime.observability_logger import ObservabilityLogger
from .persona_runtime.prompt_composer import PromptComposer
from .persona_runtime.prompt_merge_policy import PromptMergePolicy
from .persona_runtime.runtime_injection_guard import RuntimeInjectionGuard
from .persona_runtime.scope_lock_manager import ScopeLockManager
from .persona_runtime.state_decay import StateDecay
from .persona_runtime.state_repository import StateRepository
from .persona_runtime.state_resolver import StateResolver
from .persona_runtime.state_scope_resolver import StateScopeResolver
from .persona_runtime.token_budget_guard import TokenBudgetGuard
from .persona_runtime.tool_context_tracker import ToolContextTracker
from .persona_runtime.turn_registry import TurnRegistry
from .persona_runtime.utils import extract_response_text, safe_get_unified_msg_origin


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.input_normalizer = InputNormalizer()
        self.eligibility_checker = EligibilityChecker()
        self.bypass_router = BypassRouter()
        self.scope_resolver = StateScopeResolver()
        self.scope_locks = ScopeLockManager()
        self.state_decay = StateDecay()
        self.state_resolver = StateResolver()
        self.behavior_engine = BehaviorProbabilityEngine()
        self.dialogue_policy = DialoguePolicy()
        self.example_selector = ExampleSelector()
        self.learning_filter = LearningFilter()
        self.learning_analyzer = BatchLearningAnalyzer(
            min_batch_size=int(config.get("learning_min_batch_size", 3))
        )
        self.evaluation_suite = EvaluationSuite()
        self.lore_injector = LoreInjector(self.data_dir / "lore_keywords.json")
        self.prompt_merge_policy = PromptMergePolicy(static_persona_prompt=config.get("static_persona_prompt", ""))
        self.token_guard = TokenBudgetGuard(max_runtime_chars=int(config.get("max_runtime_chars", 800)))
        self.prompt_composer = PromptComposer()
        self.runtime_guard = RuntimeInjectionGuard()
        self.fallback = FallbackController()
        self.debug_mode = bool(config.get("DEBUG_MODE", True))
        self.deterministic_mode = bool(config.get("deterministic_mode", False))
        self.logger = ObservabilityLogger(enabled=self.debug_mode)
        commit_timeout_seconds = int(config.get("commit_timeout_seconds", 15))
        self.turn_registry = TurnRegistry(default_timeout_seconds=commit_timeout_seconds)
        self.tool_tracker = ToolContextTracker()
        self.repository = StateRepository(self.data_dir / "persona_runtime.db")
        self.commit_watchdog = CommitWatchdog(
            turn_registry=self.turn_registry,
            timeout_seconds=commit_timeout_seconds,
            on_timeout=self._rollback_turn,
            logger=self.logger,
        )

        self._ready = False

    async def _ensure_ready(self):
        if self._ready:
            return
        await self.repository.init_db()
        await self.commit_watchdog.start()
        self._ready = True
        self.logger.info("plugin_ready", message="persona_runtime initialized")

    async def terminate(self):
        await self.commit_watchdog.stop()
        await self.repository.close()
        self.logger.info("plugin_terminate", message="persona_runtime terminated")

    def _tool_flags(self, turn_id: str | None) -> dict[str, Any]:
        trace = self.tool_tracker.get(turn_id) if turn_id else None
        if trace:
            return trace.to_log_dict()
        return {
            "agent_active": False,
            "tool_called": False,
            "tool_name": "",
            "tool_success": None,
            "tool_result_empty": None,
            "tool_error_flag": False,
            "tool_call_count": 0,
            "tool_result_count": 0,
            "tool_error_count": 0,
        }

    def _debug_turn(
        self,
        debug_event: str,
        *,
        turn_id: str | None = None,
        turn: Any = None,
        bypass_reason: str = "",
        streaming_flag: bool | None = None,
        selected_policy: dict[str, Any] | None = None,
        selected_lore_ids: list[str] | None = None,
        tool_flags: dict[str, Any] | None = None,
        commit_stage: str | None = None,
        final_committed: bool | None = None,
        user_state_version: int | None = None,
        session_state_version: int | None = None,
        prompt_plan_summary: dict[str, Any] | None = None,
    ):
        if not self.debug_mode:
            return
        if turn is None and turn_id:
            turn = self.turn_registry.get(turn_id)
        if turn_id is None and turn is not None:
            turn_id = turn.turn_id

        bundle = getattr(turn, "bundle", None) if turn is not None else None
        if selected_policy is None and bundle is not None:
            selected_policy = bundle.policy.to_log_dict()
        if selected_lore_ids is None and bundle is not None:
            selected_lore_ids = bundle.selected_lore_ids
        if streaming_flag is None and bundle is not None:
            streaming_flag = bool(bundle.normalized_input.is_streaming)
        if commit_stage is None and turn is not None:
            commit_stage = turn.commit_stage
        if final_committed is None and turn is not None:
            final_committed = turn.final_committed
        if prompt_plan_summary is None and bundle is not None:
            prompt_plan_summary = bundle.prompt_plan.to_log_dict()

        self.logger.info(
            "debug_turn",
            debug_event=debug_event,
            turn_id=turn_id or "",
            user_scope_key=getattr(turn, "user_scope_key", ""),
            session_scope_key=getattr(turn, "session_scope_key", ""),
            bypass_reason=bypass_reason,
            streaming_flag=bool(streaming_flag),
            selected_policy=selected_policy or {},
            selected_lore_ids=selected_lore_ids or [],
            tool_flags=tool_flags if tool_flags is not None else self._tool_flags(turn_id),
            commit_stage=commit_stage or "none",
            final_committed=bool(final_committed),
            user_state_version=user_state_version,
            session_state_version=session_state_version,
            prompt_plan=prompt_plan_summary or {},
        )

    @filter.command("prhello")
    async def prhello(self, event: AstrMessageEvent):
        await self._ensure_ready()
        yield event.plain_result("人格运行层 starter 已加载。")

    @filter.command("prstate")
    async def prstate(self, event: AstrMessageEvent):
        await self._ensure_ready()
        scope = self.scope_resolver.resolve(event)
        user_state = await self.repository.get_user_state(scope.user_scope_key)
        session_state = await self.repository.get_session_state(scope.session_scope_key)
        text = (
            f"user_scope={scope.user_scope_key}\n"
            f"session_scope={scope.session_scope_key}\n"
            f"user_state={user_state.to_json()}\n"
            f"session_state={session_state.to_json()}"
        )
        yield event.plain_result(text)

    @filter.command("prturns")
    async def prturns(self, event: AstrMessageEvent):
        await self._ensure_ready()
        turns = self.turn_registry.snapshot()
        yield event.plain_result(str(turns))

    @filter.command("prexamples")
    async def prexamples(self, event: AstrMessageEvent):
        await self._ensure_ready()
        examples = await self.repository.list_examples(enabled_only=False, limit=20)
        if not examples:
            yield event.plain_result("No saved examples.")
            return
        lines = ["examples:"]
        for example in examples:
            tags = ",".join(example.tags) if example.tags else "-"
            status = "enabled" if example.enabled else "disabled"
            lines.append(
                f"- #{example.example_id} [{status}] scene={example.scene} tags={tags} user={example.user_message} reply={example.assistant_reply}"
            )
        yield event.plain_result("\n".join(lines))

    @filter.command("prexample")
    async def prexample(self, event: AstrMessageEvent):
        await self._ensure_ready()
        action, count, example_id, scene, tags = self._parse_prexample_command(getattr(event, "message_str", "") or "")
        if action == "last":
            created = await self._capture_last_examples(event, count=count, scene=scene, tags=tags)
            if not created:
                yield event.plain_result("No eligible committed turns with assistant replies were found.")
                return
            lines = ["saved_examples:"]
            for example in created:
                tag_text = ",".join(example.tags) if example.tags else "-"
                lines.append(f"- #{example.example_id} scene={example.scene} tags={tag_text}")
            yield event.plain_result("\n".join(lines))
            return

        if action == "tag" and example_id is not None:
            updated = await self.repository.update_example_metadata(example_id, scene=scene, tags=tags)
            if updated is None:
                yield event.plain_result(f"Example #{example_id} not found.")
                return
            tag_text = ",".join(updated.tags) if updated.tags else "-"
            yield event.plain_result(f"Updated example #{example_id}: scene={updated.scene} tags={tag_text}")
            return

        if action == "disable" and example_id is not None:
            updated = await self.repository.set_example_enabled(example_id, enabled=False)
            if updated is None:
                yield event.plain_result(f"Example #{example_id} not found.")
                return
            yield event.plain_result(f"Disabled example #{example_id}.")
            return

        if action == "delete" and example_id is not None:
            deleted = await self.repository.delete_example(example_id)
            if not deleted:
                yield event.plain_result(f"Example #{example_id} not found.")
                return
            yield event.plain_result(f"Deleted example #{example_id}.")
            return

        yield event.plain_result(
            "Usage: /prexample last [N] scene=... tags=a,b | /prexample tag <id> scene=... tags=a,b | /prexample disable <id> | /prexample delete <id>"
        )

    @filter.command("prwhy")
    async def prwhy(self, event: AstrMessageEvent):
        await self._ensure_ready()
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            yield event.plain_result("No prompt plan found for the current conversation.")
            return
        turn = self.turn_registry.get(turn_id)
        bundle = getattr(turn, "bundle", None) if turn else None
        if bundle is None:
            yield event.plain_result("No prompt plan found for the current conversation.")
            return
        yield event.plain_result(self._render_prompt_plan(bundle.prompt_plan))

    def _render_prompt_plan(self, plan: PromptPlan) -> str:
        lines = [
            f"main_scene: {plan.main_scene}",
            f"is_feedback: {str(plan.is_feedback).lower()}",
            f"feedback_target: {plan.feedback_target}",
            "",
            "scene_scores:",
        ]
        for scene, score in sorted(plan.scene_scores.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {scene}: {score:.3f}")
        lines.extend(
            [
                "",
                f"selected_behavior: {plan.selected_behavior}",
                "",
                "behavior_probabilities:",
            ]
        )
        for behavior, score in sorted(plan.behavior_probabilities.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {behavior}: {score:.3f}")
        if plan.learning_effects:
            lines.extend(["", "learning_effects:"])
            for effect in plan.learning_effects:
                patch_label = f"#{effect.patch_id}" if effect.patch_id is not None else "pending"
                lines.append(
                    f"- {patch_label} {effect.patch_type} target={effect.target_key} delta={effect.delta:+.3f} reason={effect.reason}"
                )
        if plan.selected_examples:
            lines.extend(["", "selected_examples:"])
            for example in plan.selected_examples:
                tag_text = ",".join(example.tags) if example.tags else "-"
                lines.append(f"- #{example.example_id} scene={example.scene} tags={tag_text} score={example.score:.3f} reason={example.reason}")
        lines.extend(["", "selected_modules:"])
        for module in plan.selected_modules:
            lines.append(f"- {module.module_id} score={module.score:.2f} reason={module.reason}")
        lines.extend(
            [
                "",
                "token_budget:",
                f"- max_runtime_chars: {plan.token_budget.max_runtime_chars}",
                f"- used_chars: {plan.token_budget.used_chars}",
                f"- trimmed: {str(plan.token_budget.trimmed).lower()}",
            ]
        )
        if plan.debug_reasons:
            lines.extend(["", "why_this_turn:"])
            for reason in plan.debug_reasons:
                lines.append(f"- {reason}")
        return "\n".join(lines)

    @filter.command("prlearn")
    async def prlearn(self, event: AstrMessageEvent):
        await self._ensure_ready()
        summary = await self._maybe_run_learning_analysis(force=True)
        if summary["pending_count"] == 0:
            yield event.plain_result("No pending learning items.")
            return
        lines = [
            f"analyzed_pending: {summary['pending_count']}",
            f"created_patches: {len(summary['created_patches'])}",
        ]
        if summary["created_patches"]:
            lines.append("patches:")
            for patch in summary["created_patches"]:
                lines.append(
                    f"- #{patch.patch_id} {patch.patch_type} scene={patch.scene} target={patch.target_key} delta={patch.delta:+.3f} evidence={patch.evidence_count}"
                )
        else:
            lines.append("patches: none")
        yield event.plain_result("\n".join(lines))

    @filter.command("prpatches")
    async def prpatches(self, event: AstrMessageEvent):
        await self._ensure_ready()
        patches = await self.repository.list_weight_patches(limit=20)
        if not patches:
            yield event.plain_result("No persisted patches.")
            return
        lines = ["patches:"]
        for patch in patches:
            status = "active" if patch.active else "inactive"
            lines.append(
                f"- #{patch.patch_id} [{status}] {patch.patch_type} scene={patch.scene} target={patch.target_key} delta={patch.delta:+.3f} evidence={patch.evidence_count} reason={patch.reason}"
            )
        yield event.plain_result("\n".join(lines))

    @filter.command("preval")
    async def preval(self, event: AstrMessageEvent):
        await self._ensure_ready()
        results = await self.evaluation_suite.run(self._build_eval_prompt_plan)
        passed_count = sum(1 for result in results if result.passed)
        lines = [
            f"deterministic_mode: {str(self.deterministic_mode).lower()}",
            f"evaluation_cases: {len(results)}",
            f"passed: {passed_count}",
            f"failed: {len(results) - passed_count}",
        ]
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(
                f"- [{status}] {result.case.case_id} scene={result.prompt_plan.main_scene} behavior={result.prompt_plan.selected_behavior}"
            )
            for reason in result.reasons:
                lines.append(f"  reason: {reason}")
        yield event.plain_result("\n".join(lines))

    async def _capture_last_examples(
        self,
        event: AstrMessageEvent,
        *,
        count: int,
        scene: str | None,
        tags: list[str] | None,
    ) -> list[ExampleRecord]:
        origin = safe_get_unified_msg_origin(event)
        recent_turns = self.turn_registry.list_recent_turns(origin, include_finished=True, limit=max(10, count + 4))
        captured: list[ExampleRecord] = []
        for turn in recent_turns:
            if len(captured) >= count:
                break
            if not turn.final_committed or turn.aborted:
                continue
            bundle = getattr(turn, "bundle", None)
            if bundle is None:
                continue
            if bundle.normalized_input.is_command_like:
                continue
            assistant_reply = (bundle.assistant_reply_text or "").strip()
            if not assistant_reply:
                continue
            captured.append(
                await self.repository.create_example(
                    ExampleRecord(
                        turn_id=turn.turn_id,
                        scene=scene or bundle.prompt_plan.main_scene,
                        tags=list(tags or []),
                        user_message=bundle.normalized_input.raw_text,
                        assistant_reply=assistant_reply,
                        source="manual_capture",
                    )
                )
            )
        return captured

    @staticmethod
    def _parse_prexample_command(raw_text: str) -> tuple[str, int, int | None, str | None, list[str] | None]:
        tokens = [token for token in raw_text.strip().split() if token]
        if len(tokens) < 2:
            return "", 1, None, None, None
        action = tokens[1].lower()
        count = 1
        example_id: int | None = None
        scene: str | None = None
        tags: list[str] | None = None
        index = 2
        if action == "last" and index < len(tokens) and tokens[index].isdigit():
            count = max(1, int(tokens[index]))
            index += 1
        elif action in {"tag", "disable", "delete"} and index < len(tokens) and tokens[index].isdigit():
            example_id = int(tokens[index])
            index += 1
        for token in tokens[index:]:
            if token.startswith("scene="):
                value = token.split("=", 1)[1].strip()
                scene = value or None
            elif token.startswith("tags="):
                value = token.split("=", 1)[1].strip()
                tags = [item.strip() for item in value.split(",") if item.strip()]
        return action, count, example_id, scene, tags

    async def _build_prompt_plan_bundle(
        self,
        normalized: NormalizedInput,
        user_state: UserState,
        session_state: SessionState,
        *,
        apply_learning_patches: bool = True,
    ) -> tuple[Any, Any, Any, list[Any], PromptPlan]:
        scene_resolution = self.state_resolver.resolve(normalized, user_state, session_state)
        behavior_patches = []
        example_tag_patches = []
        if apply_learning_patches:
            behavior_patches = await self.repository.list_weight_patches(
                patch_type="behavior_weight_patch",
                active_only=True,
                limit=100,
            )
            example_tag_patches = await self.repository.list_weight_patches(
                patch_type="example_tag_weight_patch",
                active_only=True,
                limit=100,
            )
        behavior_result = self.behavior_engine.build(normalized, scene_resolution, patches=behavior_patches)
        policy = self.dialogue_policy.build(normalized, scene_resolution, behavior_result)
        candidate_examples = await self.repository.list_examples(enabled_only=True, limit=50)
        selected_examples = self.example_selector.select(
            normalized,
            PromptPlan(main_scene=scene_resolution.main_scene, scene_scores=scene_resolution.scene_scores),
            candidate_examples,
            patches=example_tag_patches,
        )
        learning_effects = list(behavior_result.applied_effects)
        for example in selected_examples:
            learning_effects.extend(example.applied_effects)
        lore_items = self.lore_injector.pick(
            normalized,
            policy,
            int(self.config.get("lore_top_k", 2)),
            float(self.config.get("lore_score_threshold", 0.55)),
            int(self.config.get("max_lore_chars", 300)),
        )
        prompt_plan = self.prompt_merge_policy.merge(
            scene_resolution=scene_resolution,
            behavior_result=behavior_result,
            policy=policy,
            lore_items=lore_items,
            selected_examples=selected_examples,
            learning_effects=learning_effects,
            max_runtime_chars=int(self.config.get("max_runtime_chars", 800)),
        )
        prompt_plan = self.token_guard.apply(prompt_plan)
        return scene_resolution, behavior_result, policy, lore_items, prompt_plan

    async def _build_eval_prompt_plan(self, raw_text: str) -> PromptPlan:
        normalized = NormalizedInput(raw_text=raw_text)
        scene_resolution, behavior_result, policy, _lore_items, prompt_plan = await self._build_prompt_plan_bundle(
            normalized,
            UserState(scope_key="eval-user"),
            SessionState(scope_key="eval-session"),
            apply_learning_patches=False,
        )
        self.logger.info(
            "evaluation_case",
            raw_text=raw_text,
            scene_scores=scene_resolution.scene_scores,
            behavior_probabilities=behavior_result.behavior_probabilities,
            selected_policy=policy.to_log_dict(),
            prompt_plan=prompt_plan.to_log_dict(),
        )
        return prompt_plan

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        await self._ensure_ready()
        if not self.config.get("enable_runtime", True):
            return

        normalized = self.input_normalizer.normalize(event)
        if not self.eligibility_checker.is_eligible(normalized):
            self.logger.info("skip_ineligible", raw_text=normalized.raw_text)
            self._debug_turn("skip_ineligible", bypass_reason="ineligible", streaming_flag=normalized.is_streaming)
            return

        bypass = self.bypass_router.decide(normalized)
        if bypass.bypass:
            self.logger.info("bypass_runtime", reason=bypass.reason, raw_text=normalized.raw_text)
            self._debug_turn("bypass_runtime", bypass_reason=bypass.reason, streaming_flag=normalized.is_streaming)
            return

        scope = self.scope_resolver.resolve(event)
        turn = self.turn_registry.create_turn(scope.user_scope_key, scope.session_scope_key)
        self.logger.info(
            "turn_created",
            turn_id=turn.turn_id,
            user_scope_key=scope.user_scope_key,
            session_scope_key=scope.session_scope_key,
            streaming_flag=normalized.is_streaming,
        )
        self._debug_turn("turn_created", turn=turn, streaming_flag=normalized.is_streaming)

        try:
            async with self.scope_locks.guard(scope.user_scope_key, scope.session_scope_key):
                user_state = await self.repository.get_user_state(scope.user_scope_key)
                session_state = await self.repository.get_session_state(scope.session_scope_key)
                turn.pre_user_state = copy.deepcopy(user_state)
                turn.pre_session_state = copy.deepcopy(session_state)
                user_state, session_state = self.state_decay.apply(user_state, session_state)
                scene_resolution, behavior_result, policy, _lore_items, prompt_plan = await self._build_prompt_plan_bundle(
                    normalized,
                    user_state,
                    session_state,
                    apply_learning_patches=True,
                )
                runtime_text = self.prompt_composer.compose(prompt_plan)
                runtime_text = self.runtime_guard.wrap(runtime_text)

                # pre-turn counter update
                session_state.turn_counter += 1
                turn.pre_turn_written = True
                await self.repository.upsert_user_state(user_state)
                await self.repository.upsert_session_state(session_state)

            original = req.system_prompt or ""
            req.system_prompt = self.runtime_guard.inject(original, runtime_text)

            bundle = RuntimeContextBundle(
                turn_id=turn.turn_id,
                user_scope_key=scope.user_scope_key,
                session_scope_key=scope.session_scope_key,
                normalized_input=normalized,
                feature_scores=scene_resolution.feature_scores,
                scene_resolution=scene_resolution,
                behavior_result=behavior_result,
                policy=policy,
                prompt_plan=prompt_plan,
                selected_lore_ids=prompt_plan.selected_lore_ids,
                injected_len=len(runtime_text),
                assistant_reply_text="",
                unified_msg_origin=safe_get_unified_msg_origin(event),
            )
            self.turn_registry.attach_bundle(turn.turn_id, bundle)
            self.logger.info(
                "runtime_injected",
                turn_id=turn.turn_id,
                scope_key_user=scope.user_scope_key,
                scope_key_session=scope.session_scope_key,
                feature_scores=scene_resolution.feature_scores,
                scene_scores=scene_resolution.scene_scores,
                behavior_probabilities=behavior_result.behavior_probabilities,
                selected_examples=[example.to_log_dict() for example in prompt_plan.selected_examples],
                selected_policy=policy.to_log_dict(),
                selected_lore_ids=prompt_plan.selected_lore_ids,
                prompt_plan=prompt_plan.to_log_dict(),
                injected_len=len(runtime_text),
                pre_turn_written=turn.pre_turn_written,
                streaming_flag=normalized.is_streaming,
                user_state_version=user_state.state_version,
                session_state_version=session_state.state_version,
            )
            self._debug_turn(
                "runtime_injected",
                turn=turn,
                streaming_flag=normalized.is_streaming,
                selected_policy=policy.to_log_dict(),
                selected_lore_ids=prompt_plan.selected_lore_ids,
                user_state_version=user_state.state_version,
                session_state_version=session_state.state_version,
                prompt_plan_summary=prompt_plan.to_log_dict(),
            )
        except Exception as exc:  # noqa: BLE001
            decision = self.fallback.handle(exc, stage="on_llm_request", logger=self.logger)
            if decision.should_abort_turn:
                await self._rollback_turn(turn.turn_id, reason="fallback_on_llm_request")
            return

    @filter.on_agent_begin()
    async def on_agent_begin(self, event: AstrMessageEvent, run_context: Any):
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        self.tool_tracker.on_agent_begin(turn_id)
        self.logger.info("agent_begin", turn_id=turn_id)
        self._debug_turn("tool_entered", turn_id=turn_id)

    @filter.on_using_llm_tool()
    async def on_using_llm_tool(self, event: AstrMessageEvent, tool: Any, tool_args: dict | None):
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        self.tool_tracker.on_tool_begin(turn_id, getattr(tool, "name", "unknown"), tool_args or {})
        self.logger.info("tool_begin", turn_id=turn_id, tool_name=getattr(tool, "name", "unknown"), tool_args=tool_args or {})
        self._debug_turn("tool_begin", turn_id=turn_id)

    @filter.on_llm_tool_respond()
    async def on_llm_tool_respond(self, event: AstrMessageEvent, tool: Any, tool_args: dict | None, tool_result: Any):
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        self.tool_tracker.on_tool_result(turn_id, getattr(tool, "name", "unknown"), tool_result)
        trace = self.tool_tracker.get(turn_id)
        self.logger.info(
            "tool_result",
            turn_id=turn_id,
            tool_name=getattr(tool, "name", "unknown"),
            tool_trace=trace.to_log_dict() if trace else None,
        )
        self._debug_turn("tool_result", turn_id=turn_id)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        turn = self.turn_registry.get(turn_id)
        if turn and turn.bundle:
            extracted_text = extract_response_text(resp)
            if extracted_text:
                turn.bundle.assistant_reply_text = extracted_text
        self.turn_registry.mark_candidate(turn_id, stage="on_llm_response")
        self.logger.info("candidate_commit", turn_id=turn_id, commit_stage="on_llm_response")
        self._debug_turn("candidate_write", turn_id=turn_id, commit_stage="on_llm_response")

    @filter.on_agent_done()
    async def on_agent_done(self, event: AstrMessageEvent, run_context: Any, resp: LLMResponse):
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        turn = self.turn_registry.get(turn_id)
        if turn and turn.bundle:
            extracted_text = extract_response_text(resp)
            if extracted_text:
                turn.bundle.assistant_reply_text = extracted_text
        self.turn_registry.mark_candidate(turn_id, stage="on_agent_done")
        self.logger.info("candidate_commit", turn_id=turn_id, commit_stage="on_agent_done")
        self._debug_turn("candidate_write", turn_id=turn_id, commit_stage="on_agent_done")

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        await self._ensure_ready()
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        try:
            await self._final_commit(turn_id)
        except Exception as exc:  # noqa: BLE001
            decision = self.fallback.handle(exc, stage="after_message_sent", logger=self.logger)
            if decision.should_abort_turn:
                await self._rollback_turn(turn_id, reason="fallback_after_message_sent")

    async def _final_commit(self, turn_id: str):
        turn = self.turn_registry.get(turn_id)
        if not turn or turn.final_committed or turn.aborted:
            return

        async with self.scope_locks.guard(turn.user_scope_key, turn.session_scope_key):
            user_state = await self.repository.get_user_state(turn.user_scope_key)
            session_state = await self.repository.get_session_state(turn.session_scope_key)
            trace = self.tool_tracker.get(turn_id)

            user_updates: dict[str, Any] = {}
            session_updates: dict[str, Any] = {"last_sent_success": True}
            if trace and trace.tool_called:
                session_updates["last_tool_used"] = trace.tool_name
                session_updates["tool_chain_active"] = False
                if trace.tool_success:
                    user_updates["tool_style_bias"] = min(10, user_state.tool_style_bias + 1)

            # optimistic checks and transactional update
            await self.repository.final_commit_states(
                user_state=user_state,
                session_state=session_state,
                user_updates=user_updates,
                session_updates=session_updates,
            )

            try:
                await self._persist_turn_records(turn)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception(
                    "record_chain_persist_failed",
                    exc,
                    turn_id=turn_id,
                    stage="after_message_sent",
                )

        self.turn_registry.mark_final_committed(turn_id, stage="after_message_sent")
        trace = self.tool_tracker.pop(turn_id)
        self.logger.info(
            "final_commit",
            turn_id=turn_id,
            commit_stage="after_message_sent",
            final_committed=True,
            tool_trace=trace.to_log_dict() if trace else None,
            user_state_version=user_state.state_version,
            session_state_version=session_state.state_version,
        )
        self._debug_turn(
            "final_commit",
            turn_id=turn_id,
            tool_flags=trace.to_log_dict() if trace else None,
            commit_stage="after_message_sent",
            final_committed=True,
            user_state_version=user_state.state_version,
            session_state_version=session_state.state_version,
        )

    async def _persist_turn_records(self, turn: Any):
        bundle = getattr(turn, "bundle", None)
        if bundle is None:
            return

        raw_turn = RawTurnRecord(
            turn_id=turn.turn_id,
            timestamp=turn.created_at,
            user_message=bundle.normalized_input.raw_text,
            assistant_reply=bundle.assistant_reply_text,
            selected_policy=bundle.policy.to_log_dict(),
            final_committed=True,
        )
        turn_trace = TurnTraceRecord(
            turn_id=turn.turn_id,
            scene=bundle.prompt_plan.main_scene,
            selected_behavior=bundle.prompt_plan.selected_behavior,
            behavior_probabilities=bundle.prompt_plan.behavior_probabilities,
            feedback_label=bundle.scene_resolution.feedback_target if bundle.scene_resolution.is_feedback else None,
            scene_scores=bundle.prompt_plan.scene_scores,
            selected_example_ids=[example.example_id for example in bundle.prompt_plan.selected_examples],
            selected_lore_ids=bundle.prompt_plan.selected_lore_ids,
        )
        await self.repository.create_raw_turn(raw_turn)
        await self.repository.create_turn_trace(turn_trace)

        learning_item = self.learning_filter.build_item(bundle)
        if learning_item is not None:
            await self.repository.create_learning_buffer_item(learning_item)
            self.logger.info(
                "learning_buffer_appended",
                turn_id=turn.turn_id,
                content_mode=learning_item.content_mode,
                scene=learning_item.scene,
            )
        else:
            self.logger.info("learning_buffer_skipped", turn_id=turn.turn_id)

        await self._maybe_run_learning_analysis()

    async def _maybe_run_learning_analysis(self, *, force: bool = False) -> dict[str, Any]:
        if self.deterministic_mode and not force:
            self.logger.info("learning_analysis_skipped", reason="deterministic_mode")
            return {"pending_count": 0, "created_patches": []}
        pending = await self.repository.list_pending_learning_buffer(limit=50)
        if not pending:
            return {"pending_count": 0, "created_patches": []}
        if not force and len(pending) < self.learning_analyzer.min_batch_size:
            return {"pending_count": len(pending), "created_patches": []}

        traces_by_turn: dict[str, TurnTraceRecord] = {}
        example_ids: set[int] = set()
        for item in pending:
            trace = await self.repository.get_turn_trace(item.turn_id)
            if trace is None:
                continue
            traces_by_turn[item.turn_id] = trace
            example_ids.update(trace.selected_example_ids)

        examples_by_id = {
            int(example.example_id or 0): example
            for example in await self.repository.list_examples_by_ids(sorted(example_ids))
            if example.example_id is not None
        }
        patches = self.learning_analyzer.analyze(pending, traces_by_turn, examples_by_id, force=force)
        created_patches = []
        for patch in patches:
            created_patches.append(await self.repository.create_weight_patch(patch))
        await self.repository.mark_learning_buffer_analyzed(
            [int(item.buffer_id) for item in pending if item.buffer_id is not None]
        )
        self.logger.info(
            "learning_analysis_completed",
            pending_count=len(pending),
            patch_count=len(created_patches),
            patch_ids=[patch.patch_id for patch in created_patches],
        )
        return {
            "pending_count": len(pending),
            "created_patches": created_patches,
        }

    async def _rollback_turn(self, turn_id: str, reason: str):
        turn = self.turn_registry.get(turn_id)
        if not turn or turn.final_committed or turn.aborted:
            return
        restored = False
        restore_skipped_reason = ""
        if turn.pre_turn_written and turn.pre_user_state and turn.pre_session_state:
            async with self.scope_locks.guard(turn.user_scope_key, turn.session_scope_key):
                current_user = await self.repository.get_user_state(turn.user_scope_key)
                current_session = await self.repository.get_session_state(turn.session_scope_key)
                expected_turn_counter = int(getattr(turn.pre_session_state, "turn_counter", 0)) + 1
                if (
                    current_user.state_version == turn.pre_user_state.state_version
                    and current_session.state_version == turn.pre_session_state.state_version
                    and current_session.turn_counter == expected_turn_counter
                ):
                    await self.repository.upsert_user_state(copy.deepcopy(turn.pre_user_state))
                    await self.repository.upsert_session_state(copy.deepcopy(turn.pre_session_state))
                    restored = True
                else:
                    restore_skipped_reason = "state_moved_forward"

        self.turn_registry.mark_aborted(turn_id, reason=reason)
        trace = self.tool_tracker.pop(turn_id)
        self.logger.info(
            "rollback_turn",
            turn_id=turn_id,
            reason=reason,
            restored=restored,
            restore_skipped_reason=restore_skipped_reason,
            tool_trace=trace.to_log_dict() if trace else None,
            commit_stage=turn.commit_stage,
            final_committed=turn.final_committed,
            user_state_version=getattr(turn.pre_user_state, "state_version", None),
            session_state_version=getattr(turn.pre_session_state, "state_version", None),
        )
        self._debug_turn(
            "watchdog_rollback" if reason == "commit_timeout" else "rollback",
            turn_id=turn_id,
            tool_flags=trace.to_log_dict() if trace else None,
            commit_stage=turn.commit_stage,
            final_committed=turn.final_committed,
            user_state_version=getattr(turn.pre_user_state, "state_version", None),
            session_state_version=getattr(turn.pre_session_state, "state_version", None),
        )
