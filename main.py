from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from .persona_runtime.bypass_router import BypassRouter
from .persona_runtime.commit_watchdog import CommitWatchdog
from .persona_runtime.dialogue_policy import DialoguePolicy
from .persona_runtime.eligibility_checker import EligibilityChecker
from .persona_runtime.fallback_controller import FallbackController
from .persona_runtime.input_normalizer import InputNormalizer
from .persona_runtime.lore_injector import LoreInjector
from .persona_runtime.models import RuntimeContextBundle, ToolTrace
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
from .persona_runtime.utils import safe_get_unified_msg_origin


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
        self.dialogue_policy = DialoguePolicy()
        self.lore_injector = LoreInjector(self.data_dir / "lore_keywords.json")
        self.prompt_merge_policy = PromptMergePolicy()
        self.token_guard = TokenBudgetGuard(max_runtime_chars=int(config.get("max_runtime_chars", 800)))
        self.prompt_composer = PromptComposer(static_persona_prompt=config.get("static_persona_prompt", ""))
        self.runtime_guard = RuntimeInjectionGuard()
        self.fallback = FallbackController()
        self.logger = ObservabilityLogger(enabled=bool(config.get("debug_log", True)))
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

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        await self._ensure_ready()
        if not self.config.get("enable_runtime", True):
            return

        normalized = self.input_normalizer.normalize(event)
        if not self.eligibility_checker.is_eligible(normalized):
            self.logger.info("skip_ineligible", raw_text=normalized.raw_text)
            return

        bypass = self.bypass_router.decide(normalized)
        if bypass.bypass:
            self.logger.info("bypass_runtime", reason=bypass.reason, raw_text=normalized.raw_text)
            return

        scope = self.scope_resolver.resolve(event)
        turn = self.turn_registry.create_turn(scope.user_scope_key, scope.session_scope_key)

        try:
            async with self.scope_locks.guard(scope.user_scope_key, scope.session_scope_key):
                user_state = await self.repository.get_user_state(scope.user_scope_key)
                session_state = await self.repository.get_session_state(scope.session_scope_key)
                turn.pre_user_state = copy.deepcopy(user_state)
                turn.pre_session_state = copy.deepcopy(session_state)
                user_state, session_state = self.state_decay.apply(user_state, session_state)
                resolution = self.state_resolver.resolve(normalized, user_state, session_state)
                policy = self.dialogue_policy.build(normalized, resolution)
                lore_items = self.lore_injector.pick(normalized, policy, int(self.config.get("lore_top_k", 2)), float(self.config.get("lore_score_threshold", 0.55)), int(self.config.get("max_lore_chars", 300)))
                merged = self.prompt_merge_policy.merge(policy=policy, lore_items=lore_items)
                budgeted = self.token_guard.apply(merged)
                runtime_text = self.prompt_composer.compose(budgeted)
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
                feature_scores=resolution.feature_scores,
                policy=policy,
                selected_lore_ids=[item.lore_id for item in lore_items],
                injected_len=len(runtime_text),
                unified_msg_origin=safe_get_unified_msg_origin(event),
            )
            self.turn_registry.attach_bundle(turn.turn_id, bundle)
            self.logger.info(
                "runtime_injected",
                turn_id=turn.turn_id,
                scope_key_user=scope.user_scope_key,
                scope_key_session=scope.session_scope_key,
                feature_scores=resolution.feature_scores,
                selected_policy=policy.to_log_dict(),
                selected_lore_ids=[item.lore_id for item in lore_items],
                injected_len=len(runtime_text),
                pre_turn_written=turn.pre_turn_written,
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

    @filter.on_using_llm_tool()
    async def on_using_llm_tool(self, event: AstrMessageEvent, tool: Any, tool_args: dict | None):
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        self.tool_tracker.on_tool_begin(turn_id, getattr(tool, "name", "unknown"), tool_args or {})
        self.logger.info("tool_begin", turn_id=turn_id, tool_name=getattr(tool, "name", "unknown"), tool_args=tool_args or {})

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

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        self.turn_registry.mark_candidate(turn_id, stage="on_llm_response")
        self.logger.info("candidate_commit", turn_id=turn_id, commit_stage="on_llm_response")

    @filter.on_agent_done()
    async def on_agent_done(self, event: AstrMessageEvent, run_context: Any, resp: LLMResponse):
        turn_id = self.turn_registry.find_latest_turn_id(event)
        if not turn_id:
            return
        self.turn_registry.mark_candidate(turn_id, stage="on_agent_done")
        self.logger.info("candidate_commit", turn_id=turn_id, commit_stage="on_agent_done")

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

        self.turn_registry.mark_final_committed(turn_id, stage="after_message_sent")
        trace = self.tool_tracker.pop(turn_id)
        self.logger.info(
            "final_commit",
            turn_id=turn_id,
            commit_stage="after_message_sent",
            final_committed=True,
            tool_trace=trace.to_log_dict() if trace else None,
        )

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
        )
