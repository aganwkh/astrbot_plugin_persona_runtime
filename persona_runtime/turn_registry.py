from __future__ import annotations

import time
from typing import Any

from .models import RuntimeContextBundle, TurnRecord
from .utils import safe_get_unified_msg_origin


class TurnRegistry:
    def __init__(self, default_timeout_seconds: int = 15):
        self._turns: dict[str, TurnRecord] = {}
        self._latest_by_origin: dict[str, str] = {}
        self._turns_by_origin: dict[str, list[str]] = {}
        self.default_timeout_seconds = default_timeout_seconds

    def create_turn(self, user_scope_key: str, session_scope_key: str) -> TurnRecord:
        turn = TurnRecord(user_scope_key=user_scope_key, session_scope_key=session_scope_key)
        turn.deadline_at = int(time.time()) + self.default_timeout_seconds
        self._turns[turn.turn_id] = turn
        return turn

    def attach_bundle(self, turn_id: str, bundle: RuntimeContextBundle):
        if turn_id in self._turns:
            self._turns[turn_id].bundle = bundle
            if bundle.unified_msg_origin:
                self._latest_by_origin[bundle.unified_msg_origin] = turn_id
                self._turns_by_origin.setdefault(bundle.unified_msg_origin, []).append(turn_id)

    def find_latest_turn_id(self, event: Any) -> str | None:
        origin = safe_get_unified_msg_origin(event)
        raw_text = (getattr(event, "message_str", "") or "").strip()
        for turn_id in reversed(self._turns_by_origin.get(origin, [])):
            turn = self._turns.get(turn_id)
            if not turn or turn.final_committed or turn.aborted:
                continue
            bundle = turn.bundle
            if raw_text and bundle and getattr(bundle.normalized_input, "raw_text", "") == raw_text:
                return turn_id
        for turn_id in reversed(self._turns_by_origin.get(origin, [])):
            turn = self._turns.get(turn_id)
            if turn and not turn.final_committed and not turn.aborted:
                return turn_id
        return self._latest_by_origin.get(origin)

    def get(self, turn_id: str) -> TurnRecord | None:
        return self._turns.get(turn_id)

    def mark_candidate(self, turn_id: str, stage: str):
        turn = self._turns.get(turn_id)
        if turn:
            turn.commit_stage = stage

    def mark_final_committed(self, turn_id: str, stage: str):
        turn = self._turns.get(turn_id)
        if turn:
            turn.final_committed = True
            turn.commit_stage = stage
            turn.finished_at = int(time.time())

    def mark_aborted(self, turn_id: str, reason: str):
        turn = self._turns.get(turn_id)
        if turn:
            turn.aborted = True
            turn.abort_reason = reason
            turn.finished_at = int(time.time())

    def iter_pending(self):
        return [t for t in self._turns.values() if not t.final_committed and not t.aborted]

    def snapshot(self):
        return {
            t.turn_id: {
                "user_scope_key": t.user_scope_key,
                "session_scope_key": t.session_scope_key,
                "commit_stage": t.commit_stage,
                "pre_turn_written": t.pre_turn_written,
                "final_committed": t.final_committed,
                "aborted": t.aborted,
                "abort_reason": t.abort_reason,
                "deadline_at": t.deadline_at,
                "finished_at": t.finished_at,
            }
            for t in self._turns.values()
        }

    def prune_finished(self, older_than_seconds: int = 300):
        now = int(time.time())
        expired = [
            turn_id
            for turn_id, turn in self._turns.items()
            if turn.finished_at and now - turn.finished_at > older_than_seconds
        ]
        for turn_id in expired:
            self._turns.pop(turn_id, None)
        live_turn_ids = set(self._turns)
        self._latest_by_origin = {
            origin: turn_id
            for origin, turn_id in self._latest_by_origin.items()
            if turn_id in live_turn_ids
        }
        self._turns_by_origin = {
            origin: [turn_id for turn_id in turn_ids if turn_id in live_turn_ids]
            for origin, turn_ids in self._turns_by_origin.items()
        }
