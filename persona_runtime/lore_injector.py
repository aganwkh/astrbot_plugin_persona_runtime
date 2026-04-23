from __future__ import annotations

import json
from pathlib import Path

from .models import DialoguePolicyResult, LoreItem, NormalizedInput


class LoreInjector:
    def __init__(self, lore_path: Path):
        self.lore_path = lore_path
        self._items = self._load_items()

    def _load_items(self) -> list[dict]:
        if not self.lore_path.exists():
            return []
        try:
            data = json.loads(self.lore_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        return [row for row in data if isinstance(row, dict)]

    def pick(
        self,
        normalized: NormalizedInput,
        policy: DialoguePolicyResult,
        top_k: int,
        score_threshold: float,
        max_chars: int,
    ) -> list[LoreItem]:
        if not policy.need_lore_injection:
            return []
        text = normalized.raw_text.casefold()
        candidates: list[LoreItem] = []
        for row in self._items:
            keywords = [
                str(k).casefold().strip()
                for k in row.get("keywords", [])
                if str(k).strip()
            ]
            if not keywords:
                continue
            hit_lengths = [len(kw) for kw in keywords if kw in text]
            hits = len(hit_lengths)
            coverage = hits / max(1, len(keywords))
            density = min(1.0, sum(hit_lengths) / max(1, len(text)))
            score = (coverage * 0.8) + (density * 0.2)
            if score < score_threshold:
                continue
            chunk = str(row.get("chunk", "")).strip()
            if not chunk:
                continue
            candidates.append(
                LoreItem(
                    lore_id=str(row.get("id", "unknown")),
                    text=chunk,
                    score=score,
                    topic=str(row.get("topic", "")),
                )
            )

        picked: list[LoreItem] = []
        used = 0
        for item in sorted(candidates, key=lambda x: x.score, reverse=True):
            remaining = max_chars - used
            if remaining <= 0:
                break
            text_chunk = self._truncate(item.text, remaining)
            if not text_chunk:
                break
            picked.append(
                LoreItem(
                    lore_id=item.lore_id,
                    text=text_chunk,
                    score=item.score,
                    topic=item.topic,
                )
            )
            used += len(text_chunk)
            if len(picked) >= top_k:
                break
        return picked

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        suffix = "...<trimmed>"
        if limit <= len(suffix):
            return text[:limit]
        return text[: limit - len(suffix)].rstrip() + suffix
