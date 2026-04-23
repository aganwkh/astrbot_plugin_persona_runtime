from __future__ import annotations


class OutputRefiner:
    def refine(self, text: str, streaming: bool) -> str:
        if streaming:
            return text
        return text.replace("好的好的", "好的").replace("我来为你", "我来")
