from __future__ import annotations


class RuntimeInjectionGuard:
    PREFIX = "<<PR_RUNTIME_BEGIN>>"
    SUFFIX = "<<PR_RUNTIME_END>>"

    def wrap(self, text: str) -> str:
        return f"{self.PREFIX}\n{text}\n{self.SUFFIX}"

    def inject(self, original: str, runtime_text: str) -> str:
        original = (original or "").strip()
        if original:
            return f"{original}\n\n{runtime_text}"
        return runtime_text
