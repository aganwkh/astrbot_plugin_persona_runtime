from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class ScopeLockManager:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    @asynccontextmanager
    async def guard(self, user_scope_key: str, session_scope_key: str):
        first = self._get_lock(f"user:{user_scope_key}")
        second = self._get_lock(f"session:{session_scope_key}")
        await first.acquire()
        await second.acquire()
        try:
            yield
        finally:
            second.release()
            first.release()
