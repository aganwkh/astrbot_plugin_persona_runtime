from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from .observability_logger import ObservabilityLogger
from .turn_registry import TurnRegistry


class CommitWatchdog:
    def __init__(
        self,
        turn_registry: TurnRegistry,
        timeout_seconds: int,
        on_timeout: Callable[[str, str], Awaitable[None]],
        logger: ObservabilityLogger,
    ):
        self.turn_registry = turn_registry
        self.timeout_seconds = timeout_seconds
        self.on_timeout = on_timeout
        self.logger = logger
        self._task: asyncio.Task | None = None
        self._stop = False

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop = False
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stop = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        while not self._stop:
            now = int(time.time())
            for turn in self.turn_registry.iter_pending():
                if turn.deadline_at and now >= turn.deadline_at:
                    try:
                        await self.on_timeout(turn.turn_id, "commit_timeout")
                        self.logger.info(
                            "commit_watchdog_timeout",
                            turn_id=turn.turn_id,
                            deadline_at=turn.deadline_at,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.logger.error(
                            "commit_watchdog_error",
                            turn_id=turn.turn_id,
                            error=repr(exc),
                        )
            self.turn_registry.prune_finished()
            await asyncio.sleep(1)
