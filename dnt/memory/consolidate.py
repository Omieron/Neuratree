from __future__ import annotations

import asyncio
from typing import List, Optional

from dnt.config import DNTConfig
from dnt.core.models import AtomicObservation
from dnt.core.tree import NeuronTree
from dnt.learning.ghsom import GHSOMGrower
from dnt.learning.hebbian import HebbianLearner
from dnt.learning.triplet import TripletExtractor


class ConsolidationEngine:
    """
    Async event-driven consolidation engine.

    Two modes:
    - Inline (default): run_once() processes observations immediately.
    - Background: start() launches a worker task; enqueue() feeds it
      without blocking observe(). Call stop() to drain and shut down.
    """

    def __init__(
        self,
        config: DNTConfig,
        triplet_extractor: TripletExtractor,
        hebbian: HebbianLearner,
        ghsom: GHSOMGrower,
        tree: NeuronTree,
    ) -> None:
        self._config = config
        self._triplet_extractor = triplet_extractor
        self._hebbian = hebbian
        self._ghsom = ghsom
        self._tree = tree
        self._queue: asyncio.Queue[List[AtomicObservation]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Background mode
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch background worker if not already running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Drain the queue, then cancel the worker."""
        await self.drain()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def drain(self) -> None:
        """Block until all queued consolidation events are processed."""
        await self._queue.join()

    async def enqueue(self, observations: List[AtomicObservation]) -> None:
        """Put observations on the queue (non-blocking)."""
        await self._queue.put(observations)

    @property
    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    # ------------------------------------------------------------------
    # Inline mode
    # ------------------------------------------------------------------

    async def run_once(self, observations: List[AtomicObservation]) -> None:
        """Process observations immediately in the calling coroutine."""
        if not observations:
            return
        await self._process(observations)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _worker(self) -> None:
        while True:
            try:
                observations = await self._queue.get()
            except asyncio.CancelledError:
                break
            try:
                await self._process(observations)
            finally:
                self._queue.task_done()

    async def _process(self, observations: List[AtomicObservation]) -> None:
        all_triplets = []
        for obs in observations:
            triplets = await self._triplet_extractor.extract(obs)
            all_triplets.extend(triplets)

        if all_triplets:
            self._hebbian.update(self._tree, all_triplets)
            self._ghsom.grow_if_needed(self._tree)
