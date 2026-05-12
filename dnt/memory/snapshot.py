from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from dnt.config import DNTConfig
from dnt.core.buffer import WorkingMemoryBuffer
from dnt.core.models import AtomicObservation
from dnt.core.tree import NeuronTree

_SNAPSHOT_VERSION = "1.0"


class SnapshotManager:
    """Serializes and restores full DNT state. Supports dict and JSON file I/O."""

    @staticmethod
    def export(
        config: DNTConfig,
        tree: NeuronTree,
        buffer: WorkingMemoryBuffer,
        observe_count: int,
    ) -> dict:
        return {
            "version": _SNAPSHOT_VERSION,
            "config": config.model_dump(),
            "tree": tree.to_dict(),
            "buffer": [obs.model_dump(mode="json") for obs in buffer.peek()],
            "observe_count": observe_count,
        }

    @staticmethod
    def restore(
        snapshot: dict,
    ) -> Tuple[DNTConfig, NeuronTree, List[AtomicObservation], int]:
        config = DNTConfig(**snapshot.get("config", {}))
        tree = NeuronTree.from_dict(snapshot.get("tree", {}))
        buffer_items = [
            AtomicObservation(**obs_data) for obs_data in snapshot.get("buffer", [])
        ]
        observe_count = snapshot.get("observe_count", 0)
        return config, tree, buffer_items, observe_count

    @staticmethod
    def save(snapshot: dict, path: str) -> None:
        """Persist snapshot to a JSON file."""
        Path(path).write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def load(path: str) -> dict:
        """Load snapshot from a JSON file."""
        return json.loads(Path(path).read_text(encoding="utf-8"))
