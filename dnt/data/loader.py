"""
Seed data loader — loads pre-built observation sets into a DNT instance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

_SEEDS_DIR = Path(__file__).parent / "seeds"


def list_seeds() -> List[Dict]:
    """Return metadata for all available seed files."""
    seeds = []
    for f in sorted(_SEEDS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            seeds.append({
                "name":        data.get("name", f.stem),
                "label":       data.get("label", f.stem),
                "description": data.get("description", ""),
                "count":       len(data.get("observations", [])),
            })
        except Exception:
            pass
    return seeds


def get_seed(name: str) -> Dict:
    """Return a seed by name. Raises FileNotFoundError if not found."""
    path = _SEEDS_DIR / f"{name}.json"
    if not path.exists():
        available = [f.stem for f in _SEEDS_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Seed '{name}' not found. Available: {available}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


async def load_seed(dnt, name: str) -> Dict:
    """
    Feed a seed dataset into a DNT instance.
    Consolidates in batches so the tree grows progressively.
    Returns a summary dict.
    """
    seed = get_seed(name)
    observations: List[str] = seed.get("observations", [])

    for obs in observations:
        await dnt.observe(obs)

    await dnt.consolidate()

    stats = dnt.stats()
    return {
        "seed":     seed["name"],
        "label":    seed.get("label", name),
        "loaded":   len(observations),
        "nodes":    stats.node_count,
        "edges":    stats.edge_count,
    }
