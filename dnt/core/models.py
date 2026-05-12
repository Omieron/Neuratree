from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class NeuronNode(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    label: str
    summary: Optional[str] = None
    quantization_error: float = 0.0
    vector: Optional[List[float]] = None
    level: int = 0
    source_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NeuronEdge(BaseModel):
    source: UUID
    target: UUID
    relation: str
    weight: float = 0.1
    last_activated: float = Field(default_factory=time.time)


class AtomicObservation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    raw_text: str
    source: str = "user"
    timestamp: float = Field(default_factory=time.time)
    logic_type: Literal["fact", "rule", "preference"] = "fact"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Triplet(BaseModel):
    subject: str
    predicate: str
    object: str
    logic_type: Literal["fact", "rule", "preference"] = "fact"
    confidence: float = 1.0


class DNTStats(BaseModel):
    node_count: int
    edge_count: int
    buffer_size: int
    observe_count: int
