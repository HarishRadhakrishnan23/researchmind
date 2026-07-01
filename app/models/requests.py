from typing import Any
from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str | None = None   # override default collection


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    page: int | None
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    query: str


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000,
                      description="The research task for the agent to complete")
    max_steps: int = Field(default=10, ge=1, le=20)


class AgentStep(BaseModel):
    step: int
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: str


class AgentResponse(BaseModel):
    task: str
    answer: str
    steps: list[AgentStep]
    total_steps: int
    sources: list[SourceChunk] = []
