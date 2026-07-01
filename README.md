# ResearchMind

An end-to-end production RAG pipeline with a ReAct agentic workflow, built with FastAPI.

**Live API:** https://harishrad-researchmind.hf.space/docs  
**GitHub:** https://github.com/HarishRadhakrishnan23/researchmind

---

## What this project demonstrates

| Concept | Implementation |
|---------|---------------|
| RAG pipeline | PDF ingestion → parent-child chunking → OpenAI embeddings → Qdrant vector store |
| Hybrid retrieval | Dense search + BM25 → Reciprocal Rank Fusion → Cohere reranker |
| Agentic workflow | ReAct loop (Reason → Act → Observe) with OpenAI function calling |
| Guardrails | Max step limits, loop detection, dangerous pattern filtering |
| FastAPI production | Async routes, dependency injection, rate limiting, structured logging, streaming SSE |

---

## Tech stack

- **Backend:** FastAPI, Python 3.12, uvicorn
- **LLM:** OpenAI gpt-4o-mini + text-embedding-3-small
- **Vector DB:** Qdrant Cloud
- **Reranker:** Cohere rerank-english-v3.0
- **Chunking:** Parent-child strategy with LangChain splitters
- **Hybrid search:** qdrant-client + rank-bm25
- **PDF parsing:** PyMuPDF (fitz)
- **Deployment:** Hugging Face Spaces (Docker)

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Upload PDF → chunk → embed → store in Qdrant |
| POST | `/query` | RAG query with grounded answer + source citations |
| POST | `/query/stream` | Streaming SSE response |
| POST | `/agent/run` | Multi-step ReAct agent with full reasoning trace |
| GET  | `/health` | Health check |

---

## RAG pipeline — key design decisions

### Parent-child chunking

Two-level chunk hierarchy instead of flat fixed-size chunks:

- **Child chunks** (128 tokens) — indexed for retrieval precision
- **Parent chunks** (800 tokens) — returned as context for richness

When a child chunk matches, its parent is returned. Prevents the failure where the answer exists but the retrieved chunk is too narrow.

### Hybrid retrieval

Dense search finds semantic matches. BM25 catches exact keyword matches. Reciprocal Rank Fusion merges both ranked lists without needing score normalization. Cohere reranker does final scoring.

---

## ReAct agent loop

The agent reasons and acts in a loop: Thought → Tool Call → Observation → repeat until finish() is called or max steps hit. Full reasoning trace returned in steps[] — every thought, tool call, and observation is visible.

**Available tools:** rag_search, summarize_topic, compare_and_analyze, finish

**Guardrails:** max step limit, loop detection, dangerous pattern filtering

---

## Quickstart

```bash
git clone https://github.com/HarishRadhakrishnan23/researchmind.git
cd researchmind/backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install uv && uv pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

---

## Project structure

```
researchmind/
├── backend/
│   └── app/
│       ├── main.py
│       ├── core/
│       │   ├── config.py
│       │   ├── security.py
│       │   └── middleware.py
│       ├── api/routes/
│       │   ├── ingest.py
│       │   ├── query.py
│       │   └── agent.py
│       └── services/
│           ├── rag/
│           │   ├── chunker.py
│           │   ├── embedder.py
│           │   ├── vector_store.py
│           │   ├── retriever.py
│           │   └── pipeline.py
│           └── agent/
│               ├── tools.py
│               ├── react_loop.py
│               └── guardrails.py
└── scripts/
    └── eval_rag.py
```

---

Built by [Harish Radhakrishnan](https://www.linkedin.com/in/harish-radhakrishnan-207523255/) · [GitHub](https://github.com/HarishRadhakrishnan23)
