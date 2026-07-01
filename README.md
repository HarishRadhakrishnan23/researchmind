---
title: ResearchMind
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ResearchMind

Production RAG pipeline with ReAct agentic workflow built with FastAPI.

**Endpoints:**
- `POST /ingest` — Upload PDF and index it
- `POST /query` — RAG query with citations
- `POST /agent/run` — Multi-step ReAct agent
- `GET /docs` — Swagger UI
