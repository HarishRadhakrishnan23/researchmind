from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.core.security import require_api_key
from app.models.requests import QueryRequest, QueryResponse
from app.services.rag.pipeline import query_rag, stream_query_rag

router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(
    request: QueryRequest,
    _: str = Depends(require_api_key),
):
    """
    RAG query: retrieve relevant chunks → generate grounded answer.
    Returns full answer with source citations.
    """
    try:
        result = await query_rag(
            query=request.query,
            collection=request.collection,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream", tags=["RAG"])
async def query_stream(
    request: QueryRequest,
    _: str = Depends(require_api_key),
):
    """
    Streaming SSE version of /query.
    Returns Server-Sent Events: data: <token>\\n\\n
    Frontend: const es = new EventSource('/query/stream')
    """
    return StreamingResponse(
        stream_query_rag(
            query=request.query,
            collection=request.collection,
            top_k=request.top_k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )
