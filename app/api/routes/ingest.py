from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.core.config import get_settings, Settings
from app.core.security import require_api_key
from app.models.requests import IngestResponse
from app.services.rag.pipeline import ingest_document

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    settings: Settings = Depends(get_settings),
    _: str = Depends(require_api_key),
):
    """
    Upload a PDF → chunk → embed → store in Qdrant.
    Returns document_id and number of chunks created.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Validate file size
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    try:
        result = await ingest_document(
            file_bytes=file_bytes,
            filename=file.filename,
        )
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")
