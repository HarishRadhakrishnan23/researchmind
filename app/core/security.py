from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(_api_key_header)) -> str:
    """
    Dependency: validates X-API-Key header against API_SECRET_KEY.
    Use as: route(... _: str = Depends(require_api_key))
    """
    settings = get_settings()

    # Skip auth in development so you can curl freely
    if not settings.is_production:
        return "dev-bypass"

    if not key or key != settings.api_secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return key
