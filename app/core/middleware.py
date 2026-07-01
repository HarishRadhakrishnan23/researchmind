import time
from collections import defaultdict
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import structlog

logger = structlog.get_logger()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attaches a unique X-Request-ID to every request and response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status, and duration for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            request_id=getattr(request.state, "request_id", "-"),
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-process sliding-window rate limiter.
    For production, replace with Redis-backed solution.
    Default: 60 requests per 60 seconds per IP.
    """

    def __init__(self, app, calls: int = 60, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Health checks are exempt
        if request.url.path == "/health":
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Evict timestamps outside the window
        self._hits[ip] = [t for t in self._hits[ip] if now - t < self.period]

        if len(self._hits[ip]) >= self.calls:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit: {self.calls} requests/{self.period}s"},
                headers={"Retry-After": str(self.period)},
            )

        self._hits[ip].append(now)
        return await call_next(request)
