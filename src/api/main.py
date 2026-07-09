from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BookSearchResult,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    SampleUser,
)
from src.api.service import ApiState, build_state, recommendation_response, sample_users


def create_app(load_artifacts: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_artifacts:
            try:
                app.state.bigbook = build_state()
            except Exception as exc:  # pragma: no cover - exercised manually with missing artifacts.
                app.state.bigbook = ApiState(load_error=str(exc))
        else:
            app.state.bigbook = ApiState()
        yield

    app = FastAPI(
        title="BigBook Recommendation API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})

    def state() -> ApiState:
        return app.state.bigbook

    def require_ready() -> ApiState:
        current = state()
        if not current.ready:
            raise HTTPException(
                status_code=503,
                detail=current.load_error or "Recommendation artifacts are not loaded.",
            )
        return current

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current = state()
        recommender = current.recommender
        catalog = current.catalog
        return HealthResponse(
            status="ok" if current.ready else "degraded",
            ready=current.ready,
            books=len(catalog.books) if catalog is not None else 0,
            users=len(recommender.user_ids) if recommender is not None else 0,
            error=current.load_error,
        )

    @app.get("/books/search", response_model=list[BookSearchResult])
    def search_books(
        q: str = Query(default="", min_length=0, max_length=120),
        limit: int = Query(default=10, ge=1, le=25),
    ) -> list[BookSearchResult]:
        current = require_ready()
        assert current.catalog is not None
        return current.catalog.search(q, limit)

    @app.get("/users/sample", response_model=list[SampleUser])
    def users_sample(limit: int = Query(default=8, ge=1, le=25)) -> list[SampleUser]:
        require_ready()
        return sample_users(limit)

    @app.post("/recommendations", response_model=RecommendationResponse)
    def recommendations(request: RecommendationRequest) -> RecommendationResponse:
        current = require_ready()
        try:
            return recommendation_response(current, request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
