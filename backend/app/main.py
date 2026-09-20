from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import APP_NAME, APP_VERSION, CORS_ORIGINS
from app.core.logger import configure_logging, get_logger
from app.routes.analyze import router as analyze_router
from app.routes.explain import router as explain_router
from app.routes.generate import router as generate_router
from app.routes.news import router as news_router
from app.routes.predict import router as predict_router
from app.routes.simulate import router as simulate_router
from app.services.model_service import get_model_service


configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model_service()
    logger.info("Model service loaded at startup")
    yield


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check() -> dict:
    return {"status": "ok"}


app.include_router(generate_router)
app.include_router(news_router)
app.include_router(simulate_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(analyze_router)