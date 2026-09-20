from fastapi import APIRouter, Query

from app.services.news_service import fetch_news_by_topic, fetch_trending_news


router = APIRouter(prefix="/news", tags=["news"])


@router.get("/trending")
async def trending_news() -> list[dict[str, str]]:
    return fetch_trending_news()


@router.get("/search")
async def search_news(topic: str = Query(..., min_length=1)) -> list[dict[str, str]]:
    return fetch_news_by_topic(topic)
