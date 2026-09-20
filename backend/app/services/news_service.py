from __future__ import annotations

import os
from datetime import date
from typing import List, TypedDict

import requests

from app.core.config import REQUEST_TIMEOUT_SECONDS
from app.core.logger import get_logger


logger = get_logger(__name__)


class NewsItem(TypedDict):
    title: str
    description: str
    source: str
    url: str


NEWS_API_URL = "https://newsapi.org/v2"
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
def _get_news_api_key() -> str:
    return os.getenv("NEWS_API_KEY", "").strip()


def _mock_news(topic: str) -> List[NewsItem]:
    base_topic = topic.strip() or "global affairs"
    return [
        {
            "title": f"{base_topic.title()}: Early reports drive rapid online discussion",
            "description": "Initial narratives are spreading quickly as communities react to developing updates.",
            "source": "MockWire",
            "url": "https://example.com/mock/1",
        },
        {
            "title": f"Analysts track sentiment shifts around {base_topic}",
            "description": "Experts note changing discourse patterns across social platforms and media channels.",
            "source": "MockWire",
            "url": "https://example.com/mock/2",
        },
        {
            "title": f"Public response to {base_topic} continues evolving",
            "description": "Observers report mixed interpretations as new claims and corrections emerge.",
            "source": "MockWire",
            "url": "https://example.com/mock/3",
        },
        {
            "title": f"Fact-check groups issue updates related to {base_topic}",
            "description": "Verification efforts focus on viral claims and context gaps in circulating posts.",
            "source": "MockWire",
            "url": "https://example.com/mock/4",
        },
        {
            "title": f"{base_topic.title()} remains a top-trending conversation",
            "description": "Coverage volume remains elevated as audiences follow unfolding developments.",
            "source": "MockWire",
            "url": "https://example.com/mock/5",
        },
    ]


def _normalize_newsapi_articles(articles: list[dict], limit: int = 5) -> List[NewsItem]:
    normalized: List[NewsItem] = []
    for article in articles[:limit]:
        normalized.append(
            {
                "title": str(article.get("title") or "Untitled"),
                "description": str(article.get("description") or ""),
                "source": str((article.get("source") or {}).get("name") or "Unknown"),
                "url": str(article.get("url") or ""),
            }
        )
    return normalized


def _normalize_gdelt_articles(items: list[dict], limit: int = 5) -> List[NewsItem]:
    normalized: List[NewsItem] = []
    for item in items[:limit]:
        normalized.append(
            {
                "title": str(item.get("title") or "Untitled"),
                "description": str(item.get("seendate") or item.get("domain") or ""),
                "source": str(item.get("sourcecountry") or item.get("domain") or "GDELT"),
                "url": str(item.get("url") or ""),
            }
        )
    return normalized


def _fetch_from_newsapi(topic: str, *, from_date: str | None = None, sort_by: str = "publishedAt", page_size: int = 5) -> List[NewsItem]:
    news_api_key = _get_news_api_key()
    if not news_api_key:
        raise RuntimeError("NEWS_API_KEY is not configured")

    endpoint = f"{NEWS_API_URL}/everything"
    params = {
        "q": topic,
        "language": "en",
        "sortBy": sort_by,
        "pageSize": page_size,
        "apiKey": news_api_key,
    }
    if from_date:
        params["from"] = from_date
    response = requests.get(endpoint, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("articles", [])
    return _normalize_newsapi_articles(articles)


def _fetch_from_gdelt(topic: str) -> List[NewsItem]:
    params = {
        "query": topic,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": "5",
        "sort": "DateDesc",
    }
    response = requests.get(GDELT_API_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("articles", [])
    return _normalize_gdelt_articles(articles)


def fetch_news_by_topic(topic: str) -> List[NewsItem]:
    clean_topic = topic.strip()
    if not clean_topic:
        return _mock_news("trending")

    if _get_news_api_key():
        try:
            news = _fetch_from_newsapi(clean_topic)
            if news:
                return news
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for topic=%s: %s", clean_topic, exc)

    try:
        gdelt_news = _fetch_from_gdelt(clean_topic)
        if gdelt_news:
            return gdelt_news
    except Exception as exc:
        logger.warning("GDELT fetch failed for topic=%s: %s", clean_topic, exc)

    return _mock_news(clean_topic)


def fetch_trending_news() -> List[NewsItem]:
    today = date.today().isoformat()
    try:
        news = _fetch_from_newsapi("India", from_date=today, sort_by="popularity", page_size=10)
        if news:
            return news
    except Exception as exc:
        logger.warning("NewsAPI trending fetch failed for India: %s", exc)

    return fetch_news_by_topic("India")
