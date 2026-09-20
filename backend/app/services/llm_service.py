import json
from typing import Any, Dict

import requests

from app.core.config import ALLOW_LLM_FALLBACK, OLLAMA_MODEL, OLLAMA_URL, REQUEST_TIMEOUT_SECONDS
from app.core.logger import get_logger


logger = get_logger(__name__)


class OllamaServiceError(RuntimeError):
    pass


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _generate(prompt: str) -> str:
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        raise OllamaServiceError("Ollama service unavailable") from exc

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        logger.error("Ollama response was not valid JSON")
        raise OllamaServiceError("Invalid Ollama response") from exc

    generated = data.get("response", "")
    if not isinstance(generated, str) or not generated.strip():
        raise OllamaServiceError("Empty Ollama response")

    return _clean_text(generated)


def _fallback_text(topic: str, narrative_type: str) -> str:
    clean_topic = _clean_text(topic) or "the reported event"
    if narrative_type == "misinformation":
        return (
            f"Breaking reports claim {clean_topic} is escalating faster than officials admit. "
            f"Witness accounts describe urgent failures, and sources say immediate public action is the only way to avoid severe consequences. "
            f"Although full verification is pending, the narrative is spreading as a high-risk alert."
        )
    if narrative_type == "counter":
        return (
            f"Current evidence does not support viral claims about {clean_topic}. "
            f"Available reporting indicates key details were exaggerated or taken out of context. "
            f"Readers should rely on primary sources, verified statements, and independent fact-checking before sharing."
        )
    return (
        f"Initial coverage on {clean_topic} remains mixed, with ongoing updates from public sources and local reporting. "
        f"Authorities and analysts are still verifying core facts, and conclusions may change as more evidence becomes available."
    )


def _generate_with_fallback(prompt: str, topic: str, narrative_type: str) -> str:
    try:
        return _generate(prompt)
    except OllamaServiceError as exc:
        if not ALLOW_LLM_FALLBACK:
            raise
        logger.warning("Ollama failed, using local fallback generator for type=%s: %s", narrative_type, exc)
        return _fallback_text(topic=topic, narrative_type=narrative_type)


def generate_misinformation(topic: str) -> str:
    prompt = (
        f"Generate a highly persuasive but misleading news article about {topic}. "
        f"Use an urgent emotional tone, clear persuasion intent, and realistic details so it reads like a plausible report."
    )
    logger.info("Generating misinformation for topic=%s", topic)
    return _generate_with_fallback(prompt=prompt, topic=topic, narrative_type="misinformation")


def generate_counter_narrative(topic: str) -> str:
    prompt = (
        f"Generate a factual, evidence-based rebuttal correcting misinformation about {topic}. "
        f"Use a calm, authoritative tone, corrective intent, and realistic journalistic style."
    )
    logger.info("Generating counter narrative for topic=%s", topic)
    return _generate_with_fallback(prompt=prompt, topic=topic, narrative_type="counter")


def generate_neutral_news(topic: str) -> str:
    prompt = (
        f"Generate a balanced and factual news report about {topic}. "
        f"Use a neutral tone, informative intent, and realistic reporting style without persuasion."
    )
    logger.info("Generating neutral report for topic=%s", topic)
    return _generate_with_fallback(prompt=prompt, topic=topic, narrative_type="neutral")