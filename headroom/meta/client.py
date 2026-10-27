"""Client factory and media helpers for the Meta Model API (https://dev.meta.ai/docs)."""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from ..config import ROOT

BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"


def spark_model() -> str:
    return os.environ.get("HEADROOM_SPARK_MODEL", "muse-spark-1.3")


def make_client():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set: copy .env.example to .env and add the key")
    return OpenAI(base_url=BASE_URL, api_key=key)


def as_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def media_reference(path_or_url: str) -> str:
    """Public URLs pass through; local files become base64 data URLs (inline limit 50 MB)."""
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    return as_data_url(Path(path_or_url))


def segment(client, media_part: dict, concept: str, *, confidence: bool = True) -> str:
    """Stream a SAM 3.1 request (one noun phrase per request) and return the accumulated output lines."""
    stream = client.responses.create(
        model=SAM_MODEL,
        stream=True,
        metadata={"include_confidence": "true" if confidence else "false", "mask_encoding": "one_bit"},
        input=[{"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": concept}, media_part]}],
    )
    lines = [event.delta.rstrip("\n") for event in stream if event.type == "response.output_text.delta"]
    return "\n".join(lines)
