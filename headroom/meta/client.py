"""One client for Meta's open-weight models on any OpenAI-compatible partner, with record/replay.

Roles, each configured in .env (see .env.example):
    spark    Muse Spark Open, the operator agent (Baseten, Fireworks, Modal, Groq, ...)
    glimmer  Muse Glimmer, the offline backup operator (may sit on a different provider)

Every call is keyed by a hash of its full request. HEADROOM_MODEL_MODE:
    auto    replay a recording if one exists, otherwise call live and record (default)
    live    always call the model, and record
    replay  never call the network; a missing recording is an error (the hosted demo's mode)
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import DATA, ROOT

RECORDINGS = DATA / "recordings"

PROVIDERS = {  # base URLs; model ids come from .env because each partner names them differently
    "baseten": "https://inference.baseten.co/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
DEFAULT_MODELS = {"spark": "meta-models/muse-spark-open-1026"}  # Baseten's hackathon id


def _env(name: str, default: str = "") -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class RoleConfig:
    role: str
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls, role: str) -> "RoleConfig":
        prefix = f"HEADROOM_{role.upper()}"
        provider = _env(f"{prefix}_PROVIDER", "baseten").lower()
        base_url = _env(f"{prefix}_BASE_URL") or PROVIDERS.get(provider, "")
        return cls(role=role, base_url=base_url, api_key=_env(f"{prefix}_API_KEY"),
                   model=_env(f"{prefix}_MODEL", DEFAULT_MODELS.get(role, "")))


class RecordingMissing(RuntimeError):
    pass


class ModelClient:
    """`chat()` mirrors chat.completions.create and returns the response as a plain dict."""

    def __init__(self, role: str = "spark", mode: str | None = None, recordings: Path = RECORDINGS):
        self.config = RoleConfig.from_env(role)
        self.mode = (mode or _env("HEADROOM_MODEL_MODE", "auto")).lower()
        self.recordings = recordings
        self._client = None

    @property
    def model(self) -> str:
        return self.config.model

    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            if not (self.config.api_key and self.config.base_url and self.config.model):
                raise RuntimeError(f"{self.config.role}: set HEADROOM_{self.config.role.upper()}_API_KEY, "
                                   f"_PROVIDER (or _BASE_URL) and _MODEL in .env")
            self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return self._client

    def _key(self, request: dict) -> str:
        canonical = json.dumps(request, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def chat(self, messages: list[dict], **kwargs) -> dict:
        request = {"role": self.config.role, "model": self.config.model, "messages": messages, **kwargs}
        path = self.recordings / f"{self._key(request)}.json"
        if self.mode in ("auto", "replay") and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.mode == "replay":
            raise RecordingMissing(f"no recording for this request ({path.name}); run once in live mode")
        started = time.perf_counter()
        response = self._openai().chat.completions.create(model=self.config.model, messages=messages, **kwargs)
        payload = response.model_dump(exclude_none=True)
        self.recordings.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": request, "response": payload,
                                    "seconds": round(time.perf_counter() - started, 2),
                                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1),
                        encoding="utf-8")
        return payload


def as_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
