"""One client for Meta's open-weight models on any OpenAI-compatible partner, with record/replay.

Roles, each configured in .env (see .env.example):
    spark    Muse Spark, the operator agent (Meta Model API, or Spark Open on Baseten, Fireworks, ...)
    glimmer  Muse Glimmer, the backup operator on the site's own GPU (FailoverClient switches to it
             when the cloud can't be reached)

Every call is keyed by a hash of its full request. HEADROOM_MODEL_MODE:
    auto    replay a recording if one exists, otherwise call live and record (default)
    live    always call the model, and record
    replay  never call the network; a missing recording is an error (the hosted demo's mode)

Spend guard: every live call's token usage is priced and added to data/spend.json; once the total
reaches HEADROOM_BUDGET_USD (default $100) no further live call is made. Replays cost nothing.
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
    "meta": "https://api.meta.ai/v1",  # Meta Model API (muse-spark-1.3)
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


class BudgetExceeded(RuntimeError):
    pass


SPEND_LOG = DATA / "spend.json"


def _price(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def spent_usd() -> float:
    return json.loads(SPEND_LOG.read_text(encoding="utf-8"))["usd"] if SPEND_LOG.exists() else 0.0


def _charge(usage: dict | None) -> float:
    """Add a call's estimated cost to the running total. Prices are per million tokens (.env)."""
    usage = usage or {}
    cost = (usage.get("prompt_tokens", 0) * _price("HEADROOM_USD_PER_M_INPUT", 1.25)
            + usage.get("completion_tokens", 0) * _price("HEADROOM_USD_PER_M_OUTPUT", 4.25)) / 1e6
    total = spent_usd() + cost
    SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    SPEND_LOG.write_text(json.dumps({"usd": round(total, 4), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                         encoding="utf-8")
    return total


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
        budget = _price("HEADROOM_BUDGET_USD", 100.0)
        if spent_usd() >= budget:
            raise BudgetExceeded(f"spend guard: ${spent_usd():.2f} used of the ${budget:.0f} budget; "
                                 f"no more live calls (raise HEADROOM_BUDGET_USD in .env to continue)")
        started = time.perf_counter()
        response = self._openai().chat.completions.create(model=self.config.model, messages=messages, **kwargs)
        payload = response.model_dump(exclude_none=True)
        _charge(payload.get("usage"))
        self.recordings.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": request, "response": payload,
                                    "seconds": round(time.perf_counter() - started, 2),
                                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1),
                        encoding="utf-8")
        return payload


class LinkDown(ConnectionError):
    """The site's internet link is down (raised for the cloud model when the demo cuts the link)."""


def _unreachable() -> tuple[type[BaseException], ...]:
    from openai import APIConnectionError, InternalServerError  # APITimeoutError is an APIConnectionError

    return (LinkDown, APIConnectionError, InternalServerError)


class FailoverClient:
    """The cloud model first; if it can't be reached, the open-weight model on the site's own GPU.

    Only network failures fail over; anything else (a bad request, the spend guard) still raises.
    If neither model answers, LinkDown is raised: the planner and safety layer
    keep running either way, since no model sits in the control loop. `answered_by` names the
    role that served the last call.
    """

    def __init__(self, primary: str = "spark", backup: str = "glimmer", mode: str | None = None,
                 link_up: bool = True, recordings: Path = RECORDINGS):
        self.primary = ModelClient(primary, mode, recordings)
        self.backup = ModelClient(backup, mode, recordings)
        self.link_up = link_up
        self.answered_by = primary

    @property
    def model(self) -> str:
        return (self.primary if self.answered_by == self.primary.config.role else self.backup).model

    def chat(self, messages: list[dict], **kwargs) -> dict:
        try:
            if not self.link_up:
                raise LinkDown("internet link down")
            response = self.primary.chat(messages, **kwargs)
            self.answered_by = self.primary.config.role
            return response
        except _unreachable():
            try:
                response = self.backup.chat(messages, **kwargs)
            except _unreachable() as exc:
                raise LinkDown("no model reachable") from exc
            self.answered_by = self.backup.config.role
            return response


def as_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
