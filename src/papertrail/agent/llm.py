"""A small chat client with failover over a CURATED list of free models: Groq first (fast, about 1-2 s), then
OpenRouter's free tier as the backup. Both speak the OpenAI chat-completions protocol, so plain httpx is enough (no SDK).

Generation never looks models up at runtime (slow, and free catalogues change monthly): it walks CURATED, or starts with
the model the user picked. `discover_free_models()` fetches OpenRouter's current free models only when the user asks
("Find more free models" in the app), so they can be picked too.

Keys come from the environment: LLM_API_KEY (or GROQ_API_KEY) and OPENROUTER_API_KEY. The provider of a key is told by
its prefix (gsk_ Groq, sk-or- OpenRouter, xai- xAI), so a key pasted under the wrong name still goes to the right place.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

ENDPOINTS = {"groq": "https://api.groq.com/openai/v1", "openrouter": "https://openrouter.ai/api/v1", "xai": "https://api.x.ai/v1"}
PREFIXES = {"gsk_": "groq", "sk-or-": "openrouter", "xai-": "xai"}
RETRYABLE = {402, 404, 408, 409, 413, 422, 429, 500, 502, 503, 504}   # move on to the next model
MAX_ATTEMPTS = 6                                                      # model attempts per call
OPENROUTER_SKIP = ("safety", "code", "guard", "-vl", "omni", "2.6b", "nano")  # free models built for a narrow job


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    model: str
    label: str
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


# Tested on a real case on Oct 20, 2026: each returned a schema-valid summary that passed the checker on the first draft.
CURATED: tuple[ModelChoice, ...] = (
    ModelChoice("groq", "openai/gpt-oss-120b", "GPT-OSS 120B", "Groq · about 2 s · best quality"),
    ModelChoice("groq", "qwen/qwen3.8-27b", "Qwen3.8 27B", "Groq · about 2 s"),
    ModelChoice("groq", "openai/gpt-oss-20b", "GPT-OSS 20B", "Groq · about 1 s · fastest"),
    ModelChoice("openrouter", "minimax/minimax-m3.1-flash:free", "MiniMax M3.1 Flash", "OpenRouter free · about 10 s · backup"),
)


class LLMError(RuntimeError):
    pass


def load_env(path: Path | str = ".env") -> None:
    """Read KEY=VALUE lines into os.environ without overriding what is already set (no python-dotenv needed)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def provider_of(key: str) -> str | None:
    return next((p for pre, p in PREFIXES.items() if key.startswith(pre)), None)


@dataclass
class Provider:
    name: str
    key: str


def providers_from_env() -> list[Provider]:
    """Groq first, then OpenRouter, then anything else; each key goes to the provider its prefix names."""
    found: dict[str, str] = {}
    for var in ("LLM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "XAI_API_KEY"):
        key = os.environ.get(var, "").strip()
        if key:
            found.setdefault(provider_of(key) or {"GROQ_API_KEY": "groq", "OPENROUTER_API_KEY": "openrouter", "XAI_API_KEY": "xai"}.get(var, "groq"), key)
    order = ("groq", "openrouter", "xai")
    return [Provider(n, found[n]) for n in order if n in found]


def available(catalogue: tuple[ModelChoice, ...] | list[ModelChoice] = CURATED, providers: list[Provider] | None = None) -> list[ModelChoice]:
    """The curated models whose provider has a key here."""
    have = {p.name for p in (providers if providers is not None else providers_from_env())}
    return [c for c in catalogue if c.provider in have]


def discover_free_models(http: httpx.Client | None = None, exclude: tuple[ModelChoice, ...] = CURATED, limit: int = 12) -> list[ModelChoice]:
    """OpenRouter's current free text models that can answer in JSON, newest catalogue (public endpoint, no key needed).
    Called only when the user asks for more models; not tested by us, so they are labelled as such."""
    http = http or httpx.Client(timeout=20)
    r = http.get(f"{ENDPOINTS['openrouter']}/models")
    r.raise_for_status()
    known = {c.model for c in exclude}
    out = []
    for m in r.json().get("data", []):
        mid = m.get("id", "")
        params = set(m.get("supported_parameters") or [])
        if (mid.endswith(":free") and mid not in known and not any(x in mid for x in OPENROUTER_SKIP)
                and "text" in ((m.get("architecture") or {}).get("output_modalities") or ["text"])
                and {"response_format", "structured_outputs"} & params):
            out.append((m.get("context_length") or 0, ModelChoice("openrouter", mid, (m.get("name") or mid).replace(" (free)", ""),
                                                                  "OpenRouter free · found live · untested")))
    return [c for _, c in sorted(out, key=lambda t: -t[0])][:limit]


class ChatClient:
    """chat_json(messages) -> (parsed JSON, meta). Starts with the picked model (if any), then walks the curated list."""

    def __init__(self, providers: list[Provider] | None = None, http: httpx.Client | None = None, timeout: float = 60.0,
                 model: ModelChoice | None = None, catalogue: tuple[ModelChoice, ...] | list[ModelChoice] = CURATED):
        self.providers = providers if providers is not None else providers_from_env()
        self.http = http or httpx.Client(timeout=timeout)
        self.pinned = model
        self.catalogue = tuple(catalogue)
        self.log: list[dict] = []                     # every attempt, for the eval report and for debugging
        self.last_good: tuple[str, str] | None = None # (provider, model) that answered last: tried first next time
        self.skip: set[tuple[str, str]] = set()       # models that cannot do the job (out of tokens while thinking)

    def _plan(self) -> list[tuple[Provider, str]]:
        """The picked model, then the last model that worked, then the rest of the curated list; known-bad models left out."""
        by_name = {p.name: p for p in self.providers}
        order = ([self.pinned] if self.pinned else []) + list(self.catalogue)
        plan, seen = [], set()
        for c in order:
            if c.provider in by_name and (c.provider, c.model) not in self.skip and (c.provider, c.model) not in seen:
                seen.add((c.provider, c.model))
                plan.append((by_name[c.provider], c.model))
        if self.last_good:
            head = 1 if self.pinned else 0                                  # the user's pick always goes first
            rest = sorted(plan[head:], key=lambda pm: (pm[0].name, pm[1]) != self.last_good)
            plan = plan[:head] + rest
            if plan and (plan[0][0].name, plan[0][1]) == self.last_good:
                plan.append(plan[0])                  # free models glitch: the one that worked gets a second try at the end
        return plan

    @staticmethod
    def _headers(p: Provider) -> dict:
        h = {"Authorization": f"Bearer {p.key}", "Content-Type": "application/json"}
        if p.name == "openrouter":
            h.update({"HTTP-Referer": "https://github.com/abhibarnwal707/PaperTrail", "X-Title": "PaperTrail investigator"})
        return h

    # ------------------------------------------------------------------ the call
    def chat_json(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 4000) -> tuple[dict, dict]:
        if not self.providers:
            raise LLMError("no API key found: set LLM_API_KEY (Groq, starts with gsk_) and/or OPENROUTER_API_KEY (starts with sk-or-)")
        attempts, reasons, dead = 0, [], set()
        for p, model in self._plan():
            if p.name in dead:
                continue
            if attempts >= MAX_ATTEMPTS:
                raise LLMError("no model answered: " + "; ".join(reasons[-6:]))
            attempts += 1
            body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"}}
            # many free models think before answering: keep the thinking short or it eats the whole token budget
            if p.name == "openrouter":
                body["reasoning"] = {"effort": "low", "exclude": True}
            elif p.name == "groq" and "gpt-oss" in model:
                body["reasoning_effort"] = "low"
            elif p.name == "groq" and "qwen" in model:
                body["reasoning_effort"] = "none"
            t0 = time.time()
            try:
                r = self.http.post(f"{ENDPOINTS[p.name]}/chat/completions", headers=self._headers(p), json=body)
                if r.status_code == 400 and "response_format" in r.text:          # model without JSON mode: ask in the prompt only
                    body.pop("response_format")
                    r = self.http.post(f"{ENDPOINTS[p.name]}/chat/completions", headers=self._headers(p), json=body)
            except httpx.HTTPError as e:
                self._note(p, model, None, t0, repr(e)[:120])
                reasons.append(f"{p.name}/{model}: {type(e).__name__}")
                continue
            if r.status_code in (401, 403):                                      # this key is dead: skip the whole provider
                self._note(p, model, r.status_code, t0, _err(r))
                reasons.append(f"{p.name}: key rejected ({r.status_code}: {_err(r)})")
                dead.add(p.name)
                continue
            if r.status_code in RETRYABLE or r.status_code >= 500:
                self._note(p, model, r.status_code, t0, _err(r))
                reasons.append(f"{p.name}/{model}: {r.status_code} {_err(r)}")
                continue
            if r.status_code != 200:
                self._note(p, model, r.status_code, t0, _err(r))
                reasons.append(f"{p.name}/{model}: {r.status_code} {_err(r)}")
                continue
            j = r.json()
            choice = (j.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content") or ""
            parsed = parse_json(text)
            if parsed is None:
                why = "ran out of tokens while thinking" if choice.get("finish_reason") == "length" and not text.strip() else "reply was not JSON"
                if why.startswith("ran out"):
                    self.skip.add((p.name, model))
                self._note(p, model, 200, t0, f"{why}: {text[:80]!r}")
                reasons.append(f"{p.name}/{model}: {why}")
                continue
            usage = j.get("usage") or {}
            meta = {"provider": p.name, "model": j.get("model") or model, "latency_s": round(time.time() - t0, 2),
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "attempts": attempts}
            self._note(p, model, 200, t0, "ok")
            self.last_good = (p.name, model)
            return parsed, meta
        raise LLMError("no model answered: " + "; ".join(reasons[-6:]))

    def _note(self, p: Provider, model: str, status, t0: float, info: str) -> None:
        self.log.append({"provider": p.name, "model": model, "status": status, "secs": round(time.time() - t0, 2), "info": info})


def parse_json(text: str) -> dict | None:
    """The first JSON object in a reply: plain, inside ``` fences, or after a model's <think> block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)):
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return v
        except (json.JSONDecodeError, TypeError):
            pass
    a, b = text.find("{"), text.rfind("}")
    if 0 <= a < b:
        try:
            v = json.loads(text[a:b + 1])
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _err(r: httpx.Response) -> str:
    try:
        j = r.json()
        e = j.get("error", j)
        return str(e.get("message", e) if isinstance(e, dict) else e)[:160]
    except Exception:  # noqa: BLE001
        return r.text[:160]
