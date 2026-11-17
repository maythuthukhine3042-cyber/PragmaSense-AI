"""Thin LLM client. Supports Google Gemini (free tier), any OpenAI-compatible
API, and an offline MOCK backend that is only for testing the code.

Every response is cached in results/llm_cache.jsonl, so if a run stops because
of a rate limit you can just run the same command again and it continues.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

CACHE_PATH = Path(__file__).resolve().parent.parent / "results" / "llm_cache.jsonl"


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, backend=None, model=None, temperature=0.0, sleep=0.0, use_cache=True):
        self.backend = (backend or os.environ.get("LLM_BACKEND", "gemini")).lower()
        if self.backend == "gemini":
            self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
            self.api_key = os.environ.get("GEMINI_API_KEY")
        elif self.backend == "openai":
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        elif self.backend == "mock":
            self.model = "mock"
            self.api_key = None
        else:
            raise ValueError(f"unknown backend {self.backend}")
        if self.backend != "mock" and not self.api_key:
            raise LLMError(f"No API key found. Set {'GEMINI_API_KEY' if self.backend == 'gemini' else 'OPENAI_API_KEY'} "
                           "in your environment (see README).")
        self.temperature = temperature
        self.sleep = sleep
        self.use_cache = use_cache and self.backend != "mock"
        self.n_calls = 0
        self.total_latency = 0.0
        self._cache = self._load_cache() if self.use_cache else {}

    # ----------------------------------------------------------------- cache
    def _key(self, prompt):
        return hashlib.sha256(f"{self.backend}|{self.model}|{self.temperature}|{prompt}".encode()).hexdigest()

    def _load_cache(self):
        cache = {}
        if CACHE_PATH.exists():
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        cache[d["key"]] = d
                    except json.JSONDecodeError:
                        pass
        return cache

    def _save(self, key, text, latency):
        CACHE_PATH.parent.mkdir(exist_ok=True)
        rec = {"key": key, "model": self.model, "text": text, "latency": latency}
        self._cache[key] = rec
        with open(CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ call
    def generate(self, prompt):
        """Returns (text, latency_seconds). Cached answers return their original latency."""
        key = self._key(prompt)
        if self.use_cache and key in self._cache:
            rec = self._cache[key]
            return rec["text"], rec["latency"]

        if self.backend == "mock":
            return _mock_answer(prompt), 0.0

        for attempt in range(6):
            t0 = time.time()
            try:
                text = self._call(prompt)
                latency = time.time() - t0
                break
            except _RateLimited as e:
                wait = min(60, 5 * 2 ** attempt)
                print(f"   [rate limited, waiting {wait}s] {e}")
                time.sleep(wait)
        else:
            raise LLMError("Too many rate-limit errors. You may have hit the daily quota; "
                           "try again later - finished calls are cached.")
        self.n_calls += 1
        self.total_latency += latency
        if self.use_cache:
            self._save(key, text, latency)
        if self.sleep:
            time.sleep(self.sleep)
        return text, latency

    def _call(self, prompt):
        if self.backend == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": self.temperature}}
            r = requests.post(url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=120)
            if r.status_code in (429, 500, 503):
                raise _RateLimited(r.text[:200])
            if r.status_code != 200:
                raise LLMError(f"Gemini error {r.status_code}: {r.text[:500]}")
            data = r.json()
            try:
                parts = data["candidates"][0]["content"]["parts"]
                # skip "thought" parts if the model returns them
                return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            except (KeyError, IndexError):
                raise LLMError(f"Unexpected Gemini response: {json.dumps(data)[:500]}")
        else:
            url = f"{self.base_url}/chat/completions"
            body = {"model": self.model, "temperature": self.temperature,
                    "messages": [{"role": "user", "content": prompt}]}
            r = requests.post(url, json=body, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=120)
            if r.status_code in (429, 500, 503):
                raise _RateLimited(r.text[:200])
            if r.status_code != 200:
                raise LLMError(f"API error {r.status_code}: {r.text[:500]}")
            return r.json()["choices"][0]["message"]["content"].strip()


class _RateLimited(Exception):
    pass


def _mock_answer(prompt):
    """Offline stand-in used ONLY to test the code path. Its output is not a
    real translation and must never be used for results."""
    if "PREVIOUS TRANSLATION" in prompt:
        prev = prompt.split("PREVIOUS TRANSLATION:")[1].split("\n")[1].strip()
        import re
        sug = re.findall(r"such as: ([^.\n]+)", prompt) + re.findall(r"Use ([^.\n]+) instead", prompt)
        add = sug[0].split(",")[0].strip() if sug else ""
        return f"{prev} {add}".strip()
    if "Translation:" in prompt and "step by step" in prompt:
        return "Reasoning: (mock)\nTranslation: [MOCK] ဘာသာပြန်"
    return "[MOCK] ဘာသာပြန်"
