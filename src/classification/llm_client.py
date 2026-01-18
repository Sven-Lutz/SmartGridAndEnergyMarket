#llm_client.py
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

from ..utils.config import project_root
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient:
    """
    Wrapper um ein LLM (OpenAI, Ollama, Gemini, …) basierend auf configs/llm.yml.

    Erwartete Struktur von config (aus llm.yml):
      - llm: { enabled, dry_run, default_provider, default_profile, retry: {...}, ... }
      - providers: { openai: {...}, ollama: { type, base_url, request_timeout_seconds, models: {...} }, ... }
      - prompts: { name: 'prompts/xyz.txt', ... }
      - profiles: { measure_gate_closed_taxonomy_ollama: { provider, model, prompt_ref, generation: {...}, ... }, ... }

    Verhalten:
      - disabled / dry_run → Stub
      - OpenAI ohne API-Key → Stub
      - Ollama ohne API-Key → normal (lokal via base_url)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: str | None = None,
        profile: str | None = None,
    ) -> None:
        self.config = config or {}

        llm_cfg = self.config.get("llm", {}) or {}
        providers_cfg = self.config.get("providers", {}) or {}
        prompts_cfg = self.config.get("prompts", {}) or {}
        profiles_cfg = self.config.get("profiles", {}) or {}

        self.enabled: bool = bool(llm_cfg.get("enabled", True))
        self.dry_run: bool = bool(llm_cfg.get("dry_run", False))

        retry_cfg = llm_cfg.get("retry", {}) or {}
        self.retry_max_attempts: int = int(retry_cfg.get("max_attempts", 1))
        self.retry_backoff_seconds: float = float(retry_cfg.get("backoff_seconds", 0.0))
        self.retry_backoff_factor: float = float(retry_cfg.get("backoff_factor", 1.0))

        default_profile = llm_cfg.get("default_profile", "classification_baseline")
        self.profile_name: str = profile or default_profile
        self.profile_cfg: Dict[str, Any] = profiles_cfg.get(self.profile_name, {}) or {}

        if not self.profile_cfg:
            logger.warning(
                "LLMClient: Profil '%s' nicht in llm.yml/profiles gefunden. Fallback auf leeres Profil.",
                self.profile_name,
            )

        # Provider (Profil > global)
        self.provider_name: str = self.profile_cfg.get(
            "provider", llm_cfg.get("default_provider", "openai")
        )
        self.providers_cfg = providers_cfg
        self.provider_cfg: Dict[str, Any] = providers_cfg.get(self.provider_name, {}) or {}
        self.provider_type: str = self.provider_cfg.get("type", self.provider_name)

        # Modell bestimmen
        model_key: Optional[str] = self.profile_cfg.get("model")
        models_cfg = self.provider_cfg.get("models", {}) or {}
        model_cfg: Dict[str, Any] = models_cfg.get(model_key, {}) if model_key else {}

        # Wenn model_key nicht als Key in models_cfg existiert, kann es ein direktes Modell sein
        if model_key and not model_cfg:
            self.model_name = model_key
        else:
            if self.provider_type == "ollama":
                self.model_name = model_cfg.get("model_name") or "mistral"
            else:
                self.model_name = model_cfg.get("model_name") or "gpt-4o"

        # CLI-Override: "provider:model" oder nur "model"
        if model is not None:
            if ":" in model:
                prov, mdl = model.split(":", 1)
                self.provider_name = prov
                self.provider_cfg = providers_cfg.get(self.provider_name, {}) or {}
                self.provider_type = self.provider_cfg.get("type", self.provider_name)
                self.model_name = mdl
                model_cfg = {}
            else:
                self.model_name = model

        # Generationseinstellungen
        gen_cfg = self.profile_cfg.get("generation", {}) or {}

        # OpenAI Responses API nutzt max_output_tokens; wir akzeptieren beides.
        max_tok = gen_cfg.get("max_tokens", None)
        if max_tok is None:
            max_tok = gen_cfg.get("max_output_tokens", None)

        self.temperature: float = float(gen_cfg.get("temperature", model_cfg.get("default_temperature", 0.0)))
        self.top_p: float = float(gen_cfg.get("top_p", 1.0))
        self.max_tokens: int = int(max_tok if max_tok is not None else model_cfg.get("max_tokens", 2048))

        # API-Key
        self.api_key_env: str = self.provider_cfg.get("api_key_env", "LLM_API_KEY")
        raw_key = os.getenv(self.api_key_env)
        self.api_key: Optional[str] = raw_key.strip() if isinstance(raw_key, str) and raw_key.strip() else None

        if self.provider_type == "openai":
            if not self.api_key:
                logger.warning(
                    "LLMClient: Kein API-Key in ENV '%s'. OpenAI-Aufrufe laufen im Stub-Modus.",
                    self.api_key_env,
                )
        else:
            # Für Ollama & andere lokale Provider ist das normal.
            self.api_key = None

        # Prompt laden
        prompt_ref: str | None = self.profile_cfg.get("prompt_ref")

        prompt_rel: str
        if prompt_ref:
            if prompt_ref in prompts_cfg:
                prompt_rel = cast(str, prompts_cfg[prompt_ref])
            else:
                prompt_rel = prompt_ref
        else:
            prompt_rel = cast(str, prompts_cfg.get("measure_extraction", "prompts/measure_extraction.txt"))

        self.prompt_path = self._resolve_prompt_path(prompt_rel)
        if not self.prompt_path.exists():
            logger.warning("Prompt-Datei nicht gefunden: %s", self.prompt_path)
            self.prompt_template = ""
        else:
            self.prompt_template = self.prompt_path.read_text(encoding="utf-8")

        self.system_prompt: str = self.profile_cfg.get(
            "system_prompt",
            (
                "You are an information extraction and classification component for a climate-policy knowledge base. "
                "Return ONLY valid JSON. No markdown. No extra text."
            ),
        )

        logger.info(
            "LLMClient initialisiert: provider=%s (type=%s), model=%s, profile=%s, enabled=%s, dry_run=%s",
            self.provider_name,
            self.provider_type,
            self.model_name,
            self.profile_name,
            self.enabled,
            self.dry_run,
        )

    # -------------------------------------------------------
    # Batchprocessing
    # -------------------------------------------------------
    def extract_batch(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[str]] = None,
        urls: Optional[Sequence[str | None]] = None,
        scope: str | None = None,
    ) -> List[Dict[str, Any]]:
        texts_list: list[str] = list(texts)
        titles_list: list[str] = list(titles) if titles is not None else [""] * len(texts_list)
        urls_list: list[str | None] = list(urls) if urls is not None else [None] * len(texts_list)

        if not self.enabled:
            logger.info("LLMClient.extract_batch: LLM global disabled → Stub.")
            return self._stub_batch(texts_list, titles_list, urls_list, scope, label_source="llm_disabled")

        if self.dry_run:
            logger.info("LLMClient.extract_batch: dry_run=true → Stub.")
            return self._stub_batch(texts_list, titles_list, urls_list, scope, label_source="llm_dry_run")

        if self.provider_type == "ollama":
            results: List[Dict[str, Any]] = []
            for txt, title, url in zip(texts_list, titles_list, urls_list):
                user_prompt = self._build_user_prompt(text=txt, title=title, url=url, scope=scope)
                try:
                    obj = self._ollama_chat(system_prompt=self.system_prompt, user_prompt=user_prompt)
                except Exception as e:
                    logger.error("Ollama-Request fehlgeschlagen: %s", e)
                    results.append(self._error_result(txt, title, url, scope, label_source="llm_error_ollama"))
                    continue
                results.append(self._map_result(obj, txt, title, url, scope, default_label_source="llm_ollama"))
            return results

        if self.provider_type == "openai":
            if not self.api_key:
                return self._stub_batch(texts_list, titles_list, urls_list, scope, label_source="llm_stub")

            try:
                from openai import OpenAI  # type: ignore
            except ImportError:
                logger.error("openai-Bibliothek nicht installiert. → Stub.")
                return self._stub_batch(texts_list, titles_list, urls_list, scope, label_source="llm_stub")

            client = OpenAI(api_key=self.api_key)
            results: List[Dict[str, Any]] = []

            for txt, title, url in zip(texts_list, titles_list, urls_list):
                user_prompt = self._build_user_prompt(text=txt, title=title, url=url, scope=scope)
                try:
                    resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    content = resp.choices[0].message.content or "{}"
                except Exception as e:
                    logger.error("LLM-Request fehlgeschlagen: %s", e)
                    results.append(self._error_result(txt, title, url, scope, label_source="llm_error"))
                    continue

                try:
                    obj = json.loads(content)
                    if not isinstance(obj, dict):
                        raise ValueError("LLM-Antwort ist kein JSON-Objekt.")
                except Exception as e:
                    logger.error("Konnte LLM-JSON nicht parsen: %s; content=%r", e, content)
                    obj = {}

                results.append(self._map_result(obj, txt, title, url, scope, default_label_source="llm"))
            return results

        logger.warning("LLMClient.extract_batch: Provider-Typ '%s' nicht implementiert. → Stub.", self.provider_type)
        return self._stub_batch(texts_list, titles_list, urls_list, scope, label_source="llm_stub")

    # -------------------------------------------------------
    # Stub / Error
    # -------------------------------------------------------
    def _stub_batch(
        self,
        texts: List[str],
        titles: List[str],
        urls: List[str | None],
        scope: str | None,
        label_source: str = "llm_stub",
    ) -> List[Dict[str, Any]]:
        return [self._error_result(txt, t, u, scope, label_source=label_source) for txt, t, u in zip(texts, titles, urls)]

    def _error_result(
        self,
        txt: str,
        title: str | None,
        url: str | None,
        scope: str | None,
        label_source: str,
    ) -> Dict[str, Any]:
        return {
            "measure_id": None,
            "title": title or None,
            "snippet": (txt or "")[:300],
            "source_url": url or None,
            "source_scope": scope,
            # legacy fields (for inspect-measures)
            "policy_area": None,
            "instrument_type": None,
            "instrument_subtype": None,
            "climate_dimension": None,
            "target_sector": None,
            "digitalization_level": None,
            "funding_program_level": None,
            "funding_program_code": None,
            "confidence_score": 0.0,
            # new gate+taxonomy fields
            "is_policy_measure": None,
            "relevance_confidence": None,
            "policy_field": None,
            "measure_type": None,
            "target_group": None,
            "jurisdiction_level": None,
            "classification_confidence": None,
            "short_title": None,
            "actor": None,
            "instrument": None,
            "rationale": None,
            "evidence_quotes": None,
            "label_source": label_source,
        }

    # -------------------------------------------------------
    # Mapping: NEW schema + legacy compatibility
    # -------------------------------------------------------
    def _map_result(
        self,
        obj: Dict[str, Any],
        txt: str,
        title: str | None,
        url: str | None,
        scope: str | None,
        default_label_source: str,
    ) -> Dict[str, Any]:
        # --- aliases for schema drift (prompt/model variants) ---
        policy_area = obj.get("policy_area", obj.get("policy_field"))
        instrument_type = obj.get("instrument_type", obj.get("measure_type"))
        target_sector = obj.get("target_sector", obj.get("target_group"))
        funding_program_level = obj.get("funding_program_level", obj.get("jurisdiction_level"))

        # confidence: prefer explicit schema field; else fall back to common alternatives
        conf = obj.get("confidence_score")
        if conf is None:
            conf = obj.get("classification_confidence")
        if conf is None:
            conf = obj.get("relevance_confidence")
        # make inspect-measures stable: avoid missing confidence unless explicitly desired
        if conf is None:
            conf = 0.0

        return {
            "measure_id": obj.get("measure_id"),
            "title": obj.get("measure_title") or obj.get("title") or title or None,
            "snippet": obj.get("description") or obj.get("snippet") or txt[:300],
            "source_url": obj.get("source_url") or url or None,
            "source_scope": obj.get("source_scope") or scope,

            # MVP schema fields (expected by inspect-measures)
            "policy_area": policy_area,
            "instrument_type": instrument_type,
            "instrument_subtype": obj.get("instrument_subtype"),
            "climate_dimension": obj.get("climate_dimension"),
            "target_sector": target_sector,
            "digitalization_level": obj.get("digitalization_level"),
            "funding_program_level": funding_program_level,
            "funding_program_code": obj.get("funding_program_code"),

            # provenance
            "label_source": obj.get("label_source") or default_label_source,
            "confidence_score": conf,

            # keep extra fields if model returns them (harmless, useful later)
            "is_policy_measure": obj.get("is_policy_measure"),
            "relevance_confidence": obj.get("relevance_confidence"),
            "classification_confidence": obj.get("classification_confidence"),
            "actor": obj.get("actor"),
            "instrument": obj.get("instrument"),
            "rationale": obj.get("rationale"),
            "evidence_quotes": obj.get("evidence_quotes"),
        }

    # -------------------------------------------------------
    # Prompt
    # -------------------------------------------------------
    def _build_user_prompt(self, text: str, title: str | None = None, url: str | None = None, scope: str | None = None) -> str:
        base = (self.prompt_template or "").strip()
        if not base:
            base = "Return only a single JSON object for the given snippet."

        try:
            return base.format(text=text, title=title or "", url=url or "", scope=scope or "")
        except Exception:
            return (
                base
                + "\n\nDOCUMENT_TITLE: " + (title or "")
                + "\nDOCUMENT_URL: " + (url or "")
                + "\nSCOPE: " + (scope or "")
                + "\nDOCUMENT_TEXT:\n" + (text or "")
            )

    def extract_measures_from_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = doc.get("text") or ""
        title = doc.get("title") or ""
        url = doc.get("url") or doc.get("source_url") or None
        return self.extract_batch(texts=[text], titles=[title], urls=[url], scope=doc.get("municipality_id"))

    # -------------------------------------------------------
    # Prompt Path Resolver
    # -------------------------------------------------------
    def _resolve_prompt_path(self, prompt_rel: str) -> Path:
        p = Path(prompt_rel)
        if p.is_absolute():
            return p

        root = project_root()

        cand1 = root / prompt_rel
        if cand1.exists():
            return cand1

        cand2 = root / "configs" / prompt_rel
        if cand2.exists():
            return cand2

        return cand1

    # -------------------------------------------------------
    # Ollama Helpers
    # -------------------------------------------------------
    def _extract_first_json_object(self, s: str) -> dict:
        s = (s or "").strip()
        if not s:
            return {}

        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

        start = s.find("{")
        if start == -1:
            return {}

        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            return obj if isinstance(obj, dict) else {}
                        except Exception:
                            return {}

        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _ollama_chat(self, system_prompt: str, user_prompt: str) -> dict:
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("httpx nicht installiert. Bitte `pip install httpx`.") from e

        base_url = (self.provider_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        }

        last_err: Exception | None = None
        timeout_s = float(self.provider_cfg.get("request_timeout_seconds", 240.0))
        timeout = httpx.Timeout(timeout_s, connect=30.0, read=timeout_s, write=30.0, pool=30.0)

        for attempt in range(1, max(1, self.retry_max_attempts) + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    r = client.post(url, json=payload)
                    r.raise_for_status()
                    data = r.json()

                content = (((data or {}).get("message") or {}).get("content")) or ""
                obj = self._extract_first_json_object(content)
                return obj if isinstance(obj, dict) else {}
            except Exception as e:
                last_err = e
                logger.warning("Ollama call failed (attempt %s/%s): %s", attempt, self.retry_max_attempts, e)
                if attempt < self.retry_max_attempts and self.retry_backoff_seconds > 0:
                    sleep_s = self.retry_backoff_seconds * (self.retry_backoff_factor ** (attempt - 1))
                    time.sleep(float(sleep_s))

        raise RuntimeError(f"Ollama call failed after {self.retry_max_attempts} attempts: {last_err}") from last_err
