"""
SONAR.AI — Reusable Semantic Text Classifier
=============================================

Domain-agnostic classification engine.  Give it a taxonomy (dict of
category descriptions + prototypes) and it classifies any text against it.

Two backends, chosen automatically:
    1. SentenceTransformer  — best quality, needs `sentence-transformers`
    2. TF-IDF (sklearn)     — good fallback, needs only `scikit-learn`

Usage:
    from taxonomy import CATEGORIES, DOMAIN_DESCRIPTION, DOMAIN_ANCHOR_KEYWORDS
    from services.classifier import TaxonomyClassifier

    clf = TaxonomyClassifier(
        taxonomy=CATEGORIES,
        domain_description=DOMAIN_DESCRIPTION,
        domain_anchor_keywords=DOMAIN_ANCHOR_KEYWORDS,
    )
    result = clf.classify("New 380kV HVDC line approved for Amprion corridor B")
    # → {"category": "grid_infrastructure", "confidence": 0.82, ...}
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Try to load sentence-transformers; fall back to TF-IDF
# ─────────────────────────────────────────────────────────────────────
_SBERT_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_AVAILABLE = True
except ImportError:
    pass

_SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos_sim
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass


# =====================================================================
# EMBEDDING BACKENDS
# =====================================================================

class _SBERTBackend:
    """Sentence-transformer embeddings (best quality)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        log.info("Loading SentenceTransformer model '%s' …", model_name)
        self.model = SentenceTransformer(model_name)
        self.name = f"SentenceTransformer({model_name})"

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False,
                                 normalize_embeddings=True)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity; inputs already L2-normalised → dot product."""
        return np.dot(a, b.T)


class _TfidfBackend:
    """TF-IDF + cosine similarity (scikit-learn fallback)."""

    def __init__(self, corpus_texts: List[str]):
        log.info("Fitting TF-IDF vectorizer on %d prototype texts …",
                 len(corpus_texts))
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            min_df=1,
        )
        self.vectorizer.fit(corpus_texts)
        self.name = "TF-IDF(sklearn)"

    def encode(self, texts: List[str]) -> np.ndarray:
        sparse = self.vectorizer.transform(texts)
        return sparse.toarray().astype(np.float32)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return sklearn_cos_sim(a, b)


# =====================================================================
# MAIN CLASSIFIER
# =====================================================================

class TaxonomyClassifier:
    """
    Reusable semantic text classifier with pluggable taxonomies.

    Parameters
    ----------
    taxonomy : dict
        {category_id: {"name", "description", "prototypes", "boost_keywords"}}
    domain_description : str
        Overall domain description for on-topic filtering.
    domain_anchor_keywords : frozenset
        Quick-check keywords; if none match, signal is likely off-topic.
    model_name : str
        SentenceTransformer model (ignored if SBERT unavailable).
    off_topic_threshold : float
        Max similarity below which a signal is marked off-topic.
    keyword_boost_weight : float
        Maximum additional score from keyword matches.
    """

    def __init__(
        self,
        taxonomy: Dict[str, Dict[str, Any]],
        domain_description: str = "",
        domain_anchor_keywords: FrozenSet[str] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        off_topic_threshold: float = 0.15,
        keyword_boost_weight: float = 0.12,
    ):
        self.taxonomy = taxonomy
        self.domain_description = domain_description
        self.domain_anchor_keywords = domain_anchor_keywords or frozenset()
        self.off_topic_threshold = off_topic_threshold
        self.keyword_boost_weight = keyword_boost_weight

        # Choose backend
        if _SBERT_AVAILABLE:
            self.backend = _SBERTBackend(model_name)
        elif _SKLEARN_AVAILABLE:
            corpus = self._gather_corpus_texts()
            self.backend = _TfidfBackend(corpus)
        else:
            raise RuntimeError(
                "No ML backend available. Install either:\n"
                "  pip install sentence-transformers   (recommended)\n"
                "  pip install scikit-learn             (lightweight fallback)"
            )
        log.info("Classifier backend: %s", self.backend.name)

        # Pre-compute category centroids
        self._category_ids: List[str] = []
        self._centroids: np.ndarray = np.empty(0)
        self._domain_centroid: Optional[np.ndarray] = None
        self._build_centroids()

        # Pre-compile keyword sets per category (lower-cased)
        self._kw_sets: Dict[str, List[str]] = {}
        for cat_id, cat_data in self.taxonomy.items():
            self._kw_sets[cat_id] = [
                kw.lower() for kw in cat_data.get("boost_keywords", [])
            ]

    # -----------------------------------------------------------------
    # Centroid construction
    # -----------------------------------------------------------------

    def _gather_corpus_texts(self) -> List[str]:
        """Collect all text for TF-IDF fitting."""
        texts: List[str] = []
        if self.domain_description:
            texts.append(self.domain_description)
        for cat_data in self.taxonomy.values():
            texts.append(cat_data.get("description", ""))
            texts.extend(cat_data.get("prototypes", []))
        return texts

    def _build_centroids(self):
        """Encode category descriptions + prototypes → centroid per category."""
        cat_ids: List[str] = []
        all_texts: List[str] = []
        slices: List[Tuple[int, int]] = []

        for cat_id, cat_data in self.taxonomy.items():
            desc = cat_data.get("description", "")
            protos = cat_data.get("prototypes", [])
            # Weight description 3× by repeating
            cat_texts = [desc] * 3 + protos
            start = len(all_texts)
            all_texts.extend(cat_texts)
            end = len(all_texts)
            slices.append((start, end))
            cat_ids.append(cat_id)

        if not all_texts:
            return

        embeddings = self.backend.encode(all_texts)  # (N, D)

        centroids = []
        for start, end in slices:
            centroid = embeddings[start:end].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)

        self._category_ids = cat_ids
        self._centroids = np.array(centroids)  # (C, D)

        # Domain centroid for on-topic detection
        if self.domain_description:
            d_emb = self.backend.encode([self.domain_description])[0]
            norm = np.linalg.norm(d_emb)
            if norm > 0:
                d_emb = d_emb / norm
            self._domain_centroid = d_emb

        log.info("Built centroids for %d categories (dim=%s)",
                 len(cat_ids),
                 self._centroids.shape[1] if len(self._centroids) > 0 else "?")

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _has_anchor_keyword(self, text_lower: str) -> bool:
        """Quick check: does the text mention ANY domain keyword?"""
        if not self.domain_anchor_keywords:
            return True
        return any(kw in text_lower for kw in self.domain_anchor_keywords)

    def _keyword_boost(self, cat_id: str, text_lower: str) -> float:
        """Compute keyword boost for a category."""
        kw_count = sum(1 for kw in self._kw_sets.get(cat_id, [])
                       if kw in text_lower)
        return min(self.keyword_boost_weight, 0.025 * kw_count)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a single text.

        Returns dict:
            category         – best category ID (or "off_topic")
            confidence        – float 0-1
            domain_relevance  – float 0-1 (similarity to domain description)
            is_on_topic       – bool
            all_scores        – {category_id: float}
        """
        if not text or not text.strip():
            return self._empty_result()

        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Encode
        text_emb = self.backend.encode([text_clean])[0]
        norm = np.linalg.norm(text_emb)
        if norm > 0:
            text_emb = text_emb / norm

        # Domain relevance
        domain_rel = 1.0
        if self._domain_centroid is not None:
            domain_rel = float(np.clip(np.dot(text_emb, self._domain_centroid), 0, 1))

        has_anchor = self._has_anchor_keyword(text_lower)

        # Quick off-topic exit
        if not has_anchor and domain_rel < self.off_topic_threshold:
            return self._off_topic_result(domain_rel, {})

        # Similarity to each category
        raw_sims = np.dot(text_emb, self._centroids.T)  # (C,)

        # Boosted scores
        boosted: Dict[str, float] = {}
        for idx, cat_id in enumerate(self._category_ids):
            boosted[cat_id] = float(raw_sims[idx]) + self._keyword_boost(cat_id, text_lower)

        return self._pick_result(boosted, domain_rel, has_anchor)

    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a batch efficiently (batched encoding).
        """
        if not texts:
            return []

        clean = [t.strip() if t else "" for t in texts]
        embeddings = self.backend.encode(clean)

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        # Domain relevance
        if self._domain_centroid is not None:
            domain_sims = np.clip(np.dot(embeddings, self._domain_centroid), 0, 1)
        else:
            domain_sims = np.ones(len(texts))

        # Category similarity matrix (N, C)
        sim_matrix = np.dot(embeddings, self._centroids.T)

        results: List[Dict[str, Any]] = []
        for i, text in enumerate(clean):
            text_lower = text.lower()
            domain_rel = float(domain_sims[i])
            has_anchor = self._has_anchor_keyword(text_lower)

            if not has_anchor and domain_rel < self.off_topic_threshold:
                results.append(self._off_topic_result(domain_rel, {}))
                continue

            boosted: Dict[str, float] = {}
            for j, cat_id in enumerate(self._category_ids):
                boosted[cat_id] = float(sim_matrix[i, j]) + self._keyword_boost(cat_id, text_lower)

            results.append(self._pick_result(boosted, domain_rel, has_anchor))

        return results

    # -----------------------------------------------------------------
    # Result helpers
    # -----------------------------------------------------------------

    def _pick_result(self, scores: Dict[str, float],
                     domain_rel: float, has_anchor: bool) -> Dict[str, Any]:
        max_score = max(scores.values()) if scores else 0.0

        # Off-topic if scores are too low
        if max_score < (self.off_topic_threshold * 0.6):
            return self._off_topic_result(domain_rel, scores)
        if max_score < self.off_topic_threshold and not has_anchor:
            return self._off_topic_result(domain_rel, scores)

        best_cat = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_cat]
        confidence = min(1.0, max(0.0, (best_score - 0.05) / 0.45))

        return {
            "category": best_cat,
            "confidence": confidence,
            "domain_relevance": domain_rel,
            "is_on_topic": True,
            "all_scores": scores,
        }

    @staticmethod
    def _off_topic_result(domain_rel: float,
                          scores: Dict[str, float]) -> Dict[str, Any]:
        return {
            "category": "off_topic",
            "confidence": 0.0,
            "domain_relevance": domain_rel,
            "is_on_topic": False,
            "all_scores": scores,
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "category": "off_topic",
            "confidence": 0.0,
            "domain_relevance": 0.0,
            "is_on_topic": False,
            "all_scores": {},
        }

    @property
    def backend_name(self) -> str:
        return self.backend.name

    @property
    def categories(self) -> List[str]:
        return list(self._category_ids)


# =====================================================================
# LLM CLASSIFIER BACKEND
# =====================================================================

class LLMClassifier:
    """
    LLM-based classifier — sends text to an LLM API for classification.

    Same interface as TaxonomyClassifier: .classify() and .batch_classify()
    return the same dict shape. Drop-in replacement.

    Set CLASSIFIER_LLM_PROVIDER env var to enable:
        "anthropic"   → uses Claude via Anthropic SDK
        "openai"      → uses GPT via OpenAI SDK
        "selfhosted"  → uses self-hosted LLM (Ollama, vLLM, llama.cpp, etc.)

    For selfhosted, also set:
        SELFHOSTED_LLM_PROVIDER=ollama|vllm|llamacpp|textgen|localai|lmstudio|custom
        SELFHOSTED_LLM_URL=http://localhost:11434
        SELFHOSTED_LLM_MODEL=llama3.2:8b

    Requires the corresponding API key env var for commercial:
        ANTHROPIC_API_KEY or OPENAI_API_KEY
    """

    def __init__(
        self,
        taxonomy: Dict[str, Dict[str, Any]],
        domain_description: str = "",
        provider: str = "anthropic",
        model: Optional[str] = None,
    ):
        self.taxonomy = taxonomy
        self.domain_description = domain_description
        self.provider = provider
        self.backend_name = f"LLM({provider})"

        self._category_ids = list(taxonomy.keys())
        self._client = None
        self._selfhosted_llm = None
        self._model = model

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
            self._model = model or "claude-sonnet-4-20250514"
        elif provider == "openai":
            import openai
            self._client = openai.OpenAI()
            self._model = model or "gpt-4o-mini"
        elif provider == "selfhosted":
            from services.selfhosted_llm import get_selfhosted_llm
            self._selfhosted_llm = get_selfhosted_llm()
            if not self._selfhosted_llm:
                raise ValueError("Self-hosted LLM not available. Check SELFHOSTED_LLM_* env vars.")
            self._model = self._selfhosted_llm.config.model
            self.backend_name = f"LLM(selfhosted:{self._selfhosted_llm.config.provider})"
        else:
            raise ValueError(f"Unknown LLM provider: {provider}. Use: anthropic, openai, or selfhosted")

        # Build the system prompt once
        self._system_prompt = self._build_system_prompt()
        log.info("LLM classifier ready: %s / %s (%d categories)",
                 provider, self._model, len(self._category_ids))

    def _build_system_prompt(self) -> str:
        cat_descriptions = []
        for cat_id, cat_data in self.taxonomy.items():
            desc = cat_data.get("description", cat_data.get("name", cat_id))
            cat_descriptions.append(f"  {cat_id}: {desc}")
        cats_block = "\n".join(cat_descriptions)

        return (
            f"You are a signal classifier for the energy/TSO domain.\n"
            f"Domain: {self.domain_description}\n\n"
            f"Categories:\n{cats_block}\n\n"
            f"Rules:\n"
            f"- Respond with ONLY a JSON object, no markdown, no backticks\n"
            f"- Fields: category (string), confidence (float 0-1), is_on_topic (bool)\n"
            f"- category must be one of the category IDs listed above, or \"off_topic\"\n"
            f"- confidence reflects how certain you are (0.9+ = very clear, 0.5-0.7 = ambiguous)\n"
            f"- is_on_topic = false if the text has nothing to do with energy/grid/TSO\n"
        )

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify a single text via LLM. Same return shape as TaxonomyClassifier."""
        if not text or not text.strip():
            return self._empty_result()

        try:
            raw = self._call_llm(text.strip())
            return self._parse_response(raw)
        except Exception as e:
            log.warning("LLM classify failed: %s", e)
            raise  # Let CascadingClassifier catch this

    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a batch. Calls LLM per-text (no batch API yet)."""
        results = []
        for text in texts:
            try:
                results.append(self.classify(text))
            except Exception:
                results.append(self._empty_result())
        return results

    def _call_llm(self, text: str) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=150,
                system=self._system_prompt,
                messages=[{"role": "user", "content": f"Classify this signal:\n\n{text[:2000]}"}],
            )
            return resp.content[0].text
        elif self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": f"Classify this signal:\n\n{text[:2000]}"},
                ],
            )
            return resp.choices[0].message.content
        elif self.provider == "selfhosted":
            response = self._selfhosted_llm.complete(
                prompt=f"Classify this signal:\n\n{text[:2000]}",
                system_prompt=self._system_prompt
            )
            if not response:
                raise RuntimeError("Self-hosted LLM returned empty response")
            return response
        raise RuntimeError(f"Unknown provider: {self.provider}")

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        import json
        # Strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(clean)
        category = data.get("category", "off_topic")
        confidence = float(data.get("confidence", 0.0))
        is_on_topic = bool(data.get("is_on_topic", category != "off_topic"))

        # Validate category
        if category not in self._category_ids and category != "off_topic":
            log.warning("LLM returned unknown category '%s', falling back", category)
            raise ValueError(f"Unknown category: {category}")

        return {
            "category": category,
            "confidence": confidence,
            "domain_relevance": confidence if is_on_topic else 0.0,
            "is_on_topic": is_on_topic,
            "all_scores": {},  # LLM doesn't produce per-category scores
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "category": "off_topic",
            "confidence": 0.0,
            "domain_relevance": 0.0,
            "is_on_topic": False,
            "all_scores": {},
        }

    @property
    def categories(self) -> List[str]:
        return list(self._category_ids)


# =====================================================================
# CASCADING CLASSIFIER — LLM primary, SentenceTransformer fallback
# =====================================================================

class CascadingClassifier:
    """
    Tries LLM first, falls back to TaxonomyClassifier on failure.

    Same .classify() / .batch_classify() interface as both.
    Drop-in replacement anywhere TaxonomyClassifier is used.
    """

    def __init__(
        self,
        primary: LLMClassifier,
        fallback: TaxonomyClassifier,
    ):
        self.primary = primary
        self.fallback = fallback
        self.backend_name = f"Cascading({primary.backend_name} → {fallback.backend_name})"
        self._stats = {"primary_ok": 0, "fallback_used": 0}
        log.info("Cascading classifier: %s", self.backend_name)

    def classify(self, text: str) -> Dict[str, Any]:
        try:
            result = self.primary.classify(text)
            self._stats["primary_ok"] += 1
            return result
        except Exception as e:
            log.debug("Primary classifier failed (%s), using fallback", e)
            self._stats["fallback_used"] += 1
            return self.fallback.classify(text)

    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results

    @property
    def categories(self) -> List[str]:
        return self.fallback.categories

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


# =====================================================================
# SINGLETON for the default TSO classifier
# =====================================================================

_default_classifier: Optional[Any] = None


def get_classifier() -> Any:
    """
    Get (or lazily create) the default classifier.

    Priority:
        1. LLM (if CLASSIFIER_LLM_PROVIDER is set + API key/server available)
           → wrapped in CascadingClassifier with SentenceTransformer fallback
        2. SentenceTransformer (if sentence-transformers installed)
        3. TF-IDF (scikit-learn fallback)
    
    LLM providers:
        - anthropic: Claude via Anthropic API (requires ANTHROPIC_API_KEY)
        - openai: GPT via OpenAI API (requires OPENAI_API_KEY)
        - selfhosted: Local LLM via Ollama, vLLM, llama.cpp, etc.
          (requires SELFHOSTED_LLM_PROVIDER and SELFHOSTED_LLM_URL)
    """
    global _default_classifier
    if _default_classifier is not None:
        return _default_classifier

    import os
    from taxonomy import (CATEGORIES, DOMAIN_ANCHOR_KEYWORDS,
                          DOMAIN_DESCRIPTION)

    # Always build the local classifier (used standalone or as fallback)
    local = TaxonomyClassifier(
        taxonomy=CATEGORIES,
        domain_description=DOMAIN_DESCRIPTION,
        domain_anchor_keywords=DOMAIN_ANCHOR_KEYWORDS,
    )

    # Check if LLM classifier is requested
    llm_provider = os.getenv("CLASSIFIER_LLM_PROVIDER", "").strip().lower()
    if llm_provider in ("anthropic", "openai", "selfhosted"):
        try:
            llm = LLMClassifier(
                taxonomy=CATEGORIES,
                domain_description=DOMAIN_DESCRIPTION,
                provider=llm_provider,
                model=os.getenv("CLASSIFIER_LLM_MODEL"),
            )
            _default_classifier = CascadingClassifier(primary=llm, fallback=local)
            log.info("Using cascading classifier: LLM (%s) primary, local fallback", llm_provider)
        except Exception as e:
            log.warning("Failed to init LLM classifier (%s), using local only: %s", llm_provider, e)
            _default_classifier = local
    else:
        _default_classifier = local

    return _default_classifier
