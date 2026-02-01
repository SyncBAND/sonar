"""
SONAR.AI / AGTS NLP Processor  (v3 — Semantic Classification)
==============================================================

Replaces keyword-matching classifier with the semantic
TaxonomyClassifier (sentence-transformer or TF-IDF backend).

Pipeline:
  1. Semantic classification  →  category + confidence + domain_relevance
  2. Deterministic mappings   →  Amprion task, business area, strategic domain
  3. Regex enrichment         →  project linking, regulation mentions
  4. spaCy NER (optional)     →  named entity extraction
  5. Quality scoring          →  incorporates domain_relevance

Public API (unchanged from v2 — drop-in replacement):
  processor = get_nlp_processor()
  result    = processor.classify_signal(title, content)
  quality   = processor.calculate_quality_score(...)
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Semantic classifier (taxonomy.py + services/classifier.py)
# ─────────────────────────────────────────────────────────────────────
from services.classifier import get_classifier, TaxonomyClassifier
from taxonomy import (
    CATEGORIES,
    CATEGORY_TO_AMPRION_TASK,
    CATEGORY_TO_BUSINESS_AREAS,
    CATEGORY_TO_STRATEGIC_DOMAIN,
)

# ─────────────────────────────────────────────────────────────────────
# spaCy (optional — used only for named-entity extraction)
# ─────────────────────────────────────────────────────────────────────
try:
    import spacy
    _nlp_spacy = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    _nlp_spacy = None
    SPACY_AVAILABLE = False
    log.info("spaCy not available — using basic keyword extraction only")


# =====================================================================
# NLP PROCESSOR
# =====================================================================

class AGTSNLPProcessor:
    """
    Multi-dimensional signal classifier for the AGTS framework.

    Classification layers:
        1. TSO Category          — semantic (TaxonomyClassifier)
        2. Strategic Domain      — deterministic mapping
        3. Amprion Task          — deterministic mapping
        4. Business Area         — deterministic mapping
        5. Linked Projects       — regex pattern matching
        6. Keywords & Entities   — TF-IDF / spaCy
        7. Quality Score         — weighted composite
    """

    def __init__(self):
        # Semantic classifier — lazily created singleton
        self._clf: TaxonomyClassifier = get_classifier()
        log.info("NLP Processor using %s backend", self._clf.backend_name)

    # =================================================================
    # CORE CLASSIFICATION
    # =================================================================

    def classify_signal(self, title: str, content: str = "") -> Dict[str, Any]:
        """
        Full multi-dimensional classification of a signal.

        Returns dict compatible with full.py step_process expectations.
        """
        text = f"{title} {content or ''}".strip()
        text_lower = text.lower()

        # ── 1. Semantic classification ──────────────────────────────
        clf_result = self._clf.classify(text)
        category = clf_result["category"]
        confidence = clf_result["confidence"]
        domain_relevance = clf_result["domain_relevance"]
        is_on_topic = clf_result["is_on_topic"]

        # ── 2. Deterministic Amprion enrichment ─────────────────────
        amprion_task = CATEGORY_TO_AMPRION_TASK.get(category)
        business_areas = CATEGORY_TO_BUSINESS_AREAS.get(category, [])
        strategic_domain = CATEGORY_TO_STRATEGIC_DOMAIN.get(category)

        # ── 3. Project linking (regex — works well, keep it) ────────
        linked_projects = self._find_linked_projects(text_lower)

        # Project multiplier
        is_mega = bool(linked_projects)
        project_multiplier = 1.5 if is_mega else 1.0

        # ── 4. Keywords ────────────────────────────────────────────
        keywords = self.extract_keywords(text)

        # ── 5. Named entities ──────────────────────────────────────
        entities = self.extract_entities(text)

        # ── 6. Regulation mentions ─────────────────────────────────
        regulations = self._extract_regulation_mentions(text)
        if regulations:
            entities["regulations"] = regulations

        return {
            # Core classification
            "tso_category": category,
            "tso_category_confidence": confidence,
            "domain_relevance": domain_relevance,
            "is_on_topic": is_on_topic,

            # Amprion dimensions (deterministic from category)
            "strategic_domain": strategic_domain,
            "strategic_sub_category": None,
            "architectural_layer": None,          # Not used in current pipeline
            "amprion_task": amprion_task,
            "business_area": (
                business_areas[0] if business_areas else None
            ),
            "business_area_code": (
                business_areas[0] if business_areas else None
            ),
            "linked_projects": linked_projects,
            "is_mega_project_related": is_mega,
            "project_multiplier": project_multiplier,

            # Impact type (derived from category + confidence)
            "impact_type": self._derive_impact_type(category, confidence),

            # NLP outputs
            "keywords": keywords,
            "entities": entities,
            "maturity_indicators": {},            # Placeholder for future use
        }

    def batch_classify_signals(
        self,
        signals: List[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Batch-classify a list of (title, content) pairs.
        Uses the classifier's batch_encode for efficiency.
        """
        texts = [f"{t} {c or ''}".strip() for t, c in signals]
        clf_results = self._clf.batch_classify(texts)

        results = []
        for (title, content), clf_r in zip(signals, clf_results):
            text = f"{title} {content or ''}".strip()
            text_lower = text.lower()
            category = clf_r["category"]

            results.append({
                "tso_category": category,
                "tso_category_confidence": clf_r["confidence"],
                "domain_relevance": clf_r["domain_relevance"],
                "is_on_topic": clf_r["is_on_topic"],
                "strategic_domain": CATEGORY_TO_STRATEGIC_DOMAIN.get(category),
                "strategic_sub_category": None,
                "architectural_layer": None,
                "amprion_task": CATEGORY_TO_AMPRION_TASK.get(category),
                "business_area": (CATEGORY_TO_BUSINESS_AREAS.get(category, [None]) or [None])[0],
                "business_area_code": (CATEGORY_TO_BUSINESS_AREAS.get(category, [None]) or [None])[0],
                "linked_projects": self._find_linked_projects(text_lower),
                "is_mega_project_related": bool(self._find_linked_projects(text_lower)),
                "project_multiplier": 1.5 if self._find_linked_projects(text_lower) else 1.0,
                "impact_type": self._derive_impact_type(category, clf_r["confidence"]),
                "keywords": self.extract_keywords(text),
                "entities": self.extract_entities(text),
                "maturity_indicators": {},
            })
        return results

    # =================================================================
    # ENTITY EXTRACTION
    # =================================================================

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities: Dict[str, List[str]] = {
            "organizations": [],
            "locations": [],
            "technologies": [],
            "projects": [],
            "regulations": [],
        }
        if not text:
            return entities

        if SPACY_AVAILABLE and _nlp_spacy:
            doc = _nlp_spacy(text[:50_000])
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                elif ent.label_ in ("GPE", "LOC"):
                    entities["locations"].append(ent.text)
                elif ent.label_ == "PRODUCT":
                    entities["technologies"].append(ent.text)

        entities["projects"] = self._extract_project_mentions(text)
        entities["regulations"] = self._extract_regulation_mentions(text)

        # De-duplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        return entities

    # =================================================================
    # KEYWORD EXTRACTION
    # =================================================================

    _STOPWORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are",
        "were", "been", "be", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must",
        "shall", "can", "this", "that", "these", "those", "it", "its",
        "they", "their", "them", "we", "our", "us", "you", "your", "he",
        "she", "his", "her", "which", "who", "whom", "what", "when",
        "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "also", "now", "then", "here", "there", "new", "about", "into",
        "over", "after", "before", "between", "under", "through",
    })

    def extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        """Extract top keywords from text (simple TF approach)."""
        if not text:
            return []
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]*[a-zA-Z0-9]\b", text.lower())
        filtered = [w for w in words if w not in self._STOPWORDS and len(w) > 2]
        counts = Counter(filtered)
        return [w for w, _ in counts.most_common(top_n)]

    # =================================================================
    # REGEX ENRICHMENT
    # =================================================================

    @staticmethod
    def _find_linked_projects(text_lower: str) -> List[str]:
        """Find mentions of Amprion mega-projects via regex."""
        projects = []
        patterns = {
            "Korridor B":      [r"korridor\s*b", r"dc31", r"dc32"],
            "Rhein-Main Link": [r"rhein[-\s]?main\s*link?"],
            "A-Nord":          [r"a[-\s]?nord"],
            "Ultranet":        [r"ultranet"],
            "SuedLink":        [r"suedlink", r"sued[-\s]?link"],
            "BalWin1":         [r"balwin\s*1"],
            "BalWin2":         [r"balwin\s*2"],
            "NeuLink":         [r"neulink"],
            "hybridge":        [r"hybridge"],
        }
        for project, pats in patterns.items():
            for pat in pats:
                if re.search(pat, text_lower, re.IGNORECASE):
                    projects.append(project)
                    break
        return list(set(projects))

    @staticmethod
    def _extract_project_mentions(text: str) -> List[str]:
        """Extract Amprion project mentions (alias for linked_projects)."""
        return AGTSNLPProcessor._find_linked_projects(text.lower())

    @staticmethod
    def _extract_regulation_mentions(text: str) -> List[str]:
        """Extract regulatory standard mentions."""
        regulations = []
        patterns = [
            r"iec\s*\d{4,5}", r"ieee\s*\d{3,5}", r"en\s*\d{4,5}",
            r"iso\s*\d{4,5}", r"nerc\s*cip", r"nis\s*2", r"kritis",
            r"red\s*iii", r"fit\s*for\s*55", r"aregv", r"enwg",
        ]
        for pat in patterns:
            regulations.extend(re.findall(pat, text, re.IGNORECASE))
        return list(set(regulations))

    # =================================================================
    # IMPACT TYPE DERIVATION
    # =================================================================

    @staticmethod
    def _derive_impact_type(category: str, confidence: float) -> str:
        """
        Derive strategic nature from category.

        Instead of unreliable keyword matching, use category-level heuristics:
        - Core grid/stability/cyber = Transformational
        - Emerging tech (AI, digital twins, offshore) = Disruptor
        - Everything else = Accelerator
        """
        transformational = {
            "grid_infrastructure", "grid_stability",
            "cybersecurity_ot", "renewables_integration", "energy_storage",
        }
        disruptor = {
            "offshore_systems", "ai_grid_optimization",
            "digital_twin_simulation", "hydrogen_p2g",
        }
        if category in transformational:
            return "Transformational"
        if category in disruptor:
            return "Disruptor"
        return "Accelerator"

    # =================================================================
    # QUALITY SCORING
    # =================================================================

    def calculate_quality_score(
        self,
        source_type: str,
        source_quality: float,
        text_length: int,
        has_entities: bool,
        domain_relevance: float = 1.0,
    ) -> float:
        """
        Calculate signal quality score (0.0 – 1.0).

        Incorporates domain_relevance so off-topic signals get
        lower quality, reducing their weight in trend scoring.
        """
        # Source type credibility
        SOURCE_WEIGHTS = {
            "patent": 0.95, "research": 0.90, "regulatory": 0.90,
            "tso": 0.85, "research_org": 0.85, "conference": 0.75,
            "news": 0.70, "startup": 0.60,
        }
        source_w = SOURCE_WEIGHTS.get(source_type, 0.50)

        # Text length score (longer = richer, up to a point)
        length_score = min(text_length / 1000, 1.0)

        # Entity richness
        entity_score = 0.8 if has_entities else 0.5

        # Domain relevance gate:
        #   1.0 if clearly on-topic, scales down toward 0.3 for borderline
        domain_factor = 0.3 + 0.7 * min(1.0, max(0.0, domain_relevance / 0.5))

        quality = (
            source_w * 0.35
            + source_quality * 0.20
            + length_score * 0.15
            + entity_score * 0.10
            + domain_factor * 0.20
        )
        return round(min(max(quality, 0.0), 1.0), 3)


# =====================================================================
# SINGLETON
# =====================================================================

_nlp_processor: Optional[AGTSNLPProcessor] = None


def get_nlp_processor() -> AGTSNLPProcessor:
    """Get singleton NLP processor instance."""
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = AGTSNLPProcessor()
    return _nlp_processor


# =====================================================================
# QUICK SELF-TEST
# =====================================================================

if __name__ == "__main__":
    import sys
    processor = get_nlp_processor()

    tests = [
        ("New HVDC converter for Korridor B enables 4GW transmission",
         "grid_infrastructure"),
        ("V2G pilot: EVs provide grid frequency response",
         "e_mobility_v2g"),
        ("Biogas injection into German gas grid reaches record levels",
         "biogas_biomethane"),
        ("Ransomware attack disrupts SCADA at European utility",
         "cybersecurity_ot"),
        ("Virtual power plant aggregates 10,000 heat pumps",
         "flexibility_vpp"),
        ("BNetzA publishes updated network development plan",
         "regulatory_policy"),
        ("Anthropic Targets $20 Billion Raise",
         "off_topic"),
        ("Tesla to End Production of Model S and Model X",
         "off_topic"),
        ("Coal-fired generation rose during Winter Storm Fern",
         "power_generation"),
        ("Grid-scale 400MWh battery commissioned for peak shaving",
         "energy_storage"),
    ]

    print("=" * 75)
    print(f"AGTS NLP Processor v3 — Backend: {processor._clf.backend_name}")
    print("=" * 75)

    correct = 0
    for title, expected in tests:
        r = processor.classify_signal(title)
        cat = r["tso_category"]
        conf = r["tso_category_confidence"]
        dom = r["domain_relevance"]
        ok = "✅" if cat == expected else "❌"
        if cat == expected:
            correct += 1
        print(f"  {ok} {title[:55]:56s} → {cat:25s} "
              f"(conf={conf:.2f} dom={dom:.2f})  "
              f"task={r['amprion_task'] or '-':20s} "
              f"area={r['business_area_code'] or '-'}")
    print(f"\n{correct}/{len(tests)} correct")
