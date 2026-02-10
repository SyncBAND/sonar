"""
SONAR.AI / AGTS Trend Scorer v2
================================
Evidence-Weighted Multi-Criteria Decision Analysis (MCDA)
with Amprion Strategic Priors

Methodology (industry-standard Bayesian MCDA):
──────────────────────────────────────────────
Priority = (
    amprion_strategic_weight × 0.35   ← Fixed prior from Amprion framework
  + evidence_strength        × 0.25   ← Signal volume, diversity, quality
  + growth_momentum          × 0.20   ← Temporal signal velocity
  + maturity_readiness       × 0.20   ← Technology readiness from content
) / 10  +  project_bonus

This ensures:
  • CRITICAL categories rank higher even with few signals (prior dominates)
  • Evidence confidence differentiates within same priority tier
  • No ceiling compression (strategic_weight ≤ 95 → max raw ≈ 85)
  • Maturity COMPUTED from signal characteristics (not hardcoded)
  • First-run growth handled correctly (no inflation from zero baseline)

Changes from v1:
  • Replaced keyword-counting SR/GS/CE with Amprion strategic priors
  • Signal count now meaningfully affects evidence score
  • Maturity computed from source types + title keywords
  • Multiplicative project multiplier → additive project bonus
  • Added confidence levels and monitoring alerts
  • SHAP explanation preserved (additive decomposition)
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# AMPRION STRATEGIC PRIORS
# =============================================================================
# Source: Amprion Innovation Framework & TSO Fact Book
# These are FIXED domain-expert judgments about what matters most.
# They do NOT change based on signal volume — they are the organizational prior.

AMPRION_STRATEGIC_PRIORS: Dict[str, Dict[str, Any]] = {
    # ── EXISTENTIAL tier: Failure is unacceptable ─────────────────────────
    "grid_stability": {
        "weight": 95, "risk": "CRITICAL", "stability": 100,
        "tier": "existential",
        "rationale": "CRITICAL – Non-negotiable. Curative management reduces CAPEX.",
        "default_projects": ["STATCOM Gersteinwerk", "STATCOM Polsum", "Grid Boosters"],
    },
    "cybersecurity_ot": {
        "weight": 95, "risk": "CRITICAL", "stability": 95,
        "tier": "existential",
        "rationale": "CRITICAL – Existential. Defensive posture mandatory.",
    },

    # ── CRITICAL tier: Core TSO mission, actively planned ─────────────────
    "grid_infrastructure": {
        "weight": 90, "risk": "HIGH", "stability": 80,
        "tier": "critical",
        "rationale": "Core TSO operations. Grid expansion is primary mandate.",
        "default_projects": ["SuedLink", "Korridor B", "Rhein-Main Link"],
    },
    "offshore_systems": {
        "weight": 90, "risk": "HIGH", "stability": 70,
        "tier": "critical",
        "rationale": "CRITICAL. 70GW offshore target, direct Amprion investment.",
        "default_projects": ["BalWin1", "BalWin2", "NeuLink", "Korridor B"],
    },
    "flexibility_vpp": {
        "weight": 88, "risk": "MEDIUM", "stability": 65,
        "tier": "critical",
        "rationale": "CRITICAL short-term, TRANSFORMATIONAL long-term.",
        "default_projects": ["Systemmarkt", "Grid Boosters", "Korridor B"],
    },

    # ── HIGH tier: Strategically important, actively tracked ──────────────
    "renewables_integration": {
        "weight": 80, "risk": "MEDIUM", "stability": 60,
        "tier": "high",
        "rationale": "Core TSO mission — managing renewable feed-in.",
    },
    "regulatory_policy": {
        "weight": 80, "risk": "MEDIUM", "stability": 50,
        "tier": "high",
        "rationale": "Enables/constrains all other categories.",
    },
    "energy_storage": {
        "weight": 78, "risk": "MEDIUM", "stability": 75,
        "tier": "high",
        "rationale": "HIGH. Grid Boosters. Congestion relief.",
        "default_projects": ["Grid Boosters"],
    },
    "ai_grid_optimization": {
        "weight": 72, "risk": "MEDIUM", "stability": 55,
        "tier": "high",
        "rationale": "HIGH. Auto-Trader, probabilistic forecasting.",
        "default_projects": ["Auto-Trader"],
    },

    # ── MEDIUM-HIGH tier: Important but less immediate ────────────────────
    "hydrogen_p2g": {
        "weight": 68, "risk": "MEDIUM", "stability": 30,
        "tier": "medium-high",
        "rationale": "HIGH for Ruhr region. Sector coupling via hybridge.",
        "default_projects": ["hybridge"],
    },
    "energy_trading": {
        "weight": 65, "risk": "LOW", "stability": 25,
        "tier": "medium-high",
        "rationale": "Market operations, Systemmarkt evolution.",
    },
    "e_mobility_v2g": {
        "weight": 62, "risk": "MEDIUM", "stability": 20,
        "tier": "medium-high",
        "rationale": "HIGH. New load patterns, bidirectional flows.",
    },

    # ── MEDIUM tier: Watch and track ──────────────────────────────────────
    "distributed_generation": {
        "weight": 60, "risk": "MEDIUM", "stability": 35,
        "tier": "medium",
        "rationale": "MEDIUM-HIGH. Linked to Systemmarkt.",
        "default_projects": ["Systemmarkt"],
    },
    "digital_twin_simulation": {
        "weight": 58, "risk": "LOW", "stability": 15,
        "tier": "medium",
        "rationale": "MEDIUM-HIGH. Planning and ops tooling.",
    },
    "biogas_biomethane": {
        "weight": 55, "risk": "LOW", "stability": 10,
        "tier": "medium",
        "rationale": "HIGH for Ruhr region. Sector coupling.",
        "default_projects": ["hybridge"],
    },
    "power_generation": {
        "weight": 50, "risk": "LOW", "stability": 50,
        "tier": "medium",
        "rationale": "Informational. Not direct TSO action but affects operations.",
    },
}

# Default prior for unknown categories
_DEFAULT_PRIOR = {"weight": 30, "risk": "LOW", "stability": 10, "tier": "low"}


# =============================================================================
# MATURITY ASSESSMENT CONSTANTS
# =============================================================================

CATEGORY_MATURITY_BASELINE: Dict[str, int] = {
    "grid_infrastructure": 7, "grid_stability": 6, "offshore_systems": 6,
    "renewables_integration": 7, "energy_storage": 6, "power_generation": 8,
    "cybersecurity_ot": 5, "e_mobility_v2g": 5, "hydrogen_p2g": 4,
    "flexibility_vpp": 5, "energy_trading": 7, "distributed_generation": 6,
    "ai_grid_optimization": 4, "digital_twin_simulation": 5,
    "regulatory_policy": 7, "biogas_biomethane": 6,
}

# Title keywords that shift maturity estimate
_MATURITY_UP = [
    "deployed", "commissioned", "operational", "installed", "commercial",
    "production", "widespread", "standard", "mandatory", "established",
    "revenue", "gigawatt", "terawatt", "record", "utility-scale",
    "grid-scale", "megawatt-scale", "approved", "construction",
]
_MATURITY_MID = [
    "pilot", "trial", "testing", "validation", "demonstration",
    "prototype", "pre-commercial", "scale-up", "field test",
]
_MATURITY_DOWN = [
    "theoretical", "concept", "feasibility", "proposed", "hypothesis",
    "simulation", "modeling", "review", "survey", "novel", "framework",
    "algorithm", "method", "technique", "approach", "analysis",
]

# Source-type TRL contributions
_SOURCE_TRL: Dict[str, float] = {
    "research": 2.5, "research_org": 3.0, "patent": 4.0,
    "conference": 4.5, "news": 6.0, "tso_news": 6.5,
    "regulatory": 7.5,
}


# =============================================================================
# STRATEGIC NATURE MAPPING (from TSO perspective)
# =============================================================================

CATEGORY_STRATEGIC_NATURE: Dict[str, str] = {
    # Accelerator: strengthens current TSO operations
    "grid_infrastructure": "Accelerator",
    "grid_stability": "Accelerator",
    "offshore_systems": "Accelerator",
    "energy_storage": "Accelerator",
    "cybersecurity_ot": "Accelerator",
    "regulatory_policy": "Accelerator",
    "power_generation": "Accelerator",
    # Disruptor: challenges current TSO model
    "distributed_generation": "Disruptor",
    "e_mobility_v2g": "Disruptor",
    "flexibility_vpp": "Disruptor",
    # Transformational: fundamental paradigm shift
    "ai_grid_optimization": "Transformational",
    "hydrogen_p2g": "Transformational",
    "digital_twin_simulation": "Transformational",
    "renewables_integration": "Transformational",
    "biogas_biomethane": "Transformational",
    "energy_trading": "Transformational",
}


# =============================================================================
# MEGA-PROJECT BONUSES (additive, not multiplicative)
# =============================================================================

MEGA_PROJECT_KEYWORDS: Dict[str, float] = {
    "korridor b": 0.5, "dc31": 0.5, "dc32": 0.5,
    "rhein-main link": 0.5, "a-nord": 0.5,
    "suedlink": 0.5, "ultranet": 0.5,
    "balwin": 0.5, "neulink": 0.5,
    "offshore hub": 0.3, "statcom": 0.3,
    "grid booster": 0.3, "systemmarkt": 0.3,
    "hybridge": 0.3, "auto-trader": 0.3,
}


# =============================================================================
# MCDA WEIGHTS — derived via Analytic Hierarchy Process (AHP)
# =============================================================================
# Weights are computed from pairwise expert comparisons, not guessed.
# Configure via: ahp_config.json, AHP_PROFILE env var, or defaults.
# See services/ahp.py for full documentation.

from services.ahp import get_mcda_weights as _get_ahp_weights

MCDA_WEIGHTS = _get_ahp_weights()
# Signal-driven: {sum of evidence+growth+maturity weights},
# Prior-driven: {strategic_importance weight}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_date(signal: Dict) -> Optional[datetime]:
    """Extract publication date from signal dict."""
    for field in ("published_at", "scraped_at"):
        val = signal.get(field)
        if val is None:
            continue
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(
                    val.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass
    return None


def _days_ago(signal: Dict, now: datetime) -> Optional[float]:
    """How many days ago was this signal published?"""
    dt = _parse_date(signal)
    if dt is None:
        return None
    return max((now - dt).total_seconds() / 86400, 0)


# =============================================================================
# TREND SCORER
# =============================================================================

class TrendScorer:
    """
    Evidence-Weighted MCDA scorer with Amprion strategic priors.

    Usage::

        scorer = TrendScorer()
        result = scorer.score_trend(trend_data, signals)
        # result["priority_score"]  → 1.0–10.0
        # result["maturity_score"]  → 1–9 (TRL)
        # result["confidence"]      → "HIGH" / "MEDIUM" / "LOW"
    """

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def score_trend(
        self,
        trend_data: Dict[str, Any],
        signals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Score a trend using evidence-weighted MCDA.

        Args:
            trend_data: Trend metadata (tso_category, amprion_task, etc.)
            signals: List of signal dicts (title, content, quality_score, etc.)

        Returns:
            Dict with all scoring fields, backward-compatible with full.py.
        """
        category = trend_data.get("tso_category", "other")
        prior = AMPRION_STRATEGIC_PRIORS.get(category, _DEFAULT_PRIOR)
        linked_projects = trend_data.get("linked_projects") or []

        # ── 1. Strategic importance (fixed prior) ─────────────────────
        strategic_score = prior["weight"]

        # ── 2. Evidence strength (from signals) ──────────────────────
        evidence_score, evidence_detail = self._evidence_strength(signals)

        # ── 3. Growth momentum (temporal patterns) ───────────────────
        growth_score, growth_rate = self._growth_momentum(signals)

        # ── 4. Maturity readiness (content analysis) ─────────────────
        maturity_trl = self._compute_maturity(category, signals)
        maturity_score_pct = (maturity_trl / 9) * 100

        # ── 5. Combine via MCDA weights ──────────────────────────────
        raw = (
            strategic_score    * MCDA_WEIGHTS["strategic_importance"]
            + evidence_score   * MCDA_WEIGHTS["evidence_strength"]
            + growth_score     * MCDA_WEIGHTS["growth_momentum"]
            + maturity_score_pct * MCDA_WEIGHTS["maturity_readiness"]
        )
        priority_base = raw / 10.0  # → 0–10 scale

        # ── 6. Project bonus (additive) ──────────────────────────────
        project_bonus, project_type = self._project_bonus(
            signals, linked_projects, category
        )
        priority = min(max(priority_base + project_bonus, 1.0), 10.0)

        # ── 7. Derive classifications ────────────────────────────────
        lifecycle = self._lifecycle_status(maturity_trl)
        time_to_impact = self._time_to_impact(maturity_trl, growth_rate)
        strategic_nature = CATEGORY_STRATEGIC_NATURE.get(category, "Accelerator")
        confidence = self._confidence_level(evidence_score)

        # ── 8. Legacy compatibility scores ───────────────────────────
        stability_score = prior.get("stability", 30)
        cost_efficiency = self._cost_efficiency_proxy(maturity_trl, category)

        signal_count = len(signals)
        volume_score = self._volume_score(signal_count)
        quality_score = self._avg_quality(signals)

        # ── 9. Monitoring alerts ─────────────────────────────────────
        # Blind spot = strategically important but insufficient coverage
        _tier = prior.get("tier", "low")
        if _tier in ("existential", "critical"):
            # Alert when < 15 signals OR evidence < 50
            monitoring_alert = (signal_count < 15 or evidence_score < 50)
        elif _tier == "high":
            monitoring_alert = (signal_count < 10 and evidence_score < 40)
        else:
            monitoring_alert = False
        action = self._recommend_action(prior, evidence_score, growth_score)

        # ── 10. Build SHAP-lite explanation ──────────────────────────
        explanation = self._build_explanation(
            category, prior, strategic_score, evidence_score,
            growth_score, maturity_score_pct, priority_base,
            project_bonus, priority, signal_count, growth_rate,
            evidence_detail, maturity_trl, monitoring_alert,
            linked_projects, trend_data,
        )

        return {
            # ── Primary scores ──
            "priority_score": round(priority, 1),
            "overall_score": round(raw, 1),

            # ── Component scores (0–100, backward compat) ──
            "strategic_relevance_score": round(strategic_score, 1),
            "grid_stability_score": round(stability_score, 1),
            "cost_efficiency_score": round(cost_efficiency, 1),
            "volume_score": round(volume_score, 1),
            "growth_score": round(growth_score, 1),
            "quality_score": round(quality_score, 1),

            # ── NEW: evidence-based components ──
            "evidence_strength_score": round(evidence_score, 1),
            "maturity_readiness_score": round(maturity_score_pct, 1),

            # ── Maturity (COMPUTED, not hardcoded) ──
            "maturity_score": maturity_trl,
            "maturity_type": "TRL",
            "lifecycle_status": lifecycle,

            # ── Classification ──
            "time_to_impact": time_to_impact,
            "strategic_nature": strategic_nature,
            "confidence": confidence,

            # ── Actionability ──
            "monitoring_alert": monitoring_alert,
            "recommended_action": action,
            "amprion_tier": prior.get("tier", "low"),

            # ── Project linkage ──
            "project_multiplier": round(1.0 + project_bonus / 5.0, 2),
            "project_bonus": round(project_bonus, 2),
            "project_type": project_type,

            # ── Statistics ──
            "signal_count": signal_count,
            "growth_rate": round(growth_rate, 1),
            "source_diversity": evidence_detail.get("unique_sources", 0),

            # ── Explanation ──
            "shapley_explanation": explanation,
        }

    # --------------------------------------------------------------------- #
    # COMPONENT CALCULATIONS
    # --------------------------------------------------------------------- #

    def _evidence_strength(
        self, signals: List[Dict]
    ) -> Tuple[float, Dict]:
        """
        Compute evidence strength from signal characteristics.

        Components:
          • Volume (40%): logarithmic, saturates ~200 signals
          • Source diversity (25%): unique source names + types
          • Quality (20%): average signal quality score
          • Recency (15%): fraction published in last 7 days
        """
        n = len(signals)
        if n == 0:
            return 0.0, {"volume": 0, "diversity": 0, "quality": 0,
                         "recency": 0, "unique_sources": 0, "unique_types": 0}

        # Volume: log scale, 1→0, 10→43, 100→87, 200→100
        volume = min(math.log(n + 1) / math.log(201) * 100, 100)

        # Source diversity
        source_names = set(s.get("source_name") or "unknown" for s in signals)
        source_types = set(s.get("source_type") or "unknown" for s in signals)
        n_names = len(source_names)
        n_types = len(source_types)
        diversity = min((n_names / 8) * 60 + (n_types / 4) * 40, 100)

        # Quality
        quals = [s.get("quality_score", 0.5) for s in signals]
        avg_quality = (sum(quals) / len(quals)) * 100

        # Recency
        now = datetime.utcnow()
        recent_7d = sum(
            1 for s in signals if (_days_ago(s, now) or 999) <= 7
        )
        recency = (recent_7d / n) * 100

        score = (
            volume       * 0.40
            + diversity  * 0.25
            + avg_quality * 0.20
            + recency    * 0.15
        )

        detail = {
            "volume": round(volume, 1),
            "diversity": round(diversity, 1),
            "quality": round(avg_quality, 1),
            "recency": round(recency, 1),
            "unique_sources": n_names,
            "unique_types": n_types,
        }
        return min(score, 100), detail

    def _growth_momentum(
        self, signals: List[Dict]
    ) -> Tuple[float, float]:
        """
        Measure growth velocity using 14-day comparison windows.

        FIRST RUN / COLD START HANDLING:
        ─────────────────────────────────────────
        On first run, ALL signals are scraped within minutes/hours.
        RSS feeds contain articles from different dates, but this is
        NOT real growth - it's just the backlog from first scrape.

        We detect cold-start by checking if all signals were scraped
        within a 24-hour window. If so, growth_rate = 0 (unknown).
        """
        now = datetime.utcnow()

        recent = 0      # published 0–14 days ago
        previous = 0    # published 14–28 days ago
        older = 0       # published 28+ days ago
        undated = 0
        scrape_dates = []

        for s in signals:
            days = _days_ago(s, now)
            if days is None:
                undated += 1
            elif days <= 14:
                recent += 1
            elif days <= 28:
                previous += 1
            else:
                older += 1

            # Collect scrape timestamps for cold-start detection
            sd = s.get("scraped_at")
            if isinstance(sd, datetime):
                scrape_dates.append(sd)
            elif isinstance(sd, str):
                try:
                    scrape_dates.append(
                        datetime.fromisoformat(sd.replace("Z", "+00:00"))
                        .replace(tzinfo=None)
                    )
                except (ValueError, TypeError):
                    pass

        total_dated = recent + previous + older

        # ── STRICT Cold-start detection ──────────────────────────────
        # If all signals were scraped within 24 hours, this is FIRST RUN
        is_first_run = False
        if len(scrape_dates) >= 2:
            scrape_span = (max(scrape_dates) - min(scrape_dates)).total_seconds()
            if scrape_span < 24 * 3600:  # < 24 hours = first run
                is_first_run = True
        elif len(scrape_dates) <= 1:
            is_first_run = True  # Single scrape or no scrape dates

        # FIRST RUN: Return neutral score with 0% growth rate
        if is_first_run:
            # Return neutral growth score (50) and 0% rate
            # This means growth contributes nothing to ranking on first run
            return 50.0, 0.0

        # ── Normal multi-day operation ───────────────────────────────
        if total_dated == 0:
            return 50.0, 0.0

        # No baseline at all (shouldn't happen after first run)
        if previous == 0 and older == 0:
            return 50.0, 0.0

        baseline = previous if previous > 0 else max(older / 4, 1)

        # Baseline too thin - need at least 3 signals in baseline
        if baseline < 3:
            if baseline > 0:
                raw_rate = ((recent - baseline) / baseline) * 100
            else:
                raw_rate = 0.0
            # Cap at ±50% for thin baselines
            capped_rate = max(min(raw_rate, 50), -50)
            growth_score = min(max(50 + capped_rate / 4, 30), 65)
            return growth_score, round(raw_rate, 1)

        # ── Normal growth calculation (sufficient history) ────────
        growth_rate = ((recent - baseline) / baseline) * 100

        # Convert: -100%→10, 0%→50, +200%→90, capped at 95
        growth_score = min(max(50 + growth_rate / 4, 5), 95)

        return growth_score, round(growth_rate, 1)

    def _compute_maturity(
        self, category: str, signals: List[Dict]
    ) -> int:
        """
        Compute Technology Readiness Level (TRL 1–9) from signal content.

        1. Start with category baseline TRL
        2. Adjust based on signal source types (±2)
        3. Adjust based on title keyword analysis (±1)
        """
        baseline = CATEGORY_MATURITY_BASELINE.get(category, 5)

        if not signals:
            return baseline

        # Source-type adjustment
        source_trls = []
        for s in signals:
            src = s.get("source_type", "news")
            trl = _SOURCE_TRL.get(src, 5.5)
            source_trls.append(trl)

        if source_trls:
            avg_source_trl = sum(source_trls) / len(source_trls)
            source_adj = (avg_source_trl - 5.0) * 0.4
        else:
            source_adj = 0

        # Keyword adjustment
        up_count = 0
        mid_count = 0
        down_count = 0

        for s in signals:
            title = (s.get("title") or "").lower()
            up_count += sum(1 for kw in _MATURITY_UP if kw in title)
            mid_count += sum(1 for kw in _MATURITY_MID if kw in title)
            down_count += sum(1 for kw in _MATURITY_DOWN if kw in title)

        total_kw = up_count + mid_count + down_count
        if total_kw > 0:
            kw_adj = (up_count - down_count) / total_kw
        else:
            kw_adj = 0

        trl = baseline + source_adj + kw_adj
        return min(max(round(trl), 1), 9)

    def _project_bonus(
        self,
        signals: List[Dict],
        linked_projects: List[str],
        category: str,
    ) -> Tuple[float, str]:
        """Additive project bonus based on mega-project linkage."""
        # Explicit linked_projects from trend metadata
        if linked_projects:
            return 0.5, "Strategic Mega-Project"

        # Default projects from Amprion priors
        prior = AMPRION_STRATEGIC_PRIORS.get(category, {})
        if prior.get("default_projects"):
            return 0.3, "Strategic"

        # Signal text scan for mega-project mentions
        combined = " ".join(
            (s.get("title") or "") for s in signals[:50]
        ).lower()
        for keyword, bonus in MEGA_PROJECT_KEYWORDS.items():
            if keyword in combined:
                return bonus, "Mega-Project Linked"

        # Category-based tactical bonus
        if category in ("grid_infrastructure", "grid_stability",
                        "offshore_systems", "cybersecurity_ot"):
            return 0.15, "Tactical"

        return 0.0, "Operational"

    # --------------------------------------------------------------------- #
    # DERIVED CLASSIFICATIONS
    # --------------------------------------------------------------------- #

    @staticmethod
    def _lifecycle_status(trl: int) -> str:
        if trl <= 3:
            return "Scouting"
        elif trl <= 5:
            return "Pilot"
        elif trl <= 7:
            return "Implementation"
        else:
            return "Standard"

    @staticmethod
    def _time_to_impact(trl: int, growth_rate: float) -> str:
        fast = growth_rate > 50
        if trl >= 8 or (trl >= 7 and fast):
            return "<1 year"
        elif trl >= 6 or (trl >= 5 and fast):
            return "1-3 years"
        elif trl >= 4:
            return "3-5 years"
        else:
            return "5+ years"

    @staticmethod
    def _confidence_level(evidence_score: float) -> str:
        if evidence_score >= 60:
            return "HIGH"
        elif evidence_score >= 35:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def _recommend_action(
        prior: Dict, evidence: float, growth: float
    ) -> str:
        weight = prior["weight"]
        if weight >= 85 and evidence >= 50:
            return "Act"
        elif weight >= 75 or (weight >= 60 and evidence >= 60):
            return "Evaluate"
        elif growth >= 70 or evidence >= 70:
            return "Monitor"
        else:
            return "Track"

    # --------------------------------------------------------------------- #
    # COMPATIBILITY HELPERS
    # --------------------------------------------------------------------- #

    @staticmethod
    def _volume_score(n: int) -> float:
        return min(math.log10(max(n, 1) + 1) * 40, 100)

    @staticmethod
    def _avg_quality(signals: List[Dict]) -> float:
        if not signals:
            return 50.0
        quals = [s.get("quality_score", 0.5) * 100 for s in signals]
        return sum(quals) / len(quals)

    @staticmethod
    def _cost_efficiency_proxy(trl: int, category: str) -> float:
        base = trl * 10
        if category in ("ai_grid_optimization", "digital_twin_simulation",
                        "cybersecurity_ot"):
            base += 10
        elif category in ("grid_infrastructure", "offshore_systems",
                          "hydrogen_p2g"):
            base -= 5
        return min(max(base, 10), 100)

    # --------------------------------------------------------------------- #
    # SHAP-LITE EXPLANATION
    # --------------------------------------------------------------------- #

    def _build_explanation(
        self, category, prior, strategic_score, evidence_score,
        growth_score, maturity_pct, priority_base, project_bonus,
        final_priority, signal_count, growth_rate, evidence_detail,
        maturity_trl, monitoring_alert, linked_projects, trend_data,
    ) -> Dict[str, Any]:
        """Build additive SHAP-style explanation."""

        BASE = 5.0

        sr_c = (strategic_score - 50) * MCDA_WEIGHTS["strategic_importance"] / 10
        ev_c = (evidence_score - 50) * MCDA_WEIGHTS["evidence_strength"] / 10
        gr_c = (growth_score - 50) * MCDA_WEIGHTS["growth_momentum"] / 10
        mt_c = (maturity_pct - 50) * MCDA_WEIGHTS["maturity_readiness"] / 10
        pj_c = project_bonus

        abs_vals = {
            "strategic_importance": abs(sr_c),
            "evidence_strength": abs(ev_c),
            "growth_momentum": abs(gr_c),
            "maturity_readiness": abs(mt_c),
            "project_bonus": abs(pj_c),
        }
        total_abs = sum(abs_vals.values()) or 1
        importance = {k: round(v / total_abs * 100, 1)
                      for k, v in abs_vals.items()}

        # Human-readable drivers
        drivers = []
        tier = prior.get("tier", "low")

        if tier in ("existential", "critical"):
            drivers.append(
                f"Amprion tier: {tier.upper()} (strategic weight {strategic_score})"
            )
        elif strategic_score >= 70:
            drivers.append(f"High strategic relevance (weight {strategic_score})")

        if monitoring_alert:
            drivers.append(
                f"⚠ BLIND SPOT: {tier.upper()} category with only "
                f"{signal_count} signals — increase monitoring"
            )

        if evidence_score >= 60:
            drivers.append(
                f"Strong evidence ({signal_count} signals, "
                f"{evidence_detail['unique_sources']} sources)"
            )
        elif evidence_score < 30:
            drivers.append(
                f"Weak evidence ({signal_count} signals) — needs more sources"
            )

        if growth_rate > 50:
            drivers.append(f"Accelerating (+{growth_rate:.0f}% growth)")
        elif growth_rate < -30:
            drivers.append(f"Decelerating ({growth_rate:.0f}% growth)")

        if project_bonus >= 0.3:
            projects = linked_projects or prior.get("default_projects", [])
            if projects:
                drivers.append(
                    f"Mega-project linkage: {', '.join(projects[:3])} "
                    f"(+{project_bonus:.1f})"
                )

        if maturity_trl >= 7:
            drivers.append(
                f"Mature technology (TRL {maturity_trl}) — ready for deployment"
            )
        elif maturity_trl <= 3:
            drivers.append(
                f"Early stage (TRL {maturity_trl}) — scouting/research phase"
            )

        waterfall = [
            {"feature": "base_value", "value": BASE,
             "cumulative": round(BASE, 2)},
            {"feature": "strategic_importance", "value": round(sr_c, 3),
             "cumulative": round(BASE + sr_c, 2)},
            {"feature": "evidence_strength", "value": round(ev_c, 3),
             "cumulative": round(BASE + sr_c + ev_c, 2)},
            {"feature": "growth_momentum", "value": round(gr_c, 3),
             "cumulative": round(BASE + sr_c + ev_c + gr_c, 2)},
            {"feature": "maturity_readiness", "value": round(mt_c, 3),
             "cumulative": round(
                 BASE + sr_c + ev_c + gr_c + mt_c, 2)},
            {"feature": "project_bonus", "value": round(pj_c, 3),
             "cumulative": round(final_priority, 2)},
        ]

        return {
            "method": "bayesian_mcda_decomposition",
            "base_value": BASE,
            "feature_importance": importance,
            "waterfall": waterfall,
            "top_drivers": drivers,
            "amprion_context": {
                "category": category,
                "tier": tier,
                "risk_level": prior.get("risk", "LOW"),
                "amprion_task": trend_data.get("amprion_task"),
                "linked_projects": (
                    linked_projects or prior.get("default_projects", [])
                ),
            },
            "alerts": {
                "monitoring_alert": monitoring_alert,
                "confidence": self._confidence_level(evidence_score),
                "recommended_action": self._recommend_action(
                    prior, evidence_score, growth_score
                ),
            },
        }


# =============================================================================
# SINGLETON
# =============================================================================

_trend_scorer: Optional[TrendScorer] = None


def get_trend_scorer() -> TrendScorer:
    """Get singleton trend scorer instance."""
    global _trend_scorer
    if _trend_scorer is None:
        _trend_scorer = TrendScorer()
    return _trend_scorer


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    scorer = get_trend_scorer()

    test_cases = [
        ("grid_stability",         6,   "system_security"),
        ("cybersecurity_ot",       169, "system_security"),
        ("grid_infrastructure",    17,  "grid_expansion"),
        ("offshore_systems",       43,  "grid_expansion"),
        ("flexibility_vpp",        10,  "european_trading"),
        ("renewables_integration", 84,  "decarbonization"),
        ("energy_storage",         73,  "system_security"),
        ("e_mobility_v2g",         97,  "decarbonization"),
        ("ai_grid_optimization",   133, "system_security"),
        ("hydrogen_p2g",           64,  "decarbonization"),
        ("regulatory_policy",      85,  "european_trading"),
        ("energy_trading",         57,  "european_trading"),
        ("distributed_generation", 14,  "decarbonization"),
        ("digital_twin_simulation", 24, "system_security"),
        ("biogas_biomethane",      9,   "decarbonization"),
        ("power_generation",       28,  "system_security"),
    ]

    print("=" * 100)
    print("SONAR.AI Trend Scorer v2 — Simulation")
    print("=" * 100)
    print(f"\n{'#':<3} {'Category':<30} {'Sig':>4} {'Score':>6} "
          f"{'TRL':>4} {'Lifecycle':<16} {'Conf':<8} {'Action':<10} {'Tier':<12}")
    print("-" * 100)

    results = []
    for cat, count, task in test_cases:
        signals = [
            {
                "title": f"Signal about {cat.replace('_', ' ')}",
                "quality_score": 0.65,
                "source_type": "news",
                "source_name": f"Source_{i % 8}",
                "published_at": (
                    datetime.utcnow() - timedelta(days=i % 30)
                ).isoformat(),
            }
            for i in range(count)
        ]
        for i in range(min(count // 4, 20)):
            signals.append({
                "title": f"Research on {cat.replace('_', ' ')} algorithm",
                "quality_score": 0.78,
                "source_type": "research",
                "source_name": "arXiv",
                "published_at": (
                    datetime.utcnow() - timedelta(days=i % 14)
                ).isoformat(),
            })

        trend_data = {
            "tso_category": cat,
            "amprion_task": task,
            "linked_projects": [],
            "strategic_domain": "system_operation",
        }
        result = scorer.score_trend(trend_data, signals)
        results.append((cat, count, result))

    results.sort(key=lambda x: x[2]["priority_score"], reverse=True)

    for rank, (cat, count, r) in enumerate(results, 1):
        alert = " ⚠" if r["monitoring_alert"] else ""
        print(
            f"{rank:<3} {cat:<30} {count:>4} {r['priority_score']:>6.1f} "
            f"{r['maturity_score']:>4} {r['lifecycle_status']:<16} "
            f"{r['confidence']:<8} {r['recommended_action']:<10} "
            f"{r['amprion_tier']:<12}{alert}"
        )

    scores = [r["priority_score"] for _, _, r in results]
    print(f"\nScore range: {min(scores):.1f} – {max(scores):.1f}")
    print(f"Unique scores: {len(set(scores))} / {len(results)}")
