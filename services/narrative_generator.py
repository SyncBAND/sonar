"""
SONAR.AI Narrative Generator
=============================
Populates the "Brief" and "Deep Dive" fields for each trend.

Two modes:
  1. TEMPLATE (default) — Uses structured data already in the pipeline.
     No external API calls. Works immediately.

  2. LLM (upgrade) — Feeds signal content to an LLM for real summarization.
     Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.

Usage:
    from services.narrative_generator import NarrativeGenerator

    gen = NarrativeGenerator(mode="template")  # or mode="llm"
    brief = gen.generate_brief(trend, signals, scores)
    deep  = gen.generate_deep_dive(trend, signals, scores)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# AMPRION CONTEXT — used to ground narratives in company specifics
# ─────────────────────────────────────────────────────────────────────────────

AMPRION_CONTEXT = {
    "grid_stability": {
        "why_amprion": "Amprion operates 11,000 km of extra-high-voltage transmission and is directly responsible for frequency stability in its control area",
        "risk_scenario": "Loss of frequency control or cascading outages during low-inertia periods (high wind/solar, low conventional generation)",
        "regions": ["Ruhr region", "North Rhine-Westphalia", "Rhineland-Palatinate"],
    },
    "cybersecurity_ot": {
        "why_amprion": "As designated critical infrastructure (KRITIS), Amprion faces mandatory IT security under NIS2 and IT-SiKat. OT networks controlling 380kV substations are high-value targets",
        "risk_scenario": "Compromise of SCADA/EMS systems or supply-chain attack on substation control firmware",
        "regions": ["All Amprion control area"],
    },
    "grid_infrastructure": {
        "why_amprion": "Amprion's primary mandate is grid expansion under the Federal Grid Development Plan (NEP). Multi-billion-euro HVDC corridors are in planning/construction",
        "risk_scenario": "Permitting delays or technology supply bottlenecks slow Korridor B, A-Nord, or Ultranet delivery timelines",
        "regions": ["North-South corridors", "Ruhr industrial area", "Rhine-Main region"],
    },
    "offshore_systems": {
        "why_amprion": "Amprion is building offshore HVDC connections (DolWin, BorWin, BalWin) to connect North Sea wind farms to the onshore grid",
        "risk_scenario": "Cable faults, converter station delays, or insufficient onshore grid capacity to absorb offshore generation",
        "regions": ["North Sea", "German Bight", "Lower Saxony landing points"],
    },
    "flexibility_vpp": {
        "why_amprion": "With rising renewable penetration, Amprion needs aggregated flexibility (VPPs, demand response) to replace conventional balancing capacity",
        "risk_scenario": "Insufficient controllable flexibility during Dunkelflaute (low wind + solar) periods",
        "regions": ["Ruhr industrial loads", "residential heat pumps in NRW"],
    },
    "renewables_integration": {
        "why_amprion": "Amprion's control area sees massive solar + wind feed-in. Managing curtailment and redispatch is a daily operational challenge",
        "risk_scenario": "Grid congestion from north-south renewable flows exceeding corridor capacity",
        "regions": ["Schleswig-Holstein wind corridor", "NRW solar growth"],
    },
    "energy_storage": {
        "why_amprion": "Grid-scale batteries and Grid Boosters at congestion points can defer costly network reinforcement",
        "risk_scenario": "Without storage, Amprion must rely on expensive redispatch and curtailment",
        "regions": ["Congestion nodes on north-south corridors"],
    },
    "ai_grid_optimization": {
        "why_amprion": "Amprion's Auto-Trader and probabilistic forecasting initiatives aim to automate market and grid operations",
        "risk_scenario": "AI-driven trading errors or forecast failures during extreme weather events",
        "regions": ["Amprion control center Brauweiler"],
    },
    "e_mobility_v2g": {
        "why_amprion": "Mass EV adoption creates new load patterns (evening charging peaks) and potential V2G flexibility resources",
        "risk_scenario": "Unmanaged EV charging creates distribution-level peaks that cascade to transmission level",
        "regions": ["Urban NRW", "Autobahn corridor charging hubs"],
    },
    "hydrogen_p2g": {
        "why_amprion": "Amprion's hybridge project explores power-to-gas coupling. Hydrogen demand from Ruhr industry could reshape load profiles",
        "risk_scenario": "Electrolyser ramp-up creates large, variable loads at transmission level",
        "regions": ["Ruhr industrial belt", "hybridge corridor Lingen-Wietze"],
    },
    "energy_trading": {
        "why_amprion": "Amprion participates in European balancing markets and is evolving toward automated trading (Systemmarkt)",
        "risk_scenario": "Market design changes (e.g., nodal pricing) fundamentally alter Amprion's revenue model",
        "regions": ["European interconnectors", "Core bidding zone"],
    },
    "regulatory_policy": {
        "why_amprion": "BNetzA regulation and EU Clean Energy Package directly shape Amprion's investment recovery and operational latitude",
        "risk_scenario": "Regulatory lag prevents timely cost recovery for grid expansion investments",
        "regions": ["German regulatory framework", "EU-level ACER decisions"],
    },
    "digital_twin_simulation": {
        "why_amprion": "Digital twins of grid corridors enable construction risk simulation and predictive maintenance planning",
        "risk_scenario": "Without simulation capability, Amprion cannot efficiently plan multi-decade asset lifecycles",
        "regions": ["Korridor B route", "existing 380kV corridor assets"],
    },
    "distributed_generation": {
        "why_amprion": "Distributed generation changes power flow patterns and requires TSO-DSO coordination for visibility",
        "risk_scenario": "Loss of observability as generation shifts from large central plants to millions of small units",
        "regions": ["Residential solar in NRW", "industrial CHP in Ruhr"],
    },
    "biogas_biomethane": {
        "why_amprion": "Biogas provides dispatchable renewable generation, relevant for sector coupling via existing gas infrastructure",
        "risk_scenario": "Limited — biogas is peripheral to Amprion's core transmission mandate",
        "regions": ["Agricultural regions in Amprion control area"],
    },
    "power_generation": {
        "why_amprion": "The generation mix evolution (coal exit, nuclear phase-out, gas transition) directly affects system inertia and balancing needs",
        "risk_scenario": "Accelerated conventional plant retirements reduce available inertia before grid-forming inverters are deployed at scale",
        "regions": ["Ruhr coal plants", "Rhineland lignite district"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# GROWTH DESCRIPTORS
# ─────────────────────────────────────────────────────────────────────────────

def _growth_phrase(rate: float) -> str:
    if rate > 200: return "explosive growth"
    if rate > 100: return "rapid acceleration"
    if rate > 50:  return "strong growth"
    if rate > 20:  return "steady growth"
    if rate > 0:   return "moderate activity"
    if rate == 0:  return "stable activity"
    if rate > -20: return "slight decline"
    return "declining activity"


def _tier_phrase(tier: str) -> str:
    return {
        "existential": "an existential priority",
        "critical": "a critical strategic priority",
        "high": "a high strategic priority",
        "medium-high": "a medium-high priority",
        "medium": "a monitored category",
    }.get(tier, "a tracked category")


def _nature_phrase(nature: str) -> str:
    return {
        "Accelerator": "enhances existing operations",
        "Transformational": "could fundamentally reshape the business model",
        "Disruptor": "poses a potential disruption to the status quo",
    }.get(nature, "has strategic relevance")


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY AGGREGATION — collect key players from signal data
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_entities(signals: List[Dict]) -> Dict[str, List[str]]:
    """Aggregate entities across all signals for a trend."""
    orgs = Counter()
    locations = Counter()
    technologies = Counter()
    regulations = Counter()

    for sig in signals:
        entities = sig.get("entities") or {}
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except (json.JSONDecodeError, TypeError):
                entities = {}

        for org in entities.get("organizations", []):
            if len(org) > 2 and org.lower() not in ("the", "and", "for"):
                orgs[org] += 1
        for loc in entities.get("locations", []):
            locations[loc] += 1
        for tech in entities.get("technologies", []):
            technologies[tech] += 1
        for reg in entities.get("regulations", []):
            regulations[reg] += 1

    return {
        "organizations": [o for o, _ in orgs.most_common(8)],
        "locations": [l for l, _ in locations.most_common(5)],
        "technologies": [t for t, _ in technologies.most_common(6)],
        "regulations": [r for r, _ in regulations.most_common(4)],
    }


def _aggregate_keywords(signals: List[Dict], top_n: int = 10) -> List[str]:
    """Aggregate keywords across signals."""
    kw_counts = Counter()
    for sig in signals:
        kws = sig.get("keywords") or []
        if isinstance(kws, str):
            try:
                kws = json.loads(kws)
            except (json.JSONDecodeError, TypeError):
                kws = []
        for kw in kws:
            kw_counts[kw] += 1
    return [kw for kw, _ in kw_counts.most_common(top_n)]


# ═════════════════════════════════════════════════════════════════════════════
# TEMPLATE-BASED GENERATION (works now, no dependencies)
# ═════════════════════════════════════════════════════════════════════════════

class TemplateNarrative:
    """
    Generates Brief and Deep Dive from structured data.
    Quality: ~70% of what an LLM would produce.
    Strengths: Deterministic, fast, no API cost, no hallucination risk.
    Weakness: Can't summarize actual article content or identify novel insights.
    """

    def generate_brief(
        self,
        category: str,
        trend_name: str,
        scores: Dict[str, Any],
        signal_count: int,
        growth_rate: float,
        tier: str,
        strategic_impact: str,
        projects: List[str],
        nature: str,
        time_to_impact: str,
    ) -> str:
        """
        Generate 250-char Executive Summary.
        Structure: What it is + Why now + Amprion relevance
        """
        growth = _growth_phrase(growth_rate)
        tier_p = _tier_phrase(tier)

        # Build the brief
        parts = []

        # What + why now
        if strategic_impact:
            parts.append(f"{strategic_impact}.")
        else:
            parts.append(f"{trend_name} is {tier_p} for Amprion.")

        # Evidence
        parts.append(f"{signal_count} signals show {growth} ({growth_rate:+.0f}%).")

        # Time horizon
        if time_to_impact == "<1 year":
            parts.append("Near-term operational impact expected.")
        elif time_to_impact == "1-3 years":
            parts.append("Medium-term impact within 1-3 years.")
        elif time_to_impact == "3-5 years":
            parts.append("Emerging — strategic horizon 3-5 years.")
        else:
            parts.append("Long-term trend, 5+ year horizon.")

        brief = " ".join(parts)

        # Trim to 250 chars if needed
        if len(brief) > 250:
            brief = brief[:247] + "..."

        return brief

    def generate_deep_dive(
        self,
        category: str,
        trend_name: str,
        scores: Dict[str, Any],
        signal_count: int,
        growth_rate: float,
        tier: str,
        weight: int,
        rationale: str,
        strategic_impact: str,
        projects: List[str],
        nature: str,
        time_to_impact: str,
        signals: List[Dict],
        maturity_score: int = 5,
    ) -> str:
        """
        Generate up to 2000-char Contextual Analysis.
        Structure: Context → Drivers → Key Players → Amprion Implications → Score Rationale
        """
        context = AMPRION_CONTEXT.get(category, {})
        entities = _aggregate_entities(signals)
        keywords = _aggregate_keywords(signals)
        nature_p = _nature_phrase(nature)
        growth_p = _growth_phrase(growth_rate)

        sections = []

        # ── 1. Strategic Context
        why = context.get("why_amprion", f"{trend_name} is relevant to Amprion's operations.")
        sections.append(
            f"STRATEGIC CONTEXT: {why}. "
            f"Classified as {tier} tier (weight {weight}/100), this trend {nature_p}."
        )

        # ── 2. Current Evidence
        if rationale:
            sections.append(f"ASSESSMENT: {rationale}")

        sections.append(
            f"EVIDENCE: {signal_count} signals detected across monitored sources, "
            f"showing {growth_p} ({growth_rate:+.0f}%). "
            f"Technology maturity estimated at TRL {maturity_score}. "
            f"Time-to-impact: {time_to_impact}."
        )

        # ── 3. Key Players & Technologies
        players_parts = []
        if entities["organizations"]:
            players_parts.append(f"Key organizations: {', '.join(entities['organizations'][:5])}")
        if entities["technologies"]:
            players_parts.append(f"Technologies: {', '.join(entities['technologies'][:4])}")
        if entities["regulations"]:
            players_parts.append(f"Regulatory drivers: {', '.join(entities['regulations'][:3])}")
        if players_parts:
            sections.append("KEY PLAYERS: " + ". ".join(players_parts) + ".")

        # ── 4. Amprion-Specific Implications
        risk = context.get("risk_scenario")
        regions = context.get("regions", [])
        if risk:
            impl = f"AMPRION IMPLICATIONS: Primary risk scenario — {risk}."
            if regions:
                impl += f" Geographic focus: {', '.join(regions)}."
            if projects:
                impl += f" Linked projects: {', '.join(projects)}."
            sections.append(impl)

        # ── 5. Score Rationale (mini SHAP)
        score = scores.get("priority_score", 0)
        sr = scores.get("strategic_contribution", 0)
        ev = scores.get("evidence_contribution", 0)
        gr = scores.get("growth_contribution", 0)
        mt = scores.get("maturity_contribution", 0)
        pj = scores.get("project_contribution", 0)

        if score > 0:
            sections.append(
                f"SCORE BREAKDOWN: {score:.1f}/10.0 "
                f"(strategic {sr:+.2f}, evidence {ev:+.2f}, growth {gr:+.2f}, "
                f"maturity {mt:+.2f}, project {pj:+.2f})."
            )

        deep_dive = "\n\n".join(sections)

        # Trim to 2000 chars
        if len(deep_dive) > 2000:
            deep_dive = deep_dive[:1997] + "..."

        return deep_dive

    def generate_so_what(
        self,
        category: str,
        trend_name: str,
        tier: str,
        nature: str,
        growth_rate: float,
        time_to_impact: str,
        projects: List[str],
    ) -> str:
        """
        Generate 2-sentence 'So What?' summary for executives.
        """
        context = AMPRION_CONTEXT.get(category, {})
        risk = context.get("risk_scenario", "operational impact")

        if nature == "Disruptor":
            opener = f"{trend_name} poses a disruptive challenge"
        elif nature == "Transformational":
            opener = f"{trend_name} signals a fundamental shift"
        else:
            opener = f"{trend_name} is accelerating"

        if time_to_impact == "<1 year":
            urgency = "requires immediate operational attention"
        elif time_to_impact == "1-3 years":
            urgency = "should be on the 2026-2027 planning agenda"
        else:
            urgency = "warrants strategic monitoring and early positioning"

        s1 = f"{opener} with {_growth_phrase(growth_rate)} — this {urgency}."

        if projects:
            s2 = f"Direct impact on {', '.join(projects[:2])}. Risk scenario: {risk}."
        else:
            s2 = f"Risk scenario: {risk}."

        return f"{s1} {s2}"


# ═════════════════════════════════════════════════════════════════════════════
# LLM-BASED GENERATION (upgrade path)
# ═════════════════════════════════════════════════════════════════════════════

class LLMNarrative:
    """
    Generates Brief and Deep Dive using an LLM (Claude or GPT).
    Quality: ~90%+ — reads actual signal content, identifies novel insights.
    Requires: ANTHROPIC_API_KEY in environment.

    Preserves explainability by:
    - Grounding prompts in Amprion context
    - Requiring source citations
    - Storing reasoning alongside output
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found. LLM narrative generation disabled.")

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Call Anthropic API. Returns text response or None on failure."""
        if not self.api_key:
            return None
        try:
            import httpx
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def generate_brief(
        self,
        category: str,
        trend_name: str,
        scores: Dict[str, Any],
        signal_count: int,
        growth_rate: float,
        tier: str,
        signals: List[Dict],
    ) -> Optional[str]:
        """Generate 250-char Brief using LLM reading actual signal content."""
        context = AMPRION_CONTEXT.get(category, {})

        # Build signal digest — top 10 signal titles
        titles = [s.get("title", "") for s in signals[:15] if s.get("title")]
        title_block = "\n".join(f"- {t}" for t in titles)

        system = (
            "You are a strategic analyst at Amprion, a German TSO operating 11,000 km of "
            "extra-high-voltage transmission. Write concise, executive-level summaries. "
            "No jargon-free marketing language — use precise energy industry terminology."
        )

        prompt = f"""Write a 250-character maximum executive summary ("Brief") for this trend.

TREND: {trend_name}
CATEGORY: {category} ({tier} tier, priority score {scores.get('priority_score', 0):.1f}/10)
AMPRION CONTEXT: {context.get('why_amprion', 'Relevant to Amprion operations')}
SIGNAL COUNT: {signal_count} ({growth_rate:+.0f}% growth)

RECENT SIGNAL TITLES:
{title_block}

Requirements:
- Maximum 250 characters
- Explain WHAT the trend is and WHY it matters to Amprion NOW
- Reference specific evidence from the signal titles
- Write as a single paragraph, no headers"""

        return self._call_llm(system, prompt, max_tokens=200)

    def generate_deep_dive(
        self,
        category: str,
        trend_name: str,
        scores: Dict[str, Any],
        signal_count: int,
        growth_rate: float,
        tier: str,
        weight: int,
        projects: List[str],
        nature: str,
        time_to_impact: str,
        signals: List[Dict],
        maturity_score: int = 5,
    ) -> Optional[str]:
        """Generate 2000-char Deep Dive using LLM reading actual signal content."""
        context = AMPRION_CONTEXT.get(category, {})
        entities = _aggregate_entities(signals)

        # Build richer signal digest — top 15 signals with content snippets
        signal_block = ""
        for s in signals[:15]:
            title = s.get("title", "Untitled")
            content = (s.get("content") or "")[:200]
            source = s.get("source_name", "Unknown")
            signal_block += f"- [{source}] {title}\n  {content}\n\n"

        system = (
            "You are a strategic analyst at Amprion, a German TSO (Transmission System Operator) "
            "operating 11,000 km of extra-high-voltage grid. You write detailed contextual analyses "
            "for the AGTS (Agentic Global Trend Scanner) system. Your audience is Amprion's "
            "Strategy & Innovation department. Use precise energy terminology. Be specific about "
            "implications for Amprion's grid operations, projects, and business areas."
        )

        project_str = ", ".join(projects) if projects else "None directly linked"
        org_str = ", ".join(entities["organizations"][:6]) if entities["organizations"] else "Not yet identified"
        reg_str = ", ".join(entities["regulations"][:4]) if entities["regulations"] else "None detected"

        prompt = f"""Write a contextual analysis ("Deep Dive") of maximum 2000 characters for this trend.

TREND: {trend_name}
CATEGORY: {category}
TIER: {tier} (strategic weight {weight}/100)
NATURE: {nature} — {_nature_phrase(nature)}
PRIORITY SCORE: {scores.get('priority_score', 0):.1f}/10
TRL: {maturity_score}
TIME TO IMPACT: {time_to_impact}
GROWTH RATE: {growth_rate:+.0f}%
SIGNAL COUNT: {signal_count}

AMPRION CONTEXT: {context.get('why_amprion', 'N/A')}
RISK SCENARIO: {context.get('risk_scenario', 'N/A')}
GEOGRAPHIC FOCUS: {', '.join(context.get('regions', []))}
LINKED PROJECTS: {project_str}

DETECTED KEY PLAYERS: {org_str}
REGULATORY DRIVERS: {reg_str}

RECENT SIGNALS (read these carefully):
{signal_block}

Structure your response as:
1. DRIVERS: What is causing this trend to move now? (cite specific signals)
2. KEY PLAYERS: Which companies, institutions, or regulators are driving this?
3. AMPRION IMPLICATIONS: How does this specifically affect Amprion's operations, projects, or strategy? Reference specific regions (e.g., "grid congestion in the Ruhr region") and projects.
4. OUTLOOK: What should Amprion watch for in the next 6-12 months?

Requirements:
- Maximum 2000 characters
- Use section headers (DRIVERS, KEY PLAYERS, AMPRION IMPLICATIONS, OUTLOOK)
- Be specific — name actual projects, regions, regulations
- Reference evidence from the signal titles/content
- No generic filler — every sentence should carry information"""

        return self._call_llm(system, prompt, max_tokens=800)


# ═════════════════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

class NarrativeGenerator:
    """
    Unified interface — tries LLM first, falls back to template.

    Usage:
        gen = NarrativeGenerator(mode="template")   # template only
        gen = NarrativeGenerator(mode="llm")         # LLM with template fallback
        gen = NarrativeGenerator(mode="auto")        # LLM if key exists, else template
    """

    def __init__(self, mode: str = "auto"):
        self.template = TemplateNarrative()
        self.llm = None
        self.mode = mode

        if mode in ("llm", "auto"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.llm = LLMNarrative(api_key=api_key)
                logger.info("NarrativeGenerator: LLM mode active (Anthropic API)")
            elif mode == "llm":
                logger.warning("LLM mode requested but no ANTHROPIC_API_KEY. Falling back to template.")

    def generate_all(
        self,
        category: str,
        trend_name: str,
        scores: Dict[str, Any],
        signal_count: int,
        growth_rate: float,
        tier: str,
        weight: int,
        rationale: str,
        strategic_impact: str,
        projects: List[str],
        nature: str,
        time_to_impact: str,
        signals: List[Dict],
        maturity_score: int = 5,
    ) -> Dict[str, str]:
        """
        Generate all narrative fields at once.

        Returns:
            {
                "description_short": "...",    # The Brief (250 chars)
                "description_full": "...",      # The Deep Dive (2000 chars)
                "so_what_summary": "...",        # 2-sentence executive summary
                "key_players": [...],            # Aggregated entities
                "ai_reasoning": "...",           # Which mode generated this
            }
        """
        result = {}
        used_llm = False

        # ── Brief
        if self.llm:
            brief = self.llm.generate_brief(
                category, trend_name, scores, signal_count, growth_rate, tier, signals
            )
            if brief:
                result["description_short"] = brief[:250]
                used_llm = True

        if "description_short" not in result:
            result["description_short"] = self.template.generate_brief(
                category, trend_name, scores, signal_count, growth_rate,
                tier, strategic_impact, projects, nature, time_to_impact
            )

        # ── Deep Dive
        if self.llm:
            deep = self.llm.generate_deep_dive(
                category, trend_name, scores, signal_count, growth_rate,
                tier, weight, projects, nature, time_to_impact, signals, maturity_score
            )
            if deep:
                result["description_full"] = deep[:2000]
                used_llm = True

        if "description_full" not in result:
            result["description_full"] = self.template.generate_deep_dive(
                category, trend_name, scores, signal_count, growth_rate,
                tier, weight, rationale, strategic_impact, projects,
                nature, time_to_impact, signals, maturity_score
            )

        # ── So What (always template — it's structured enough)
        result["so_what_summary"] = self.template.generate_so_what(
            category, trend_name, tier, nature, growth_rate, time_to_impact, projects
        )

        # ── Key Players (always from entity aggregation — ground truth)
        entities = _aggregate_entities(signals)
        result["key_players"] = entities["organizations"][:8]

        # ── Reasoning log
        result["ai_reasoning"] = (
            f"Generated via {'LLM (Anthropic Claude)' if used_llm else 'template engine'}. "
            f"Based on {signal_count} signals, {len(signals)} analyzed for entities. "
            f"Entity extraction found {len(entities['organizations'])} organizations, "
            f"{len(entities['technologies'])} technologies, "
            f"{len(entities['regulations'])} regulatory drivers."
        )

        return result
