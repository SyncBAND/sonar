"""
SONAR.AI / AGTS - TSO Trend Interpreter
========================================
Interprets signals and trends from a TSO perspective.

Handles special scenarios:
- Biogas/Biomethane: Gas grid wheeling implications
- V2G/VPP: Distributed flexibility aggregation
- Industrial load shifts: Green steel, electrolysis hubs
- Market evolution: 15-min trading, flexibility markets

TSO-General AND Amprion-Specific interpretations.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# =============================================================================
# TSO INTERPRETATION RULES
# =============================================================================

@dataclass
class TSOInterpretation:
    """Structured interpretation of a trend for TSO decision-makers."""
    category: str
    tso_relevance: str  # Critical, High, Medium, Low
    strategic_implication: str
    amprion_specific: str
    linked_projects: List[str]
    business_areas_impacted: List[str]
    action_recommendation: str
    risk_assessment: str
    opportunity_assessment: str
    time_horizon: str
    regulatory_considerations: str

# =============================================================================
# INTERPRETATION RULES BY CATEGORY
# =============================================================================

INTERPRETATION_RULES = {
    # =========================================================================
    # BIOGAS / BIOMETHANE - Special Handling
    # =========================================================================
    "biogas_biomethane": {
        "tso_relevance": "Medium-High",
        "strategic_implications": [
            "Renewable gas can be WHEELED through existing gas infrastructure to industrial consumers",
            "Reduces need for new electrical transmission capacity when gas grid parallels electricity corridor",
            "Biogas CHP plants provide dispatchable generation for grid balancing",
            "Biomethane injection points create new flexibility resources",
            "Sector coupling opportunity: Gas TSO + Electricity TSO coordination"
        ],
        "amprion_specific": [
            "Ruhr region industrial customers can receive green gas via gas grid, reducing electrical load growth",
            "Biogas plants in northern Germany can balance wind intermittency",
            "Potential coordination with gas TSOs (OGE, Thyssengas) for sector coupling",
            "Relevant to 'hybridge' hydrogen/gas strategy"
        ],
        "business_areas": ["M", "SO", "CS"],  # Market, System Operation, Corporate Strategy
        "linked_projects": ["hybridge"],
        "action_recommendation": "Monitor biogas injection capacity growth. Coordinate with gas TSOs for integrated planning.",
        "risk_assessment": "Low direct risk. Opportunity cost if not leveraged for sector coupling.",
        "opportunity_assessment": "HIGH - Reduces electrical grid stress, enables green industry claims"
    },
    
    # =========================================================================
    # V2G / VIRTUAL POWER PLANTS - Special Handling
    # =========================================================================
    "flexibility_vpp": {
        "tso_relevance": "Critical",
        "strategic_implications": [
            "Aggregated EVs and batteries act as DISTRIBUTED BALANCING RESOURCES",
            "VPPs can provide FCR, aFRR, mFRR services at transmission level",
            "1000 EVs @ 10kW bidirectional = 10MW virtual battery",
            "Reduces need for dedicated balancing power plants",
            "TSO-DSO coordination essential: Flexibility must be 'visible' to TSO",
            "Systemmarkt design enables TSO access to DSO-connected flexibility"
        ],
        "amprion_specific": [
            "Ruhr region has high EV density - significant flexibility potential",
            "Amprion's 'Systemmarkt' platform designed for VPP integration",
            "Critical for managing renewable variability from Korridor B offshore wind",
            "VPPs reduce redispatch costs by providing local flexibility"
        ],
        "business_areas": ["SO", "M", "ITD"],  # System Operation, Market, IT/Digital
        "linked_projects": ["Systemmarkt", "Grid Boosters"],
        "action_recommendation": "Accelerate Systemmarkt rollout. Partner with VPP aggregators.",
        "risk_assessment": "MEDIUM - If VPPs bypass TSO for DSO-only services, reduces TSO flexibility access",
        "opportunity_assessment": "CRITICAL - Core enabler for 100% renewable grid operation"
    },
    
    "e_mobility_v2g": {
        "tso_relevance": "High",
        "strategic_implications": [
            "EVs shift from LOAD to FLEXIBILITY RESOURCE with bidirectional charging",
            "Unmanaged EV charging creates peak demand challenges",
            "Managed/smart charging reduces need for grid reinforcement",
            "V2G provides sub-second frequency response capability",
            "ISO 15118 enables automated grid service provision"
        ],
        "amprion_specific": [
            "Heavy industry in Amprion's area adopting e-trucks (logistics, mining)",
            "Autobahn A1/A2/A3 corridors require ultra-fast charging infrastructure",
            "E-mobility increases load but V2G turns it into flexibility asset"
        ],
        "business_areas": ["SO", "GP", "AM"],
        "linked_projects": [],
        "action_recommendation": "Integrate EV charging forecasts into load planning. Support V2G standards.",
        "risk_assessment": "MEDIUM - Unmanaged charging creates peak demand spikes",
        "opportunity_assessment": "HIGH - Mobile battery fleet for grid services"
    },
    
    # =========================================================================
    # GRID INFRASTRUCTURE & STABILITY
    # =========================================================================
    "grid_infrastructure": {
        "tso_relevance": "Critical",
        "strategic_implications": [
            "Core TSO business: Transmission capacity expansion",
            "HVDC enables long-distance renewable transport with lower losses",
            "Multi-terminal HVDC creates flexible 'supergrid'",
            "Underground cabling increases cost but improves public acceptance"
        ],
        "amprion_specific": [
            "€36.4B investment program focused on HVDC corridors",
            "Korridor B: 4GW offshore wind to Ruhr region",
            "Rhein-Main Link: 8GW to Frankfurt industrial area",
            "A-Nord: Underground HVDC to Lower Rhine heavy industry"
        ],
        "business_areas": ["GP", "AM", "CS"],
        "linked_projects": ["Korridor B", "Rhein-Main Link", "A-Nord", "Ultranet", "SuedLink"],
        "action_recommendation": "Prioritize trends affecting HVDC converter technology and underground cabling.",
        "risk_assessment": "LOW technical risk for proven technologies, HIGH regulatory/permitting risk",
        "opportunity_assessment": "ESSENTIAL - Core business"
    },
    
    "grid_stability": {
        "tso_relevance": "Critical",
        "strategic_implications": [
            "Frequency stability threatened by loss of synchronous generation",
            "Grid-forming inverters replace rotating inertia",
            "STATCOM and SVC provide reactive power support",
            "Grid boosters (batteries) enable curative congestion management"
        ],
        "amprion_specific": [
            "STATCOM deployments at Gersteinwerk and Polsum substations",
            "Grid Boosters allow higher line utilization during peak demand",
            "Critical as nuclear/coal plants retire in Amprion's area"
        ],
        "business_areas": ["SO", "AM"],
        "linked_projects": ["STATCOM Gersteinwerk", "STATCOM Polsum", "Grid Boosters"],
        "action_recommendation": "Fast-track grid-forming inverter pilots. Expand Grid Booster network.",
        "risk_assessment": "CRITICAL - Stability is non-negotiable",
        "opportunity_assessment": "HIGH - Curative management reduces CAPEX vs new lines"
    },
    
    # =========================================================================
    # OFFSHORE SYSTEMS
    # =========================================================================
    "offshore_systems": {
        "tso_relevance": "Critical",
        "strategic_implications": [
            "70 GW offshore wind target by 2045 requires massive grid connections",
            "Offshore hubs/energy islands enable multi-country connections",
            "Submarine cables are long-lead-time items",
            "Floating offshore platforms for deeper waters"
        ],
        "amprion_specific": [
            "BalWin1 & BalWin2: North Sea connections",
            "NeuLink: Pioneering Germany-UK hybrid interconnector",
            "Offshore wind landing points in Lower Saxony"
        ],
        "business_areas": ["GP", "AM", "CS"],
        "linked_projects": ["BalWin1", "BalWin2", "NeuLink"],
        "action_recommendation": "Monitor offshore hub developments. Coordinate with TenneT, 50Hertz.",
        "risk_assessment": "HIGH - Long timelines, weather dependencies, supply chain constraints",
        "opportunity_assessment": "ESSENTIAL - 70 GW offshore wind requires TSO-scale connections"
    },
    
    # =========================================================================
    # ENERGY STORAGE
    # =========================================================================
    "energy_storage": {
        "tso_relevance": "Critical",
        "strategic_implications": [
            "Grid-scale batteries enable curative congestion management",
            "Long-duration storage (>4h) critical for multi-day wind lulls",
            "Solid-state batteries may enable smaller footprint",
            "Flow batteries for transmission-level applications"
        ],
        "amprion_specific": [
            "Grid Boosters are transmission-level battery deployments",
            "Storage can defer or reduce HVDC corridor capacity needs",
            "Batteries at strategic substations provide instant reserves"
        ],
        "business_areas": ["SO", "AM", "M"],
        "linked_projects": ["Grid Boosters"],
        "action_recommendation": "Expand Grid Booster program. Evaluate long-duration storage technologies.",
        "risk_assessment": "MEDIUM - Technology evolving rapidly, cost declining",
        "opportunity_assessment": "HIGH - Enables curative management, reduces CAPEX"
    },
    
    # =========================================================================
    # HYDROGEN & POWER-TO-GAS
    # =========================================================================
    "hydrogen_p2g": {
        "tso_relevance": "High",
        "strategic_implications": [
            "Large electrolyzers create significant new electrical LOAD",
            "Hydrogen can store excess renewable generation for later use",
            "Industrial hydrogen hubs need dedicated high-voltage connections",
            "Sector coupling: Electricity TSO must coordinate with hydrogen network operators"
        ],
        "amprion_specific": [
            "'hybridge' project: Amprion + OGE demonstrating sector coupling",
            "Ruhr region steel industry transitioning to green hydrogen",
            "Potential need for dedicated connections to electrolyzer sites",
            "Hydrogen storage can provide grid flexibility services"
        ],
        "business_areas": ["GP", "CS", "M"],
        "linked_projects": ["hybridge"],
        "action_recommendation": "Map electrolyzer project pipeline. Plan grid connections for hydrogen hubs.",
        "risk_assessment": "MEDIUM - Timing uncertain, depends on hydrogen economics",
        "opportunity_assessment": "HIGH - New load = new grid revenue; flexibility potential"
    },
    
    # =========================================================================
    # RENEWABLES INTEGRATION
    # =========================================================================
    "renewables_integration": {
        "tso_relevance": "Critical",
        "strategic_implications": [
            "Variable generation requires enhanced forecasting and flexibility",
            "Curtailment indicates insufficient transmission or flexibility",
            "Solar peaks challenge midday stability",
            "Wind variability requires reserve procurement changes"
        ],
        "amprion_specific": [
            "Offshore wind from North Sea is primary renewable source",
            "Solar growth in southern Germany creates north-south flow challenges",
            "Reducing curtailment is key KPI"
        ],
        "business_areas": ["SO", "GP", "M"],
        "linked_projects": ["Korridor B", "A-Nord", "Ultranet"],
        "action_recommendation": "Improve renewable forecasting. Accelerate north-south corridors.",
        "risk_assessment": "LOW technology risk, HIGH volume risk if integration lags",
        "opportunity_assessment": "ESSENTIAL - Energy transition core"
    },
    
    # =========================================================================
    # AI & DIGITALIZATION
    # =========================================================================
    "ai_grid_optimization": {
        "tso_relevance": "High",
        "strategic_implications": [
            "AI enables predictive operations and autonomous control",
            "Multi-agent systems can manage complexity beyond human capability",
            "Machine learning improves forecasting accuracy",
            "Generative AI supports planning and documentation"
        ],
        "amprion_specific": [
            "Auto-Trader: AI-driven renewable energy marketing",
            "Agentic AI for control room support",
            "Predictive maintenance for critical assets"
        ],
        "business_areas": ["ITD", "SO", "AM"],
        "linked_projects": ["Auto-Trader"],
        "action_recommendation": "Pilot agentic AI in control room. Scale successful use cases.",
        "risk_assessment": "MEDIUM - Requires robust validation before operational deployment",
        "opportunity_assessment": "HIGH - Competitive advantage, OPEX reduction"
    },
    
    "digital_twin_simulation": {
        "tso_relevance": "High",
        "strategic_implications": [
            "Digital twins enable what-if analysis without grid risk",
            "4D modeling supports corridor planning (space + time)",
            "Real-time simulation improves operator training"
        ],
        "amprion_specific": [
            "Korridor B corridor modeled as digital twin",
            "BIM/GIS integration for asset management",
            "Scenario analysis for grid development planning"
        ],
        "business_areas": ["ITD", "AM", "GP"],
        "linked_projects": ["Korridor B", "Rhein-Main Link"],
        "action_recommendation": "Expand digital twin coverage. Integrate with operational systems.",
        "risk_assessment": "LOW - Proven technology, scaling challenge",
        "opportunity_assessment": "HIGH - Enables better planning and operations"
    },
    
    # =========================================================================
    # CYBERSECURITY
    # =========================================================================
    "cybersecurity_ot": {
        "tso_relevance": "Critical",
        "strategic_implications": [
            "Critical infrastructure is prime target for state actors",
            "OT security requires specialized approaches vs IT security",
            "Post-quantum cryptography needed before quantum computers arrive",
            "Zero trust architecture for SCADA systems"
        ],
        "amprion_specific": [
            "Control room in Brauweiler is mission-critical",
            "KRITIS regulations mandate specific security measures",
            "NIS2 directive expands compliance requirements"
        ],
        "business_areas": ["ITD", "SO"],
        "linked_projects": [],
        "action_recommendation": "Accelerate PQC migration planning. Implement zero trust.",
        "risk_assessment": "CRITICAL - Existential risk if compromised",
        "opportunity_assessment": "Defensive - No business upside, but essential"
    },
    
    # =========================================================================
    # ENERGY TRADING & MARKETS
    # =========================================================================
    "energy_trading": {
        "tso_relevance": "High",
        "strategic_implications": [
            "Market design evolution affects TSO balancing responsibilities",
            "15-minute trading improves renewable integration",
            "Flow-based market coupling optimizes cross-border capacity",
            "Flexibility markets enable TSO-DSO coordination"
        ],
        "amprion_specific": [
            "Systemmarkt: Amprion's flexibility market platform",
            "Auto-Trader: Automated renewable marketing",
            "Cross-border coordination with TenneT (NL), Elia (BE), RTE (FR)"
        ],
        "business_areas": ["M", "SO"],
        "linked_projects": ["Systemmarkt", "Auto-Trader"],
        "action_recommendation": "Lead Systemmarkt evolution. Engage in market design consultations.",
        "risk_assessment": "MEDIUM - Regulatory uncertainty",
        "opportunity_assessment": "HIGH - Better market design = lower balancing costs"
    },
    
    # =========================================================================
    # REGULATORY & POLICY
    # =========================================================================
    "regulatory_policy": {
        "tso_relevance": "High",
        "strategic_implications": [
            "Incentive regulation determines allowed returns",
            "Network codes define technical requirements",
            "Permitting speed affects project timelines",
            "European harmonization enables cross-border operations"
        ],
        "amprion_specific": [
            "BNetzA negotiations on equity yield rate",
            "X-gen efficiency factor challenges during growth phase",
            "RED III 'overriding public interest' status"
        ],
        "business_areas": ["CS", "GP"],
        "linked_projects": [],
        "action_recommendation": "Engage proactively in regulatory consultations.",
        "risk_assessment": "HIGH - Regulatory decisions have major financial impact",
        "opportunity_assessment": "MEDIUM - Favorable regulation enables investment"
    },
    
    # =========================================================================
    # HEAT & DISTRICT ENERGY
    # =========================================================================
    "heat_district_energy": {
        "tso_relevance": "Medium",
        "strategic_implications": [
            "Large heat pumps create new electrical load",
            "District heating can absorb excess electricity",
            "Power-to-heat provides flexibility services",
            "Industrial heat electrification increases load"
        ],
        "amprion_specific": [
            "Ruhr region has extensive district heating networks",
            "Industrial heat demand from chemical and steel sectors",
            "Sector coupling opportunity with heat networks"
        ],
        "business_areas": ["GP", "M"],
        "linked_projects": [],
        "action_recommendation": "Map large heat pump project pipeline. Include in load forecasts.",
        "risk_assessment": "LOW - Generally positive for grid (new load, flexibility)",
        "opportunity_assessment": "MEDIUM - New load = new grid services"
    }
}

# =============================================================================
# TSO TREND INTERPRETER CLASS
# =============================================================================

class TSOTrendInterpreter:
    """
    Interprets trends from TSO perspective.
    
    Provides:
    - Strategic implications for TSOs
    - Amprion-specific considerations
    - Action recommendations
    - Risk/opportunity assessments
    """
    
    def __init__(self):
        self.rules = INTERPRETATION_RULES
    
    def interpret_trend(
        self, 
        category: str,
        trend_name: str,
        signals: List[Dict[str, Any]],
        additional_context: Optional[Dict] = None
    ) -> TSOInterpretation:
        """
        Generate TSO interpretation for a trend.
        
        Args:
            category: TSO category (e.g., 'biogas_biomethane')
            trend_name: Name of the trend
            signals: Related signals
            additional_context: Extra context (linked projects, etc.)
        
        Returns:
            TSOInterpretation with all relevant fields
        """
        
        rules = self.rules.get(category, self._get_default_rules())
        
        # Analyze signals for specific indicators
        signal_analysis = self._analyze_signals(signals)
        
        # Build interpretation
        interpretation = TSOInterpretation(
            category=category,
            tso_relevance=rules.get("tso_relevance", "Medium"),
            strategic_implication=self._select_implication(rules, signal_analysis),
            amprion_specific=self._select_amprion_specific(rules, signal_analysis),
            linked_projects=self._find_linked_projects(rules, signals, additional_context),
            business_areas_impacted=rules.get("business_areas", ["ITD"]),
            action_recommendation=rules.get("action_recommendation", "Monitor and assess"),
            risk_assessment=rules.get("risk_assessment", "To be determined"),
            opportunity_assessment=rules.get("opportunity_assessment", "To be determined"),
            time_horizon=self._estimate_time_horizon(signal_analysis),
            regulatory_considerations=self._get_regulatory_considerations(category)
        )
        
        return interpretation
    
    def _analyze_signals(self, signals: List[Dict]) -> Dict[str, Any]:
        """Analyze signal collection for patterns."""
        if not signals:
            return {"count": 0, "growth": "unknown", "sources": []}
        
        analysis = {
            "count": len(signals),
            "sources": list(set(s.get("source_type", "unknown") for s in signals)),
            "has_patents": any(s.get("source_type") == "patent" for s in signals),
            "has_research": any(s.get("source_type") == "research" for s in signals),
            "has_regulatory": any(s.get("source_type") == "regulatory" for s in signals),
            "keywords": self._extract_common_keywords(signals)
        }
        
        # Estimate growth
        if len(signals) > 20:
            analysis["growth"] = "accelerating"
        elif len(signals) > 10:
            analysis["growth"] = "growing"
        else:
            analysis["growth"] = "emerging"
        
        return analysis
    
    def _extract_common_keywords(self, signals: List[Dict]) -> List[str]:
        """Extract most common keywords from signals."""
        from collections import Counter
        
        all_keywords = []
        for signal in signals:
            keywords = signal.get("keywords", [])
            if keywords:
                all_keywords.extend(keywords)
        
        if not all_keywords:
            return []
        
        counter = Counter(all_keywords)
        return [kw for kw, _ in counter.most_common(10)]
    
    def _select_implication(self, rules: Dict, analysis: Dict) -> str:
        """Select most relevant strategic implication."""
        implications = rules.get("strategic_implications", ["No specific implication"])
        
        # Return first (most important) implication
        return implications[0] if implications else "Monitor for strategic impact"
    
    def _select_amprion_specific(self, rules: Dict, analysis: Dict) -> str:
        """Select Amprion-specific consideration."""
        amprion = rules.get("amprion_specific", ["Review for Amprion relevance"])
        return amprion[0] if amprion else "Assess Amprion-specific impact"
    
    def _find_linked_projects(
        self, 
        rules: Dict, 
        signals: List[Dict],
        additional_context: Optional[Dict]
    ) -> List[str]:
        """Find linked Amprion projects."""
        projects = set(rules.get("linked_projects", []))
        
        # Add from signals
        for signal in signals:
            signal_projects = signal.get("linked_projects", [])
            if signal_projects:
                projects.update(signal_projects)
        
        # Add from additional context
        if additional_context:
            context_projects = additional_context.get("linked_projects", [])
            projects.update(context_projects)
        
        return list(projects)
    
    def _estimate_time_horizon(self, analysis: Dict) -> str:
        """Estimate time-to-impact."""
        if analysis.get("has_regulatory"):
            return "1-3 years (regulatory activity indicates near-term action)"
        elif analysis.get("has_patents") and analysis.get("count", 0) > 15:
            return "1-3 years (patent activity suggests commercialization)"
        elif analysis.get("growth") == "accelerating":
            return "1-3 years"
        elif analysis.get("growth") == "growing":
            return "3-5 years"
        else:
            return "5+ years (early stage)"
    
    def _get_regulatory_considerations(self, category: str) -> str:
        """Get regulatory considerations for category."""
        regulatory_map = {
            "grid_infrastructure": "BNetzA approval required. RED III overriding public interest applies.",
            "grid_stability": "System services regulated by BNetzA. Grid codes apply.",
            "offshore_systems": "Maritime spatial planning. BSH approval required.",
            "energy_storage": "Storage regulatory framework evolving. Market participation rules.",
            "flexibility_vpp": "Aggregator regulations. Prequalification requirements.",
            "e_mobility_v2g": "V2G standards (ISO 15118). Grid connection rules.",
            "hydrogen_p2g": "Hydrogen network regulation emerging. Sector coupling rules.",
            "cybersecurity_ot": "KRITIS requirements. NIS2 directive. IT-Sicherheitsgesetz.",
            "biogas_biomethane": "Gas grid injection rules. Renewable gas certificates.",
            "regulatory_policy": "Direct regulatory topic - monitor BNetzA consultations."
        }
        return regulatory_map.get(category, "Standard TSO regulatory framework applies.")
    
    def _get_default_rules(self) -> Dict:
        """Get default interpretation rules."""
        return {
            "tso_relevance": "Medium",
            "strategic_implications": ["Assess impact on TSO operations"],
            "amprion_specific": ["Review for Amprion-specific considerations"],
            "business_areas": ["ITD"],
            "linked_projects": [],
            "action_recommendation": "Monitor and assess strategic relevance",
            "risk_assessment": "To be determined based on detailed analysis",
            "opportunity_assessment": "To be determined based on detailed analysis"
        }
    
    # =========================================================================
    # SPECIAL SCENARIO HANDLERS
    # =========================================================================
    
    def interpret_biogas_increase(self, signals: List[Dict]) -> Dict[str, Any]:
        """
        Special handler for biogas/biomethane increase scenario.
        
        Key insight: Biogas can be WHEELED through gas infrastructure,
        reducing electrical grid stress.
        """
        return {
            "scenario": "Biogas Production Increase Detected",
            "tso_implication": "GRID STRESS REDUCTION OPPORTUNITY",
            "explanation": [
                "Increased biogas production means more renewable gas available",
                "This gas can be WHEELED through existing gas pipelines to industrial consumers",
                "Industrial consumers receiving gas reduce their electrical load on TSO grid",
                "Biogas CHP plants provide DISPATCHABLE generation for grid balancing"
            ],
            "amprion_specific": [
                "Ruhr region heavy industry can receive green gas via gas grid",
                "Coordinate with gas TSOs (OGE, Thyssengas) for integrated planning",
                "Biogas plants in wind-rich areas can balance offshore wind variability",
                "Relevant to 'hybridge' project for sector coupling demonstration"
            ],
            "action_items": [
                "Map biogas plant locations relative to Amprion grid",
                "Identify industrial customers who could switch to gas",
                "Coordinate with gas TSOs for integrated energy planning",
                "Include biogas CHP in flexibility resource inventory"
            ],
            "strategic_classification": "Accelerator - Reduces electrical grid investment needs"
        }
    
    def interpret_vpp_growth(self, signals: List[Dict]) -> Dict[str, Any]:
        """
        Special handler for VPP/aggregation growth scenario.
        
        Key insight: VPPs aggregate distributed resources into
        transmission-level services.
        """
        signal_count = len(signals)
        
        return {
            "scenario": "Virtual Power Plant / Aggregation Growth Detected",
            "tso_implication": "DISTRIBUTED FLEXIBILITY RESOURCE EMERGING",
            "explanation": [
                "VPPs AGGREGATE many small resources (EVs, batteries, heat pumps) into grid services",
                "Aggregation makes distributed resources VISIBLE and CONTROLLABLE at TSO level",
                f"Signal volume ({signal_count}) suggests {'mature' if signal_count > 15 else 'emerging'} market",
                "VPPs can provide FCR, aFRR, mFRR - traditional TSO balancing products",
                "1000 EVs @ 10kW bidirectional = 10MW virtual battery equivalent"
            ],
            "amprion_specific": [
                "Amprion's 'Systemmarkt' platform designed for VPP participation",
                "VPPs reduce need for dedicated balancing power plants",
                "Critical for managing offshore wind variability from Korridor B",
                "TSO-DSO coordination required: Amprion must 'see' DSO-connected flexibility"
            ],
            "flexibility_calculation": {
                "example_1000_evs": {
                    "capacity_mw": 10,
                    "duration_hours": 2,
                    "response_time_ms": 100,
                    "services_possible": ["FCR", "aFRR", "mFRR", "peak shaving"]
                },
                "scaling_potential": "Germany: 15M EVs by 2030 = potential 150 GW flexibility"
            },
            "action_items": [
                "Accelerate Systemmarkt rollout and VPP onboarding",
                "Partner with major VPP operators (sonnen, Next Kraftwerke, etc.)",
                "Develop TSO-DSO data exchange protocols",
                "Include VPP capacity in system adequacy assessments"
            ],
            "strategic_classification": "Critical Enabler - Essential for 100% renewable grid"
        }
    
    def interpret_industrial_electrification(self, signals: List[Dict]) -> Dict[str, Any]:
        """
        Special handler for industrial electrification (green steel, etc.)
        """
        return {
            "scenario": "Industrial Electrification Wave Detected",
            "tso_implication": "MAJOR NEW LOAD GROWTH",
            "explanation": [
                "Green steel, hydrogen electrolyzers, and chemical processes electrifying",
                "Individual sites can require 100+ MW connections",
                "Load growth concentrated in industrial regions (Ruhr, Rhine-Main)",
                "Timing depends on hydrogen economics and carbon pricing"
            ],
            "amprion_specific": [
                "Ruhr region steel industry (ThyssenKrupp, ArcelorMittal) transitioning",
                "Chemical industry in Leverkusen, Dormagen regions",
                "May require new dedicated transmission connections",
                "Potential need for on-site generation or storage"
            ],
            "load_growth_estimates": {
                "single_dri_plant_mw": 200,
                "electrolyzer_100mw_example": 100,
                "ruhr_region_potential_gw": 5
            },
            "action_items": [
                "Map industrial decarbonization project pipeline",
                "Proactively plan transmission connections to industrial sites",
                "Include industrial load scenarios in Grid Development Plan",
                "Coordinate with industrial customers on connection timelines"
            ],
            "strategic_classification": "Load Growth Driver - New revenue but requires investment"
        }
    
    def generate_so_what_summary(
        self, 
        category: str, 
        trend_name: str,
        priority_score: float
    ) -> str:
        """
        Generate the "So What?" summary for executive decision-makers.
        
        2-sentence format: What happens if we act vs. don't act.
        """
        so_what_templates = {
            "grid_infrastructure": (
                f"Acting now on {trend_name} enables Amprion to accelerate corridor completion and reduce redispatch costs. "
                f"Inaction risks project delays and higher grid charges for customers."
            ),
            "grid_stability": (
                f"Implementing {trend_name} maintains system security as synchronous generation retires. "
                f"Failure to act risks frequency stability events with potential blackout consequences."
            ),
            "energy_storage": (
                f"Deploying {trend_name} technology enables curative congestion management and defers new line construction. "
                f"Without action, Amprion faces higher CAPEX for traditional grid reinforcement."
            ),
            "flexibility_vpp": (
                f"Integrating {trend_name} provides distributed flexibility for grid balancing at lower cost than conventional assets. "
                f"Missing this opportunity means higher balancing costs and potential flexibility shortfall."
            ),
            "e_mobility_v2g": (
                f"Supporting {trend_name} turns EVs from load challenge into flexibility resource. "
                f"Ignoring V2G means managing EV charging peaks without the benefit of vehicle batteries for grid services."
            ),
            "hydrogen_p2g": (
                f"Engaging with {trend_name} positions Amprion for sector coupling revenue and load growth. "
                f"Passive approach risks being excluded from hydrogen network value chain."
            ),
            "cybersecurity_ot": (
                f"Implementing {trend_name} protects critical infrastructure from evolving cyber threats. "
                f"Delayed action exposes Amprion to potential grid compromise with national security implications."
            ),
            "biogas_biomethane": (
                f"Coordinating on {trend_name} enables sector coupling and reduces electrical grid stress. "
                f"Ignoring gas-electricity synergies means suboptimal infrastructure investment."
            ),
            "ai_grid_optimization": (
                f"Adopting {trend_name} improves operational efficiency and enables autonomous grid management. "
                f"Falling behind on AI adoption means higher OPEX and reduced competitive positioning."
            )
        }
        
        # Get template or generate generic
        if category in so_what_templates:
            return so_what_templates[category]
        else:
            return (
                f"Engaging with {trend_name} (Priority: {priority_score:.1f}/10) may provide strategic advantage. "
                f"Further analysis recommended to quantify impact on Amprion operations and strategy."
            )

# =============================================================================
# SINGLETON
# =============================================================================

_interpreter = None

def get_trend_interpreter() -> TSOTrendInterpreter:
    """Get singleton interpreter instance."""
    global _interpreter
    if _interpreter is None:
        _interpreter = TSOTrendInterpreter()
    return _interpreter

# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    interpreter = get_trend_interpreter()
    
    # Test biogas scenario
    print("=" * 70)
    print("BIOGAS INCREASE SCENARIO")
    print("=" * 70)
    result = interpreter.interpret_biogas_increase([{"title": "test"}] * 5)
    for key, value in result.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        elif isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {value}")
    
    # Test VPP scenario
    print("\n" + "=" * 70)
    print("VPP GROWTH SCENARIO")
    print("=" * 70)
    result = interpreter.interpret_vpp_growth([{"title": "test"}] * 20)
    for key, value in result.items():
        print(f"\n{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        elif isinstance(value, dict):
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {value}")
    
    # Test So What summary
    print("\n" + "=" * 70)
    print("SO WHAT SUMMARIES")
    print("=" * 70)
    for cat in ["grid_stability", "flexibility_vpp", "biogas_biomethane"]:
        print(f"\n{cat}:")
        print(interpreter.generate_so_what_summary(cat, f"Test Trend in {cat}", 8.5))
