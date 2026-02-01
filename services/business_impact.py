"""
SONAR.AI / AGTS - Business Impact Analyzer
===========================================
Detailed analysis: What do trend signals mean for TSOs & Amprion?

For each category, answers:
- What does an INCREASE in signals mean?
- What does a DECREASE in signals mean?
- SHORT-TERM business impact (0-2 years)
- LONG-TERM business impact (3-10 years)
- Revenue implications
- Cost implications
- Operational changes required
- Strategic positioning recommendations

This is the "SO WHAT?" layer - translating signals into business decisions.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TrendDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    EMERGING = "emerging"

@dataclass
class BusinessImpact:
    """Complete business impact analysis for a trend."""
    
    # Trend identification
    category: str
    trend_name: str
    direction: TrendDirection
    signal_count: int
    growth_rate: float  # % change
    
    # The "SO WHAT?" Summary
    executive_summary: str
    
    # Short-term impact (0-2 years)
    short_term_revenue: str
    short_term_costs: str
    short_term_operations: str
    short_term_actions: List[str]
    
    # Long-term impact (3-10 years)
    long_term_revenue: str
    long_term_costs: str
    long_term_strategy: str
    long_term_positioning: str
    
    # Amprion-specific
    amprion_opportunities: List[str]
    amprion_risks: List[str]
    amprion_projects_affected: List[str]
    
    # Quantification (where possible)
    estimated_investment_eur: Optional[str] = None
    estimated_revenue_impact: Optional[str] = None
    estimated_cost_savings: Optional[str] = None
    
    # Timing
    action_urgency: str = "Medium"  # Critical, High, Medium, Low
    decision_timeline: str = "6-12 months"


# =============================================================================
# BUSINESS IMPACT RULES BY CATEGORY
# =============================================================================

BUSINESS_IMPACT_RULES = {
    
    # =========================================================================
    # BIOGAS / BIOMETHANE - YOUR EXAMPLE
    # =========================================================================
    "biogas_biomethane": {
        "category_name": "Biogas & Biomethane",
        
        "increase_means": """
📈 INCREASING BIOGAS SIGNALS INDICATES:
- More biogas plants being built/planned
- More biomethane being injected into gas grid
- Growing demand for green gas certificates
- Industrial customers seeking renewable gas supply

🔑 KEY TSO INSIGHT: Biogas can be "WHEELED" through gas infrastructure!
- If a customer in Ruhr region wants green energy...
- Instead of building new electrical transmission...
- They can receive biomethane via existing gas grid
- TSO avoids new CAPEX, customer gets green energy
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- Biogas economics becoming challenging
- Feedstock supply issues (corn, organic waste)
- Policy uncertainty affecting investment
- Competition from green hydrogen
        """,
        
        "short_term": {
            "revenue": """
- NEUTRAL to POSITIVE: Biogas doesn't directly generate TSO revenue
- INDIRECT BENEFIT: Reduced pressure for new electrical connections
- GREEN PREMIUM: Some customers may pay premium for green gas delivery coordination
            """,
            "costs": """
- SAVINGS: €10-50M per avoided grid reinforcement project
- NEW COSTS: Coordination costs with gas TSOs (OGE, Thyssengas)
- MONITORING: Investment in sector coupling analytics (~€1-2M)
            """,
            "operations": """
- Load forecasting: Include biogas CHP dispatch in predictions
- Balancing: Biogas plants provide dispatchable backup
- TSO-DSO-Gas TSO coordination: New communication protocols needed
            """,
            "actions": [
                "Map all biogas plants >1MW in Amprion's grid area",
                "Establish data sharing with OGE/Thyssengas",
                "Model biogas CHP contribution to system services",
                "Identify customers who could switch electrical→gas delivery"
            ]
        },
        
        "long_term": {
            "revenue": """
- SECTOR COUPLING SERVICES: New revenue from integrated gas-electricity planning
- FLEXIBILITY MARKETS: Biogas plants as controllable generation (~€5-15/MWh)
- AVOIDED CAPEX: €100-500M in deferred grid expansion over 10 years
            """,
            "costs": """
- SYSTEMS INTEGRATION: €5-10M for integrated gas-electricity modeling
- WORKFORCE: Sector coupling expertise development
            """,
            "strategy": """
- Position as INTEGRATED ENERGY INFRASTRUCTURE operator
- Partner with gas TSOs for "Energy-as-a-Service" offerings
- Leverage biogas flexibility for renewable integration
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "Energy Delivery Orchestrator"
- Not just electricity TSO but energy infrastructure coordinator
- Offer customers choice: electrons OR molecules
- Optimize total system cost, not just electrical grid
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "Ruhr region industrial customers (steel, chemicals) can receive green gas via gas grid",
                "Biogas plants in northern Germany can balance offshore wind variability",
                "hybridge project synergies: Combined hydrogen + biogas strategy",
                "Reduced electrical grid stress = faster renewable integration"
            ],
            "risks": [
                "If biogas grows without coordination → suboptimal infrastructure investment",
                "Competition: Gas TSOs could capture sector coupling value alone",
                "Regulatory uncertainty: Who owns the 'green gas certificate' value?"
            ],
            "projects_affected": ["hybridge", "Grid expansion planning", "Sector coupling initiatives"]
        },
        
        "quantification": {
            "investment": "€2-5M for coordination systems",
            "revenue_impact": "€10-30M/year in flexibility services (potential)",
            "cost_savings": "€50-200M in avoided grid CAPEX over 10 years"
        }
    },
    
    # =========================================================================
    # FLEXIBILITY / VPP - Virtual Power Plants
    # =========================================================================
    "flexibility_vpp": {
        "category_name": "Flexibility & Virtual Power Plants",
        
        "increase_means": """
📈 INCREASING VPP SIGNALS INDICATES:
- More aggregators entering market
- More distributed assets (batteries, EVs, heat pumps) being connected
- Growing flexibility market liquidity
- TSO-DSO flexibility trading emerging

🔑 KEY TSO INSIGHT: VPPs are DISTRIBUTED BALANCING RESOURCES!
- 1,000 EVs × 10kW bidirectional = 10MW virtual battery
- VPPs can provide FCR, aFRR, mFRR at transmission level
- Reduces need for dedicated balancing power plants
- BUT: TSO must "see" and access these distributed resources
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- Aggregator business models struggling
- Regulatory barriers to market participation
- DSOs blocking VPP access to TSO markets
- Technology/interoperability challenges
        """,
        
        "short_term": {
            "revenue": """
- FLEXIBILITY PROCUREMENT: Lower balancing costs if VPPs compete
- SYSTEMMARKT: Revenue from platform services (~€2-5M/year)
- REDISPATCH REDUCTION: VPPs can provide local solutions
            """,
            "costs": """
- PLATFORM DEVELOPMENT: Systemmarkt enhancements €5-15M
- INTEGRATION: APIs for VPP aggregators €2-3M
- PREQUALIFICATION: Testing and validation processes
            """,
            "operations": """
- Control room: Integrate VPP dispatch into real-time operations
- Forecasting: VPP availability prediction
- Settlement: 15-minute granularity with VPP aggregators
            """,
            "actions": [
                "Accelerate Systemmarkt rollout",
                "Sign partnership agreements with top 10 VPP aggregators",
                "Develop VPP prequalification fast-track process",
                "Pilot project: 50MW VPP for aFRR provision"
            ]
        },
        
        "long_term": {
            "revenue": """
- FLEXIBILITY AS CORE BUSINESS: €50-200M/year market opportunity
- PLATFORM FEES: VPP access to Amprion markets
- AVOIDED PLANT INVESTMENT: No new peakers needed
            """,
            "costs": """
- REDUCED BALANCING COSTS: 30-50% reduction possible
- DIGITAL INFRASTRUCTURE: Ongoing investment €10M/year
            """,
            "strategy": """
- VPPs become PRIMARY balancing resource (not backup)
- TSO becomes FLEXIBILITY MARKET ORCHESTRATOR
- Geographic granularity: Nodal flexibility pricing
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "Flexibility Market Leader"
- Own the Systemmarkt platform = own the market
- Set standards that competitors must follow
- First-mover advantage in EU flexibility market design
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "Ruhr region has high EV/heat pump density = massive flexibility potential",
                "Systemmarkt can become European reference platform",
                "VPPs reduce Korridor B offshore wind balancing costs",
                "Industrial demand response: Aluminum, chemicals can shift 100s of MW"
            ],
            "risks": [
                "DSOs could develop competing local flexibility markets",
                "If VPPs only serve DSO markets, TSO loses balancing resources",
                "Aggregators may prefer other TSO platforms (TenneT, 50Hertz)",
                "Regulatory: Who pays for flexibility - TSO or DSO?"
            ],
            "projects_affected": ["Systemmarkt", "Grid Boosters", "Redispatch optimization"]
        },
        
        "quantification": {
            "investment": "€20-50M platform development",
            "revenue_impact": "€50-200M/year flexibility market",
            "cost_savings": "€100-300M/year balancing cost reduction (potential)"
        }
    },
    
    # =========================================================================
    # E-MOBILITY / V2G
    # =========================================================================
    "e_mobility_v2g": {
        "category_name": "E-Mobility & Vehicle-to-Grid",
        
        "increase_means": """
📈 INCREASING V2G SIGNALS INDICATES:
- EVs transitioning from LOAD to FLEXIBILITY ASSET
- Bidirectional charging infrastructure growing
- ISO 15118 adoption accelerating
- Fleet operators exploring grid services

🔑 KEY TSO INSIGHT: Germany's 15M EVs by 2030 = 150GW potential flexibility!
- Even 10% participating = 15GW = more than all current pumped hydro
- Sub-second frequency response capability
- Mobile storage that moves to where it's needed
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- V2G business case not closing
- Battery degradation concerns
- Charging infrastructure not V2G-ready
- User reluctance to share vehicle control
        """,
        
        "short_term": {
            "revenue": """
- FCR PROVISION: EVs can provide frequency containment reserve
- PILOT REVENUES: €1-5M from V2G demonstration projects
            """,
            "costs": """
- LOAD GROWTH: Unmanaged EV charging increases peak demand
- GRID REINFORCEMENT: €100-500M for charging corridors (Autobahn)
- FORECASTING: EV load prediction systems €2-5M
            """,
            "operations": """
- Peak management: EV charging creates new demand peaks (6-8 PM)
- Forecasting: EV charging behavior highly variable
- Emergency protocols: Mass EV load shedding capability
            """,
            "actions": [
                "Map EV charging hotspots in grid area",
                "Partner with ChargePoint, IONITY for data sharing",
                "Pilot V2G with fleet operators (postal, logistics)",
                "Develop EV-specific demand forecasting models"
            ]
        },
        
        "long_term": {
            "revenue": """
- V2G SERVICES: €20-100M/year from aggregated EV flexibility
- GRID SERVICES: EVs provide voltage support, congestion relief
- AVOIDED STORAGE: EVs replace dedicated battery installations
            """,
            "costs": """
- INFRASTRUCTURE: Ultra-fast charging corridor investment
- INTEGRATION: V2G platform development and maintenance
            """,
            "strategy": """
- EVs as DISTRIBUTED STORAGE NETWORK
- Partner with automotive OEMs (VW, BMW, Mercedes)
- Shape regulatory framework for V2G value stacking
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "E-Mobility Grid Partner"
- Essential partner for automotive industry electrification
- Influence vehicle-grid interface standards
- Capture V2G value before DSOs or aggregators
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "Autobahn A1/A2/A3 ultra-fast charging corridors in Amprion area",
                "Heavy industry e-truck fleets (logistics, mining) = large flexible loads",
                "Ruhr region high EV adoption = early V2G market",
                "Grid Boosters can be supplemented by EV batteries"
            ],
            "risks": [
                "Unmanaged charging creates peak demand spikes (+20-30% evening peak)",
                "DSOs may claim V2G value for local services only",
                "Vehicle OEMs may develop own grid services platforms"
            ],
            "projects_affected": ["Grid expansion", "Load forecasting", "Flexibility markets"]
        },
        
        "quantification": {
            "investment": "€50-100M charging infrastructure coordination",
            "revenue_impact": "€20-100M/year V2G services",
            "cost_savings": "€50-150M avoided dedicated storage investment"
        }
    },
    
    # =========================================================================
    # GRID STABILITY
    # =========================================================================
    "grid_stability": {
        "category_name": "Grid Stability & System Services",
        
        "increase_means": """
📈 INCREASING STABILITY SIGNALS INDICATES:
- Growing concern about frequency/voltage stability
- New technologies emerging (grid-forming inverters, STATCOM)
- Loss of synchronous generation creating stability challenges
- More research into low-inertia grid operation

🔑 KEY TSO INSIGHT: This is EXISTENTIAL for TSOs!
- Stability is non-negotiable - blackouts destroy trust
- Every MW of synchronous generation retired = inertia lost
- Grid-forming inverters can replace rotating mass
- BUT: Transition period is dangerous
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- Stability challenges considered "solved"
- Focus shifting to other priorities
- Technology maturation (less R&D needed)
        """,
        
        "short_term": {
            "revenue": """
- SYSTEM SERVICES: Increasing prices for FCR, aFRR (good for cost recovery)
- STABILITY PREMIUMS: Grid-forming capability commands premium
            """,
            "costs": """
- STATCOM/SVC DEPLOYMENT: €20-50M per installation
- GRID BOOSTERS: €100-200M for battery-based stability
- MONITORING: Enhanced PMU network €5-10M
            """,
            "operations": """
- Real-time stability monitoring (inertia estimation)
- Dynamic stability limits (adjusting based on conditions)
- Faster control room response protocols
            """,
            "actions": [
                "Deploy STATCOM at critical substations (Gersteinwerk, Polsum)",
                "Expand Grid Booster program",
                "Implement real-time inertia monitoring",
                "Fast-track grid-forming inverter pilots"
            ]
        },
        
        "long_term": {
            "revenue": """
- STABILITY AS SERVICE: New revenue stream from grid-forming capability
- AVOIDED BLACKOUT COSTS: Each prevented blackout = €100M+ saved
            """,
            "costs": """
- MASSIVE INVESTMENT NEEDED: €500M-1B in stability equipment
- ONGOING MAINTENANCE: Specialized equipment requires expertise
            """,
            "strategy": """
- Stability becomes COMPETITIVE ADVANTAGE
- TSO that can operate low-inertia grid = technology leader
- Export stability solutions to other TSOs
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "Stability Excellence Leader"
- Master low-inertia grid operation before others
- Develop in-house grid-forming expertise
- Be the TSO others learn from
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "STATCOM deployments already underway (technology leader)",
                "Grid Boosters program = battery-based stability",
                "Nuclear/coal retirement = urgent need for solutions",
                "Ruhr region synchronous generation declining rapidly"
            ],
            "risks": [
                "Stability failure = career-ending event for TSO leadership",
                "Underinvestment could lead to major incident",
                "Technology evolution: Today's solution may be obsolete in 5 years"
            ],
            "projects_affected": ["STATCOM Gersteinwerk", "STATCOM Polsum", "Grid Boosters", "All major projects"]
        },
        
        "quantification": {
            "investment": "€500M-1B stability equipment",
            "revenue_impact": "Indirect - reputation and reliability",
            "cost_savings": "€100M+ per avoided major incident"
        }
    },
    
    # =========================================================================
    # HYDROGEN / P2G
    # =========================================================================
    "hydrogen_p2g": {
        "category_name": "Hydrogen & Power-to-Gas",
        
        "increase_means": """
📈 INCREASING HYDROGEN SIGNALS INDICATES:
- Large electrolyzers being planned/built
- Industrial hydrogen demand growing (steel, chemicals)
- Hydrogen infrastructure investment accelerating
- Green hydrogen economics improving

🔑 KEY TSO INSIGHT: Electrolyzers are MASSIVE NEW LOADS!
- Single electrolyzer hub = 100-500MW load
- Location critical: Put near renewable surplus
- BUT also: Flexibility resource (can curtail when grid stressed)
- Sector coupling: Electricity TSO + Gas TSO coordination
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- Hydrogen economics challenging
- Technology deployment delays
- Policy uncertainty
- Blue hydrogen competing with green
        """,
        
        "short_term": {
            "revenue": """
- CONNECTION FEES: Large electrolyzer connections = €10-50M each
- GRID SERVICES: Electrolyzers can provide demand response
            """,
            "costs": """
- GRID CONNECTION: New HV connections for hydrogen hubs
- REINFORCEMENT: Local grid upgrades for 100MW+ loads
- COORDINATION: hybridge partnership costs
            """,
            "operations": """
- Load forecasting: Include electrolyzer schedules
- Flexibility: Electrolyzer curtailment during scarcity
- Coordination: With gas TSOs for integrated operation
            """,
            "actions": [
                "Map planned electrolyzer projects >10MW in grid area",
                "Develop fast-track connection process for hydrogen",
                "Establish flexibility contracts with electrolyzer operators",
                "Coordinate with OGE/Thyssengas on hybridge expansion"
            ]
        },
        
        "long_term": {
            "revenue": """
- SECTOR COUPLING SERVICES: €50-200M/year from integrated planning
- FLEXIBILITY: Electrolyzers as controllable load pool
- HYDROGEN INFRASTRUCTURE: Possible future TSO role
            """,
            "costs": """
- GRID EXPANSION: €500M-2B for hydrogen hub connections
- SYSTEMS: Integrated electricity-hydrogen modeling
            """,
            "strategy": """
- Position for HYDROGEN GRID ROLE
- Electricity + hydrogen infrastructure integration
- European hydrogen backbone participation
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "Integrated Energy Infrastructure"
- Not just electricity TSO
- Partner in European hydrogen backbone
- Orchestrator of electricity-hydrogen optimization
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "hybridge project: Joint venture with OGE",
                "Ruhr region steel industry = massive hydrogen demand",
                "Potential electrolyzer hubs at coastal renewable zones",
                "Hydrogen storage = seasonal electricity flexibility"
            ],
            "risks": [
                "Electrolyzer location: If built in wrong place, need expensive grid",
                "Timing: Grid investment vs. hydrogen demand uncertainty",
                "Competition: Gas TSOs may capture hydrogen network value"
            ],
            "projects_affected": ["hybridge", "Grid expansion", "Offshore connections"]
        },
        
        "quantification": {
            "investment": "€500M-2B grid connections",
            "revenue_impact": "€50-200M/year sector coupling services",
            "cost_savings": "Flexibility value from curtailable load"
        }
    },
    
    # =========================================================================
    # ENERGY STORAGE
    # =========================================================================
    "energy_storage": {
        "category_name": "Energy Storage Systems",
        
        "increase_means": """
📈 INCREASING STORAGE SIGNALS INDICATES:
- Battery costs continuing to fall
- Grid-scale storage projects accelerating
- Long-duration storage technologies emerging
- Storage increasingly competitive with grid expansion

🔑 KEY TSO INSIGHT: Storage can REPLACE or DEFER grid investment!
- Grid Booster: Battery provides congestion relief
- Instead of building new line → install battery
- 4-hour battery can solve 90% of congestion events
- Long-duration (100+ hours) for seasonal balancing
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- Storage economics challenging
- Supply chain constraints
- Competing technologies (hydrogen, demand response)
- Regulatory barriers
        """,
        
        "short_term": {
            "revenue": """
- GRID BOOSTERS: Curative congestion management = savings
- ANCILLARY SERVICES: Storage competes, may lower prices
            """,
            "costs": """
- GRID BOOSTERS: €100-200M investment
- OPPORTUNITY: Deferred line investment = savings
            """,
            "operations": """
- Grid Booster dispatch optimization
- State-of-charge management
- Coordinated storage-network operation
            """,
            "actions": [
                "Expand Grid Booster deployment plan",
                "Evaluate long-duration storage for Korridor B support",
                "Develop storage-as-alternative-to-reinforcement criteria",
                "Pilot co-located storage at substations"
            ]
        },
        
        "long_term": {
            "revenue": """
- STORAGE AS GRID ASSET: TSO-owned storage for reliability
- DEFERRED CAPEX: €500M+ in avoided line investment
            """,
            "costs": """
- STORAGE INVESTMENT: €200-500M strategic storage portfolio
- REPLACEMENT: Battery degradation requires periodic replacement
            """,
            "strategy": """
- Storage as STRATEGIC GRID ASSET
- Portfolio of storage at critical locations
- Long-duration for seasonal flexibility
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "Flexible Grid Operator"
- Own strategic storage assets
- Offer storage as grid service
- Technology-neutral flexibility procurement
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "Grid Boosters already proving concept",
                "Storage at Korridor B landing points for offshore balancing",
                "Strategic storage at N-S corridor bottlenecks",
                "Co-investment with industrial customers"
            ],
            "risks": [
                "Battery technology evolution may strand assets",
                "Regulatory: TSO storage ownership constraints",
                "Market: Storage competing with redispatch revenue"
            ],
            "projects_affected": ["Grid Boosters", "Korridor B", "All congestion management"]
        },
        
        "quantification": {
            "investment": "€200-500M storage portfolio",
            "revenue_impact": "Indirect - reliability and flexibility",
            "cost_savings": "€500M+ deferred line investment"
        }
    },
    
    # =========================================================================
    # AI & GRID OPTIMIZATION
    # =========================================================================
    "ai_grid_optimization": {
        "category_name": "AI & Grid Optimization",
        
        "increase_means": """
📈 INCREASING AI SIGNALS INDICATES:
- AI/ML technologies maturing for grid applications
- More data available from smart grid sensors
- Agentic AI concepts emerging
- Control room automation accelerating

🔑 KEY TSO INSIGHT: AI enables AUTONOMOUS GRID OPERATION!
- Current: Human operators make decisions
- Future: AI recommends, humans approve
- Far future: AI decides, humans supervise
- Benefit: Faster response, better optimization
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- AI hype cycle correction
- Deployment challenges
- Trust/validation concerns
- Regulatory barriers to automation
        """,
        
        "short_term": {
            "revenue": """
- EFFICIENCY GAINS: Lower OPEX from optimization
- AUTO-TRADER: AI-driven renewable marketing
            """,
            "costs": """
- AI DEVELOPMENT: €10-30M for platforms
- DATA INFRASTRUCTURE: €5-10M for ML-ready systems
- TALENT: AI expertise is expensive
            """,
            "operations": """
- AI-assisted load forecasting
- Predictive maintenance alerts
- Automated report generation
            """,
            "actions": [
                "Deploy AI load forecasting (5-10% accuracy improvement target)",
                "Pilot agentic AI for control room decision support",
                "Build AI/ML team (10-20 specialists)",
                "Establish AI governance framework"
            ]
        },
        
        "long_term": {
            "revenue": """
- OPERATIONAL EXCELLENCE: Industry-leading efficiency
- AI SERVICES: Potential to license AI tools to other TSOs
            """,
            "costs": """
- ONGOING INVESTMENT: €10-20M/year in AI capabilities
- EFFICIENCY: 20-30% OPEX reduction potential
            """,
            "strategy": """
- AI as COMPETITIVE DIFFERENTIATOR
- Attract top talent to technology-forward TSO
- Lead European TSO digital transformation
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "AI-Native TSO"
- First TSO to achieve autonomous operations
- Technology partner of choice for vendors
- Talent magnet for AI/ML engineers
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "Auto-Trader already demonstrating AI value",
                "Control room complexity demands AI support",
                "Partner with German AI research institutes",
                "Attract talent from automotive/tech sector"
            ],
            "risks": [
                "AI failure in critical operation = disaster",
                "Regulatory approval for autonomous decisions",
                "Cybersecurity: AI systems as attack vectors"
            ],
            "projects_affected": ["Auto-Trader", "Control room modernization", "Predictive maintenance"]
        },
        
        "quantification": {
            "investment": "€50-100M AI capabilities",
            "revenue_impact": "Indirect - efficiency and reliability",
            "cost_savings": "20-30% OPEX reduction potential"
        }
    },
    
    # =========================================================================
    # CYBERSECURITY
    # =========================================================================
    "cybersecurity_ot": {
        "category_name": "Cybersecurity & OT Security",
        
        "increase_means": """
📈 INCREASING CYBERSECURITY SIGNALS INDICATES:
- Growing threat landscape (state actors, ransomware)
- New regulations (NIS2, KRITIS)
- Post-quantum cryptography urgency
- OT/IT convergence security challenges

🔑 KEY TSO INSIGHT: Cyber attack on TSO = NATIONAL SECURITY EVENT!
- Control room compromise → grid manipulation
- Ransomware → operational paralysis
- Data breach → market manipulation
- This is EXISTENTIAL risk
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- Threat level perceived as lower (dangerous!)
- Security fatigue
- Budget pressure reducing investment
        """,
        
        "short_term": {
            "revenue": """
- NONE DIRECT: Security is cost center
- INDIRECT: Trust = license to operate
            """,
            "costs": """
- NIS2 COMPLIANCE: €10-30M
- OT SECURITY: €20-50M infrastructure hardening
- SOC: 24/7 security operations €5-10M/year
            """,
            "operations": """
- Incident response readiness
- Regular penetration testing
- Security awareness training
            """,
            "actions": [
                "Complete NIS2 gap analysis and remediation",
                "Implement zero trust for OT networks",
                "Begin post-quantum cryptography migration planning",
                "Establish 24/7 OT-SOC capability"
            ]
        },
        
        "long_term": {
            "revenue": """
- NONE DIRECT: Security is cost center
- EXISTENCE: Without security, no business
            """,
            "costs": """
- ONGOING: €20-50M/year security investment
- POST-QUANTUM: €50-100M crypto migration
            """,
            "strategy": """
- Security as FOUNDATION (not feature)
- Build security into all projects from start
- Attract and retain security talent
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "Trusted Infrastructure Operator"
- Security excellence = government trust
- Partner of choice for critical infrastructure
- Resilience leader among European TSOs
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "Security leadership enhances government relations",
                "Brauweiler control room as showcase facility",
                "Partner with German security agencies (BSI)"
            ],
            "risks": [
                "Single successful attack = catastrophic reputational damage",
                "Talent shortage in OT security specialists",
                "Legacy systems hard to secure"
            ],
            "projects_affected": ["All digital projects", "Control room", "SCADA systems"]
        },
        
        "quantification": {
            "investment": "€100-200M over 5 years",
            "revenue_impact": "None direct",
            "cost_savings": "Avoided breach = €100M+ saved"
        }
    },
    
    # =========================================================================
    # GRID INFRASTRUCTURE
    # =========================================================================
    "grid_infrastructure": {
        "category_name": "Grid Infrastructure & Transmission",
        
        "increase_means": """
📈 INCREASING GRID INFRASTRUCTURE SIGNALS INDICATES:
- HVDC technology advancing
- More corridor projects being planned
- Construction methods improving (underground)
- Supply chain capacity growing

🔑 KEY TSO INSIGHT: This is CORE BUSINESS!
- Grid expansion = TSO revenue base
- €36.4B investment program at Amprion alone
- Every km of line = regulated asset base
- HVDC enables long-distance renewable transport
        """,
        
        "decrease_means": """
📉 DECREASING SIGNALS INDICATES:
- Permitting delays
- Supply chain constraints
- Public opposition
- Budget pressure
        """,
        
        "short_term": {
            "revenue": """
- REGULATED RETURN: Every €1B invested → €50-80M/year return
- GRID FEES: Higher asset base = higher allowed revenue
            """,
            "costs": """
- CAPEX: €3-5B/year investment pace
- PERMITTING: €50-100M/year for approvals process
            """,
            "operations": """
- Project execution at scale
- Supply chain management
- Stakeholder engagement
            """,
            "actions": [
                "Accelerate Korridor B and Rhein-Main Link execution",
                "Secure long-term HVDC converter supply",
                "Advance permitting for next wave of projects",
                "Community engagement for public acceptance"
            ]
        },
        
        "long_term": {
            "revenue": """
- €36.4B ASSET BASE: Regulated return on investment
- GRID FEE GROWTH: Proportional to asset base
            """,
            "costs": """
- MASSIVE CAPEX: €36.4B over 10-15 years
- MAINTENANCE: Growing O&M as asset base expands
            """,
            "strategy": """
- EXECUTION EXCELLENCE: Deliver on time, on budget
- INNOVATION: Underground HVDC, multi-terminal DC
- EUROPEAN LEADERSHIP: Set standards others follow
            """,
            "positioning": """
🎯 STRATEGIC POSITION: "Infrastructure Builder of Choice"
- Proven track record of mega-project delivery
- Technology leader in HVDC
- Model for European grid expansion
            """
        },
        
        "amprion_specific": {
            "opportunities": [
                "Korridor B: 4GW offshore wind to Ruhr region",
                "Rhein-Main Link: 8GW to Frankfurt industrial area",
                "A-Nord: Underground HVDC pioneering",
                "SuedLink coordination with TenneT"
            ],
            "risks": [
                "Permitting delays can add years",
                "Supply chain: HVDC converters are bottleneck",
                "Cost overruns affect regulated return"
            ],
            "projects_affected": ["Korridor B", "Rhein-Main Link", "A-Nord", "Ultranet", "SuedLink", "BalWin1/2", "NeuLink"]
        },
        
        "quantification": {
            "investment": "€36.4B investment program",
            "revenue_impact": "€1.8-2.9B/year regulated return (at 5-8%)",
            "cost_savings": "N/A - investment focus"
        }
    }
}


# =============================================================================
# BUSINESS IMPACT ANALYZER CLASS
# =============================================================================

class BusinessImpactAnalyzer:
    """
    Generates detailed business impact analysis for trends.
    
    Answers: "So what does this trend mean for TSO/Amprion business?"
    """
    
    def __init__(self):
        self.rules = BUSINESS_IMPACT_RULES
    
    def analyze(
        self,
        category: str,
        signal_count: int,
        previous_count: int = 0,
        trend_name: str = None,
        signals: List[Dict] = None
    ) -> BusinessImpact:
        """
        Generate comprehensive business impact analysis.
        
        Args:
            category: TSO category (e.g., 'biogas_biomethane')
            signal_count: Current number of signals
            previous_count: Previous period signal count (for trend)
            trend_name: Optional trend name
            signals: Optional list of signals for deeper analysis
        
        Returns:
            BusinessImpact with complete analysis
        """
        
        rules = self.rules.get(category, self._get_default_rules())
        
        # Determine direction
        if previous_count == 0:
            if signal_count > 10:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.EMERGING
        else:
            growth_rate = ((signal_count - previous_count) / previous_count) * 100
            if growth_rate > 20:
                direction = TrendDirection.INCREASING
            elif growth_rate < -20:
                direction = TrendDirection.DECREASING
            else:
                direction = TrendDirection.STABLE
        
        growth_rate = ((signal_count - previous_count) / max(previous_count, 1)) * 100
        
        # Build impact analysis
        short_term = rules.get("short_term", {})
        long_term = rules.get("long_term", {})
        amprion = rules.get("amprion_specific", {})
        quant = rules.get("quantification", {})
        
        # Select appropriate interpretation based on direction
        if direction == TrendDirection.INCREASING:
            exec_summary = rules.get("increase_means", "Trend increasing - assess implications")
        elif direction == TrendDirection.DECREASING:
            exec_summary = rules.get("decrease_means", "Trend decreasing - assess implications")
        else:
            exec_summary = f"Trend stable at {signal_count} signals"
        
        impact = BusinessImpact(
            category=category,
            trend_name=trend_name or rules.get("category_name", category),
            direction=direction,
            signal_count=signal_count,
            growth_rate=growth_rate,
            
            executive_summary=exec_summary,
            
            short_term_revenue=short_term.get("revenue", "Assess revenue implications"),
            short_term_costs=short_term.get("costs", "Assess cost implications"),
            short_term_operations=short_term.get("operations", "Assess operational changes"),
            short_term_actions=short_term.get("actions", ["Monitor and assess"]),
            
            long_term_revenue=long_term.get("revenue", "Assess long-term revenue"),
            long_term_costs=long_term.get("costs", "Assess long-term costs"),
            long_term_strategy=long_term.get("strategy", "Develop strategic response"),
            long_term_positioning=long_term.get("positioning", "Define positioning"),
            
            amprion_opportunities=amprion.get("opportunities", []),
            amprion_risks=amprion.get("risks", []),
            amprion_projects_affected=amprion.get("projects_affected", []),
            
            estimated_investment_eur=quant.get("investment"),
            estimated_revenue_impact=quant.get("revenue_impact"),
            estimated_cost_savings=quant.get("cost_savings"),
            
            action_urgency=self._determine_urgency(direction, category),
            decision_timeline=self._determine_timeline(direction, category)
        )
        
        return impact
    
    def _determine_urgency(self, direction: TrendDirection, category: str) -> str:
        """Determine action urgency."""
        critical_categories = ["grid_stability", "cybersecurity_ot", "grid_infrastructure"]
        high_categories = ["flexibility_vpp", "energy_storage", "hydrogen_p2g"]
        
        if category in critical_categories:
            return "Critical" if direction == TrendDirection.INCREASING else "High"
        elif category in high_categories:
            return "High" if direction == TrendDirection.INCREASING else "Medium"
        else:
            return "Medium"
    
    def _determine_timeline(self, direction: TrendDirection, category: str) -> str:
        """Determine decision timeline."""
        if direction == TrendDirection.INCREASING:
            return "3-6 months"
        elif direction == TrendDirection.DECREASING:
            return "6-12 months"
        else:
            return "12+ months"
    
    def _get_default_rules(self) -> Dict:
        """Default rules for unknown categories."""
        return {
            "category_name": "Unknown Category",
            "increase_means": "Increasing signal activity - assess implications",
            "decrease_means": "Decreasing signal activity - monitor situation",
            "short_term": {
                "revenue": "Assess short-term revenue implications",
                "costs": "Assess short-term cost implications",
                "operations": "Assess operational changes required",
                "actions": ["Monitor and assess", "Identify stakeholders", "Develop response plan"]
            },
            "long_term": {
                "revenue": "Assess long-term revenue strategy",
                "costs": "Assess long-term cost structure",
                "strategy": "Develop strategic response",
                "positioning": "Define market positioning"
            },
            "amprion_specific": {
                "opportunities": ["To be identified"],
                "risks": ["To be assessed"],
                "projects_affected": []
            }
        }
    
    def get_all_category_analyses(self) -> Dict[str, Dict]:
        """Get summary of all category impact rules."""
        summaries = {}
        for cat_id, rules in self.rules.items():
            summaries[cat_id] = {
                "name": rules.get("category_name", cat_id),
                "increase_summary": rules.get("increase_means", "")[:200] + "...",
                "decrease_summary": rules.get("decrease_means", "")[:200] + "...",
                "investment_estimate": rules.get("quantification", {}).get("investment"),
                "revenue_estimate": rules.get("quantification", {}).get("revenue_impact"),
            }
        return summaries
    
    def format_impact_report(self, impact: BusinessImpact) -> str:
        """Format impact as readable report."""
        report = f"""
{'=' * 70}
BUSINESS IMPACT ANALYSIS: {impact.trend_name}
{'=' * 70}

📊 TREND STATUS
   Direction: {impact.direction.value.upper()}
   Signal Count: {impact.signal_count}
   Growth Rate: {impact.growth_rate:+.1f}%
   Action Urgency: {impact.action_urgency}
   Decision Timeline: {impact.decision_timeline}

📝 EXECUTIVE SUMMARY
{impact.executive_summary}

{'=' * 70}
⏱️ SHORT-TERM IMPACT (0-2 YEARS)
{'=' * 70}

💰 Revenue Impact:
{impact.short_term_revenue}

💸 Cost Impact:
{impact.short_term_costs}

⚙️ Operational Changes:
{impact.short_term_operations}

📋 Immediate Actions:
{chr(10).join(f'   • {action}' for action in impact.short_term_actions)}

{'=' * 70}
🔮 LONG-TERM IMPACT (3-10 YEARS)
{'=' * 70}

💰 Revenue Strategy:
{impact.long_term_revenue}

💸 Cost Structure:
{impact.long_term_costs}

📈 Strategic Direction:
{impact.long_term_strategy}

🎯 Market Positioning:
{impact.long_term_positioning}

{'=' * 70}
🏭 AMPRION-SPECIFIC IMPLICATIONS
{'=' * 70}

✅ Opportunities:
{chr(10).join(f'   • {opp}' for opp in impact.amprion_opportunities)}

⚠️ Risks:
{chr(10).join(f'   • {risk}' for risk in impact.amprion_risks)}

🔗 Projects Affected:
{chr(10).join(f'   • {proj}' for proj in impact.amprion_projects_affected)}

{'=' * 70}
💶 FINANCIAL ESTIMATES
{'=' * 70}

   Investment Required: {impact.estimated_investment_eur or 'TBD'}
   Revenue Impact: {impact.estimated_revenue_impact or 'TBD'}
   Cost Savings: {impact.estimated_cost_savings or 'TBD'}

{'=' * 70}
"""
        return report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_business_analyzer() -> BusinessImpactAnalyzer:
    """Get singleton analyzer instance."""
    return BusinessImpactAnalyzer()


def analyze_trend_impact(
    category: str,
    signal_count: int,
    previous_count: int = 0
) -> BusinessImpact:
    """Quick analysis function."""
    analyzer = BusinessImpactAnalyzer()
    return analyzer.analyze(category, signal_count, previous_count)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    analyzer = BusinessImpactAnalyzer()
    
    # Test biogas scenario
    print("\n" + "=" * 70)
    print("TESTING: Biogas trend increasing")
    print("=" * 70)
    
    impact = analyzer.analyze(
        category="biogas_biomethane",
        signal_count=45,
        previous_count=20,
        trend_name="Biogas & Biomethane Growth"
    )
    
    print(analyzer.format_impact_report(impact))
    
    # Test VPP scenario
    print("\n" + "=" * 70)
    print("TESTING: VPP trend increasing")
    print("=" * 70)
    
    impact = analyzer.analyze(
        category="flexibility_vpp",
        signal_count=120,
        previous_count=60,
        trend_name="Virtual Power Plant Expansion"
    )
    
    print(analyzer.format_impact_report(impact))
