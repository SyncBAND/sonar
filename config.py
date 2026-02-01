"""
SONAR.AI / AGTS Configuration
==============================
Agentic Global Trend Scanner for TSOs (Transmission System Operators)

Based on AGTS Framework Pages 6-22:
- Multi-agent architecture (Scout, Architect, Strategist, Validator, Prioritizer)
- Comprehensive TSO taxonomies
- Amprion-specific categorization
- No-Regret prioritization framework
"""

import os
from datetime import datetime
from typing import Dict, List, Any

# =============================================================================
# SECTION 1: AGTS STRATEGIC DOMAINS (Page 8-9)
# =============================================================================

STRATEGIC_DOMAINS = {
    "system_operation": {
        "name": "System Operation & Digital Control Room",
        "description": "Ensuring stability and managing a 100% renewable, decentralized grid",
        "sub_categories": {
            "multi_agent_orchestration": {
                "name": "Multi-Agent Orchestration",
                "keywords": ["multi-agent", "autonomous", "decentralized ai", "voltage control", 
                            "frequency control", "agentic", "orchestration", "coordination"],
                "description": "Technologies enabling autonomous coordination between decentralized assets"
            },
            "probabilistic_forecasting": {
                "name": "Probabilistic Forecasting",
                "keywords": ["probabilistic", "ensemble", "forecasting", "prediction", "weather",
                            "demand forecast", "load forecast", "risk-based", "scenario"],
                "description": "AI models for risk-based, ensemble scenarios for redispatch"
            },
            "edge_cloud_continuum": {
                "name": "Edge-to-Cloud Continuum",
                "keywords": ["edge computing", "edge ai", "fog computing", "substation computing",
                            "real-time processing", "millisecond", "latency", "edge-to-cloud"],
                "description": "Computing power at substations for millisecond-level reaction times"
            }
        }
    },
    "digital_asset_lifecycle": {
        "name": "Digital Asset Life Cycle & Infrastructure",
        "description": "Maximizing utilization and life of physical assets via digital layers",
        "sub_categories": {
            "predictive_maintenance": {
                "name": "Predictive Maintenance & IoT",
                "keywords": ["predictive maintenance", "condition monitoring", "iot sensor",
                            "remaining useful life", "transformer monitoring", "acoustic sensor",
                            "thermal imaging", "asset health", "failure prediction"],
                "description": "Sensor-based signals and AI for estimating Remaining Useful Life"
            },
            "digital_twins": {
                "name": "Digital Twins (BIM & GIS)",
                "keywords": ["digital twin", "bim", "gis", "4d modeling", "simulation",
                            "virtual model", "asset modeling", "corridor modeling"],
                "description": "4D modeling of grid corridors to simulate construction and risks"
            },
            "sf6_free_circular": {
                "name": "SF6-Free & Circular IT",
                "keywords": ["sf6", "sulfur hexafluoride", "sf6-free", "circular economy",
                            "carbon passport", "sustainable hardware", "green it", "eco-design"],
                "description": "Sustainable hardware and digital tools for carbon tracking"
            }
        }
    },
    "energy_markets_flexibility": {
        "name": "Integrated Energy Markets & Flexibility",
        "description": "Transitioning from passive delivery to an active System of Systems",
        "sub_categories": {
            "tso_dso_interoperability": {
                "name": "TSO-DSO Interoperability",
                "keywords": ["tso-dso", "flexibility", "v2g", "vehicle-to-grid", "heat pump",
                            "demand response", "aggregation", "distributed flexibility",
                            "local flexibility market", "coordination platform"],
                "description": "Platforms for sharing flexibility data between grid levels"
            },
            "automated_market_clearing": {
                "name": "Automated Market Clearing",
                "keywords": ["automated trading", "market clearing", "systemmarkt", "balancing",
                            "real-time trading", "algorithmic trading", "market coupling",
                            "15-minute market", "intraday", "spot market"],
                "description": "High-speed trading algorithms for real-time grid balancing"
            },
            "sector_coupling_p2x": {
                "name": "Sector Coupling (P2X)",
                "keywords": ["power-to-gas", "p2x", "p2g", "hydrogen", "electrolyzer",
                            "sector coupling", "green hydrogen", "industrial hydrogen",
                            "hybridge", "power-to-heat", "power-to-liquid"],
                "description": "Digital interfaces for Power-to-Gas and industrial hydrogen hubs"
            }
        }
    },
    "cybersecurity_trust": {
        "name": "Cybersecurity & Trust Foundations",
        "description": "Protecting critical infrastructure in an era of offensive AI",
        "sub_categories": {
            "preemptive_cybersecurity": {
                "name": "Preemptive Cybersecurity",
                "keywords": ["cybersecurity", "scada security", "ot security", "threat detection",
                            "intrusion detection", "cyber attack", "offensive ai", "defense"],
                "description": "AI that anticipates attack patterns on SCADA systems"
            },
            "post_quantum_cryptography": {
                "name": "Post-Quantum Cryptography (PQC)",
                "keywords": ["post-quantum", "pqc", "quantum-safe", "quantum cryptography",
                            "encryption", "quantum computing", "lattice-based", "nist pqc"],
                "description": "Upgrading encryption against quantum computing threats"
            },
            "sovereign_cloud": {
                "name": "Sovereign Cloud & Geopatriation",
                "keywords": ["sovereign cloud", "geopatriation", "data sovereignty",
                            "gaia-x", "european cloud", "data localization", "compliance"],
                "description": "Strategic placement of data for regulatory compliance"
            }
        }
    }
}

# =============================================================================
# SECTION 2: ARCHITECTURAL LAYER TAXONOMY (Page 9-10)
# =============================================================================

ARCHITECTURAL_LAYERS = {
    "perception_actuator": {
        "name": "Perception & Actuator Layer",
        "description": "The Grid's 'Senses' - Sensors, robotics, and hardware interfaces",
        "sub_categories": {
            "robotics_inspection": {
                "name": "Robotics & Autonomous Inspection",
                "keywords": ["drone", "uav", "rov", "robot", "autonomous inspection",
                            "offshore maintenance", "line inspection", "crawler"]
            },
            "smart_materials": {
                "name": "Next-Gen Smart Materials",
                "keywords": ["self-healing", "smart insulation", "embedded sensor",
                            "graphene", "nanocomposite", "htls conductor", "superconductor"]
            },
            "non_invasive_monitoring": {
                "name": "Non-Invasive Monitoring",
                "keywords": ["satellite monitoring", "das", "distributed acoustic sensing",
                            "fiber optic sensing", "lidar", "sar", "thermal imaging"]
            }
        }
    },
    "communication_connectivity": {
        "name": "Communication & Connectivity Layer",
        "description": "The Grid's 'Nerves' - How data moves between field and control room",
        "sub_categories": {
            "laminar_coordination": {
                "name": "Laminar Coordination Frameworks",
                "keywords": ["layered communication", "hierarchical control", "coordination",
                            "tso-dso communication", "interoperability protocol"]
            },
            "industrial_5g_6g": {
                "name": "Industrial 5G & 6G",
                "keywords": ["5g", "6g", "private network", "low latency", "urllc",
                            "industrial iot", "lte-m", "narrowband iot"]
            },
            "software_defined_networking": {
                "name": "Software-Defined Networking (SDN)",
                "keywords": ["sdn", "software-defined", "network virtualization",
                            "nfv", "network slicing", "programmable network"]
            }
        }
    },
    "data_intelligence": {
        "name": "Data & Intelligence Layer",
        "description": "The Grid's 'Brain' - Data processing, storage, and AI models",
        "sub_categories": {
            "federated_learning": {
                "name": "Laminar AI / Federated Learning",
                "keywords": ["federated learning", "privacy-preserving", "distributed ml",
                            "edge ml", "on-device learning", "differential privacy"]
            },
            "knowledge_graphs": {
                "name": "Knowledge Graphs for Asset Management",
                "keywords": ["knowledge graph", "semantic", "ontology", "asset graph",
                            "cim", "common information model", "linked data"]
            },
            "synthetic_data": {
                "name": "Synthesized Data Generation",
                "keywords": ["synthetic data", "data augmentation", "gan", "simulation",
                            "failure simulation", "scenario generation"]
            }
        }
    },
    "service_orchestration": {
        "name": "Service & Orchestration Layer",
        "description": "The Grid's 'Economy' - Markets, customers, and TSO interactions",
        "sub_categories": {
            "blockchain_verification": {
                "name": "Blockchain-based Verification",
                "keywords": ["blockchain", "distributed ledger", "smart contract",
                            "guarantee of origin", "energy certificate", "tokenization"]
            },
            "virtual_power_plant": {
                "name": "Virtual Power Plant (VPP) Aggregation",
                "keywords": ["vpp", "virtual power plant", "aggregation", "der aggregation",
                            "flexibility aggregation", "demand aggregation", "prosumer"]
            },
            "cross_border_coupling": {
                "name": "Cross-Border Market Coupling",
                "keywords": ["market coupling", "cross-border", "fbmc", "flow-based",
                            "capacity calculation", "border trading", "interconnector"]
            }
        }
    }
}

# =============================================================================
# SECTION 3: BUSINESS AREA TAXONOMY - AMPRION CORPORATE (Page 10-12)
# =============================================================================

AMPRION_BUSINESS_AREAS = {
    "corporate_strategy": {
        "code": "CS",
        "name": "Corporate Strategy & Development",
        "description": "Long-term orientation, board support, and competitive positioning",
        "sub_categories": {
            "customer_intelligence": "Advanced monitoring of prosumer behavior and industrial demand",
            "market_intelligence": "Analysis of global energy market evolution and competitor benchmarking",
            "regulatory_management": "Tracking BNetzA determinations and European legal frameworks",
            "ma_partnerships": "Strategic startup investments and TSO cooperative ventures"
        }
    },
    "system_operation": {
        "code": "SO",
        "name": "System Operation",
        "description": "Real-time grid stability and control room excellence",
        "sub_categories": {
            "system_control_dispatch": "Automatic frequency restoration and voltage control",
            "congestion_management": "Redispatch 2.0 and curative grid measures",
            "system_services": "Procurement of balancing power and stability services"
        }
    },
    "asset_management": {
        "code": "AM",
        "name": "Asset Management & Technology",
        "description": "Technical lifecycle of physical steel and copper assets",
        "sub_categories": {
            "technology_innovation": "TRL-focused trends in HVDC converters, transformers, switchgear",
            "asset_strategy": "Predictive maintenance models and aging infrastructure management",
            "protection_control": "Digital substation technology and secondary systems"
        }
    },
    "grid_projects": {
        "code": "GP",
        "name": "Grid Projects & Expansion",
        "description": "Large-scale CAPEX and project delivery",
        "sub_categories": {
            "project_management_onshore": "Planning and construction for overhead/underground cables",
            "offshore_connections": "Submarine cabling and offshore converter platforms",
            "permitting_public_affairs": "Digital citizen participation and environmental approval"
        }
    },
    "market_economy": {
        "code": "M",
        "name": "Market & Energy Economy",
        "description": "Trading interfaces and commercial grid access",
        "sub_categories": {
            "grid_access_charges": "Management of grid fees and connection agreements",
            "european_market_coupling": "Flow-based capacity calculation and international trading",
            "balancing_group_management": "Settlement processes and financial balancing"
        }
    },
    "corporate_functions": {
        "code": "ITD",
        "name": "Corporate Functions (IT/Digital)",
        "description": "Internal infrastructure and resource management",
        "sub_categories": {
            "it_digital_media": "Cyber-defense, data engineering, enterprise software",
            "procurement_supply_chain": "Sourcing strategy and vendor risk management",
            "human_resources": "Digital talent acquisition and Workplace of the Future"
        }
    }
}


# =============================================================================
# SECTION 5: TSO-SPECIFIC CATEGORIES (Enhanced with Enlit + Biogas, VPP, etc.)
# =============================================================================

TSO_CATEGORIES = {
    # CORE GRID TECHNOLOGY
    "grid_infrastructure": {
        "name": "Grid Infrastructure & Transmission",
        "keywords": [
            "hvdc", "high voltage direct current", "transmission line", "power line",
            "overhead line", "underground cable", "converter station", "substation",
            "transformer", "switchgear", "circuit breaker", "busbar", "insulator",
            "grid expansion", "grid reinforcement", "interconnector", "corridor",
            "korridor b", "rhein-main link", "a-nord", "ultranet", "suedlink",
            "balwin", "neulink", "supergrid", "meshed grid", "ac/dc converter",
            "gis", "gas insulated switchgear", "ais", "air insulated switchgear",
            "power transformer", "phase shifting transformer", "pst"
        ],
        "amprion_projects": ["Korridor B", "Rhein-Main Link", "A-Nord", "Ultranet", "BalWin1", "BalWin2", "NeuLink"],
    },
    
    "grid_stability": {
        "name": "Grid Stability & System Services",
        "keywords": [
            "frequency control", "voltage control", "reactive power", "statcom",
            "grid-forming", "grid-following", "inertia", "synthetic inertia",
            "frequency response", "primary reserve", "secondary reserve",
            "fcr", "afrr", "mfrr", "balancing", "ancillary services",
            "redispatch", "congestion", "grid booster", "phase shifter",
            "fault ride through", "black start", "islanding", "resynchronization"
        ],
        "amprion_projects": ["Grid Boosters", "STATCOM Gersteinwerk", "STATCOM Polsum"]
    },
    
    # ENERGY STORAGE & FLEXIBILITY
    "energy_storage": {
        "name": "Energy Storage Systems",
        "keywords": [
            "battery", "bess", "battery energy storage", "lithium", "sodium ion",
            "solid state battery", "flow battery", "vanadium", "redox flow",
            "pumped hydro", "compressed air", "caes", "flywheel", "supercapacitor",
            "grid-scale storage", "utility-scale battery", "mwh", "gwh",
            "long duration storage", "seasonal storage", "energy shifting",
            "ldes", "long duration energy storage", "gravity storage",
            "thermal storage", "molten salt", "liquid air", "cryogenic"
        ],
        "strategic_impact": "Enables curative grid management and renewable integration",
    },
    
    "flexibility_vpp": {
        "name": "Flexibility & Virtual Power Plants",
        "keywords": [
            "vpp", "virtual power plant", "aggregation", "demand response",
            "demand side management", "dsm", "load shifting", "peak shaving",
            "flexibility market", "flexibility trading", "der aggregation",
            "distributed energy resources", "prosumer", "behind the meter",
            "residential flexibility", "industrial flexibility", "heat pump",
            "thermal storage", "ice storage", "smart thermostat"
        ],
        "strategic_impact": "Creates distributed flexibility resources for grid balancing"
    },
    
    # E-MOBILITY & V2G
    "e_mobility_v2g": {
        "name": "E-Mobility & Vehicle-to-Grid",
        "keywords": [
            "ev", "electric vehicle", "ev charging", "fast charging", "ultra-fast",
            "v2g", "vehicle-to-grid", "v2h", "vehicle-to-home", "v2x",
            "bidirectional charging", "smart charging", "managed charging",
            "charging infrastructure", "charge point", "evse", "ccs", "chademo",
            "iso 15118", "plug and charge", "fleet management", "e-bus", "e-truck",
            "charging station", "chargepoint", "electromobility", "emobility",
            "electric fleet", "depot charging", "opportunity charging",
            "megawatt charging", "mcs", "pantograph charging"
        ],
        "strategic_impact": "EVs as distributed storage providing frequency response",
    },
    
    # RENEWABLES INTEGRATION
    "renewables_integration": {
        "name": "Renewables Integration",
        "keywords": [
            "solar", "pv", "photovoltaic", "wind", "offshore wind", "onshore wind",
            "wind turbine", "solar inverter", "renewable", "curtailment",
            "renewable integration", "variability", "intermittency", "ramp rate",
            "capacity factor", "floating offshore", "agrivoltaics", "bifacial",
            "wind farm", "solar farm", "renewable forecast", "weather forecast",
            "perovskite", "tandem solar", "wind power", "solar power",
            "renewable energy source", "res", "renewable portfolio standard"
        ],
        "strategic_impact": "Core driver of grid transformation",
    },
    
    # SECTOR COUPLING - HYDROGEN
    "hydrogen_p2g": {
        "name": "Hydrogen & Power-to-Gas",
        "keywords": [
            "hydrogen", "h2", "green hydrogen", "electrolyzer", "electrolysis",
            "pem electrolyzer", "alkaline electrolyzer", "soec", "fuel cell",
            "hydrogen storage", "hydrogen transport", "hydrogen pipeline",
            "power-to-gas", "p2g", "methanation", "synthetic methane",
            "hydrogen hub", "hydrogen valley", "hybridge", "get h2",
            "hydrogen backbone", "hydrogen economy", "blue hydrogen",
            "grey hydrogen", "pink hydrogen", "turquoise hydrogen",
            "hydrogen refueling", "hrs", "fcev", "fuel cell vehicle"
        ],
        "strategic_impact": "Sector coupling for long-duration storage and industrial decarbonization",
    },
    
    # SECTOR COUPLING - BIOGAS (NEW)
    "biogas_biomethane": {
        "name": "Biogas & Biomethane",
        "keywords": [
            "biogas", "biomethane", "anaerobic digestion", "biogas plant",
            "biogas upgrading", "gas grid injection", "renewable gas",
            "bio-cng", "bio-lng", "agricultural biogas", "waste-to-energy",
            "landfill gas", "sewage gas", "digestate", "feedstock",
            "biogas wheeling", "green gas certificate", "gas quality"
        ],
        "strategic_impact": "Renewable gas can be wheeled through existing gas infrastructure, providing flexibility"
    },
    
    # AI & DIGITALIZATION
    "ai_grid_optimization": {
        "name": "AI & Grid Optimization",
        "keywords": [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "reinforcement learning", "optimization algorithm",
            "grid optimization", "load forecasting", "price forecasting",
            "predictive analytics", "anomaly detection", "pattern recognition",
            "nlp", "computer vision", "generative ai", "llm", "foundation model",
            "agentic ai", "multi-agent", "autonomous system",
            "digital twin", "data analytics", "big data", "iot platform",
            "edge ai", "mlops", "aiops", "automated decision"
        ],
        "strategic_impact": "Enables autonomous grid operation and predictive management",
    },
    
    "digital_twin_simulation": {
        "name": "Digital Twins & Simulation",
        "keywords": [
            "digital twin", "simulation", "modeling", "grid model",
            "power system simulation", "real-time simulation", "hil",
            "hardware in loop", "co-simulation", "scenario analysis",
            "contingency analysis", "state estimation", "psse", "powerfactory",
            "etap", "pscad", "emtp", "probabilistic simulation"
        ],
        "strategic_impact": "Enables what-if analysis and predictive operations"
    },
    
    # CYBERSECURITY
    "cybersecurity_ot": {
        "name": "Cybersecurity & OT Security",
        "keywords": [
            "cybersecurity", "ot security", "operational technology",
            "scada security", "ics security", "critical infrastructure",
            "cip", "nerc cip", "nis2", "kritis", "zero trust",
            "network segmentation", "threat detection", "siem", "soc",
            "incident response", "vulnerability", "penetration testing",
            "post-quantum", "pqc", "quantum-safe",
            "cyber attack", "ransomware", "malware", "phishing",
            "security operations center", "threat intelligence", "cyber resilience"
        ],
        "strategic_impact": "Protecting critical grid infrastructure",
    },
    
    # MARKET & TRADING
    "energy_trading": {
        "name": "Energy Trading & Markets",
        "keywords": [
            "energy trading", "power trading", "electricity market",
            "spot market", "futures", "forward", "day-ahead", "intraday",
            "balancing market", "capacity market", "flexibility market",
            "market coupling", "price formation", "merit order", "marginal price",
            "epex spot", "eex", "nordpool", "auction", "continuous trading"
        ],
        "strategic_impact": "Market design evolution affects TSO operations"
    },
    
    # REGULATORY & POLICY
    "regulatory_policy": {
        "name": "Regulatory & Policy",
        "keywords": [
            "regulation", "policy", "bnetzA", "bundesnetzagentur", "acer",
            "entso-e", "network code", "eu regulation", "clean energy package",
            "red iii", "fit for 55", "energy law", "grid code", "connection code",
            "aregv", "incentive regulation", "tariff", "grid fee", "nep",
            "network development plan", "tyndp", "market design", "capacity mechanism",
            "network tariff", "congestion revenue", "cross-border", "unbundling"
        ],
        "strategic_impact": "Regulatory changes drive investment and operations",
    },
    
    # GRID EDGE TECHNOLOGIES (Enlit Category)
    "grid_edge": {
        "name": "Grid Edge Technologies",
        "keywords": [
            "grid edge", "edge computing", "edge device", "smart inverter",
            "building energy management", "bems", "home energy management", "hems",
            "behind the meter", "btm", "prosumer", "self-consumption",
            "peer-to-peer", "p2p energy", "local energy market", "microgrid",
            "nanogrid", "islanding", "grid-interactive building", "geb",
            "distributed energy resource management", "derms", "flexibility aggregation"
        ],
        "strategic_impact": "Grid edge devices create distributed flexibility and control challenges",
    },
    
    # SMART METERING & AMI (Enlit Category)
    "smart_metering": {
        "name": "Smart Metering & AMI",
        "keywords": [
            "smart meter", "ami", "advanced metering infrastructure", "mdm",
            "meter data management", "dlms", "cosem", "smart metering", "amr",
            "automatic meter reading", "interval data", "load profile",
            "meter roll-out", "smart meter gateway", "15-minute settlement",
            "sub-metering", "energy disaggregation", "nilm"
        ],
        "strategic_impact": "Metering data enables grid visibility and settlement",
    },
    
    # ENERGY COMMUNITIES (Enlit Category)
    "energy_communities": {
        "name": "Energy Communities",
        "keywords": [
            "energy community", "citizen energy", "cooperative", "energy cooperative",
            "community solar", "community wind", "collective self-consumption",
            "renewable energy community", "rec", "citizen energy community", "cec",
            "local energy community", "energy sharing", "community battery",
            "neighborhood energy", "energy democracy", "democratisation"
        ],
        "strategic_impact": "Energy communities create local flexibility but reduce grid visibility",
    },
    
    # POWER GENERATION (Enlit Category)
    "power_generation": {
        "name": "Power Generation",
        "keywords": [
            "power plant", "generation", "thermal power", "combined cycle",
            "ccgt", "gas turbine", "chp", "cogeneration", "combined heat power",
            "peak plant", "peaker", "baseload", "mid-merit", "capacity factor",
            "plant efficiency", "heat rate", "carbon capture", "ccs", "ccus",
            "retrofit", "repowering", "decommissioning", "coal phase-out"
        ],
        "strategic_impact": "Generation mix evolution affects system stability and balancing",
    },
    
    # GAS INFRASTRUCTURE (Enlit Category - for sector coupling)
    "gas_infrastructure": {
        "name": "Gas Infrastructure & Sector Coupling",
        "keywords": [
            "gas grid", "gas network", "lng terminal", "gas storage",
            "gas pipeline", "gas interconnector", "gas transmission",
            "gas distribution", "gas quality", "wobbe index", "calorific value",
            "gas blending", "hydrogen blending", "h2 ready", "gas tso",
            "oge", "thyssengas", "gasunie", "grtgaz", "gas entry point"
        ],
        "strategic_impact": "Gas infrastructure enables sector coupling and hydrogen transport",
    },
    
    # FINANCE & INVESTMENT (Enlit Category)
    "finance_investment": {
        "name": "Finance & Green Investment",
        "keywords": [
            "green bond", "green finance", "sustainable finance", "esg",
            "taxonomy", "eu taxonomy", "green loan", "project finance",
            "infrastructure investment", "capex", "opex", "rate base",
            "wacc", "equity return", "regulatory asset base", "rab",
            "investment recovery", "stranded asset", "climate finance"
        ],
        "strategic_impact": "Green finance enables grid expansion investment",
    },
    
    # DECARBONISATION (Enlit Category)
    "decarbonisation": {
        "name": "Decarbonisation & Net Zero",
        "keywords": [
            "decarbonisation", "decarbonization", "net zero", "carbon neutral",
            "carbon footprint", "ghg emissions", "scope 1", "scope 2", "scope 3",
            "carbon intensity", "emission factor", "carbon budget", "sbti",
            "science based targets", "paris agreement", "climate target",
            "1.5 degree", "carbon pricing", "ets", "emissions trading"
        ],
        "strategic_impact": "Decarbonisation drives grid transformation and renewable integration",
    },
    
    # DISTRIBUTED GENERATION (Enlit Category)
    "distributed_generation": {
        "name": "Distributed Generation",
        "keywords": [
            "distributed generation", "dg", "embedded generation", "dispersed generation",
            "rooftop solar", "rooftop pv", "commercial solar", "industrial solar",
            "small wind", "micro-chp", "fuel cell", "distributed solar",
            "prosumer", "net metering", "feed-in tariff", "fit", "self-generation"
        ],
        "strategic_impact": "DG changes power flows and requires TSO-DSO coordination",
    },
    
    # HEAT & COOLING (Sector Coupling)
    "heat_district_energy": {
        "name": "Heat & District Energy",
        "keywords": [
            "district heating", "district cooling", "heat network", "heat pump",
            "industrial heat", "process heat", "waste heat", "chp",
            "combined heat power", "cogeneration", "power-to-heat", "p2h",
            "thermal energy storage", "tes", "seasonal thermal storage",
            "heat exchanger", "absorption chiller", "geothermal"
        ],
        "strategic_impact": "Sector coupling increases electricity demand flexibility"
    },
    
    # OFFSHORE & MARITIME
    "offshore_systems": {
        "name": "Offshore Grid Systems",
        "keywords": [
            "offshore", "offshore wind", "offshore platform", "offshore substation",
            "submarine cable", "subsea cable", "hvdc offshore", "offshore converter",
            "floating platform", "offshore hub", "energy island", "north sea",
            "baltic sea", "offshore grid", "meshed offshore", "offshore maintenance"
        ],
        "amprion_projects": ["BalWin1", "BalWin2", "NeuLink"],
        "strategic_impact": "Critical for integrating 70 GW offshore wind by 2045"
    }
}

# =============================================================================
# SECTION 5: MATURITY FRAMEWORKS (Page 16-17)
# =============================================================================

MATURITY_TYPES = {
    "TRL": {
        "name": "Technology Readiness Level",
        "description": "Is the hardware/software physically proven?",
        "levels": {
            1: "Basic research / principles observed",
            2: "Technology concept formulated",
            3: "Experimental proof of concept",
            4: "Technology validated in lab",
            5: "Technology validated in relevant environment",
            6: "Technology demonstrated in relevant environment",
            7: "System prototype demonstration in operational environment",
            8: "System complete and qualified",
            9: "Proven in operational environment"
        }
    },
    "MRL": {
        "name": "Market Readiness Level",
        "description": "Is there a commercial ecosystem, pricing model, and demand?",
        "levels": {
            1: "Market potential identified",
            2: "Market research initiated",
            3: "Initial market validation",
            4: "Early adopter pilots",
            5: "Commercial pilot deployments",
            6: "Initial market traction",
            7: "Scaling in target markets",
            8: "Broad commercial availability",
            9: "Market saturation / mainstream adoption"
        }
    },
    "SRL": {
        "name": "Societal Readiness Level",
        "description": "Is there broad public acceptance and behavioral change?",
        "levels": {
            1: "Initial concept only",
            2: "Stakeholder identification",
            3: "First engagement with stakeholders",
            4: "Stakeholder concerns mapped",
            5: "Pilot engagement activities",
            6: "Broader public awareness",
            7: "Early behavioral change observed",
            8: "Established consumer habits",
            9: "Societal norm / widespread acceptance"
        }
    },
    "RRL": {
        "name": "Regulatory Readiness Level",
        "description": "Is it legally permitted or incentivized by BNetzA/EU?",
        "levels": {
            1: "No regulatory framework",
            2: "Initial policy discussions",
            3: "Early whitepaper / consultation",
            4: "Draft regulation proposed",
            5: "Regulatory sandbox / pilot",
            6: "Initial regulation enacted",
            7: "Full regulatory framework",
            8: "Regulatory incentives in place",
            9: "Codified in energy law / mandatory"
        }
    }
}

# =============================================================================
# SECTION 6: AMPRION KEY TASKS & STRATEGIC FIELDS (Page 17)
# =============================================================================

AMPRION_KEY_TASKS = {
    "grid_expansion": {
        "name": "Grid Expansion & Re-construction",
        "description": "Physical building and cabling projects",
        "keywords": ["grid expansion", "construction", "cabling", "corridor", "hvdc", "new line"]
    },
    "system_security": {
        "name": "System Security",
        "description": "Technologies ensuring frequency and voltage stability",
        "keywords": ["stability", "security", "frequency", "voltage", "inertia", "resilience"]
    },
    "decarbonization": {
        "name": "Decarbonization",
        "description": "Direct CO2 reduction or renewable integration",
        "keywords": ["carbon", "co2", "emission", "renewable", "green", "climate", "net-zero"]
    },
    "european_trading": {
        "name": "European Electricity Trading",
        "description": "Market-clearing and cross-border algorithms",
        "keywords": ["trading", "market", "cross-border", "coupling", "european", "exchange"]
    }
}


# =============================================================================
# SECTION 7: DATA SOURCES (Enhanced with Energy Industry Sources)
# =============================================================================

DATA_SOURCES = {
    # RESEARCH & ACADEMIC
    "arxiv": {
        "name": "arXiv",
        "type": "research",
        "url": "https://arxiv.org/",
        "categories": ["eess.SY", "cs.AI", "cs.LG", "physics.app-ph"],
        "quality_score": 0.90,
        "description": "Preprint server for electrical engineering and systems"
    },
    
    # PATENTS
    "google_patents": {
        "name": "Google Patents",
        "type": "patent",
        "url": "https://patents.google.com/",
        "quality_score": 0.95,
        "description": "Global patent database"
    },
    "wipo": {
        "name": "WIPO",
        "type": "patent",
        "url": "https://patentscope.wipo.int/",
        "quality_score": 0.95,
        "description": "World Intellectual Property Organization"
    },
    
    # ENERGY NEWS (Primary)
    "utility_dive": {
        "name": "Utility Dive",
        "type": "news",
        "url": "https://www.utilitydive.com/",
        "rss": "https://www.utilitydive.com/feeds/news/",
        "quality_score": 0.85
    },
    "pv_magazine": {
        "name": "PV Magazine",
        "type": "news",
        "url": "https://www.pv-magazine.com/",
        "rss": "https://www.pv-magazine.com/feed/",
        "quality_score": 0.85
    },
    "energy_storage_news": {
        "name": "Energy Storage News",
        "type": "news",
        "url": "https://www.energy-storage.news/",
        "rss": "https://www.energy-storage.news/feed/",
        "quality_score": 0.85
    },
    "recharge_news": {
        "name": "Recharge News",
        "type": "news",
        "url": "https://www.rechargenews.com/",
        "quality_score": 0.85
    },
    "greentech_media": {
        "name": "Greentech Media",
        "type": "news",
        "url": "https://www.greentechmedia.com/",
        "quality_score": 0.80
    },
    
    # EUROPEAN ENERGY NEWS
    "euractiv_energy": {
        "name": "EURACTIV Energy",
        "type": "news",
        "url": "https://www.euractiv.com/sections/energy/",
        "quality_score": 0.85
    },
    "clean_energy_wire": {
        "name": "Clean Energy Wire (CLEW)",
        "type": "news",
        "url": "https://www.cleanenergywire.org/",
        "quality_score": 0.85,
        "description": "German energy transition news"
    },
    "montelnews": {
        "name": "Montel News",
        "type": "news",
        "url": "https://www.montelnews.com/",
        "quality_score": 0.85
    },
    
    # INDUSTRY EVENTS & CONFERENCES
    "enlit_europe": {
        "name": "Enlit Europe",
        "type": "conference",
        "url": "https://www.enlit-europe.com/",
        "quality_score": 0.90,
        "description": "Premier European utility event"
    },
    "e_world": {
        "name": "E-world Energy & Water",
        "type": "conference",
        "url": "https://www.e-world-essen.com/",
        "quality_score": 0.90,
        "description": "Europe's leading energy trade fair"
    },
    "european_utility_week": {
        "name": "European Utility Week",
        "type": "conference",
        "url": "https://www.enlit-europe.com/",
        "quality_score": 0.85
    },
    "windeurope": {
        "name": "WindEurope",
        "type": "conference",
        "url": "https://windeurope.org/",
        "quality_score": 0.90
    },
    "solar_power_europe": {
        "name": "SolarPower Europe",
        "type": "conference",
        "url": "https://www.solarpowereurope.org/",
        "quality_score": 0.90
    },
    
    # REGULATORY & POLICY
    "bnetzA": {
        "name": "Bundesnetzagentur",
        "type": "regulatory",
        "url": "https://www.bundesnetzagentur.de/",
        "quality_score": 0.95,
        "description": "German Federal Network Agency"
    },
    "acer": {
        "name": "ACER",
        "type": "regulatory",
        "url": "https://www.acer.europa.eu/",
        "quality_score": 0.95,
        "description": "EU Agency for Cooperation of Energy Regulators"
    },
    "entsoe": {
        "name": "ENTSO-E",
        "type": "regulatory",
        "url": "https://www.entsoe.eu/",
        "quality_score": 0.95,
        "description": "European Network of TSOs for Electricity"
    },
    "eu_commission_energy": {
        "name": "EU Commission Energy",
        "type": "regulatory",
        "url": "https://energy.ec.europa.eu/",
        "quality_score": 0.95
    },
    
    # TSO PUBLICATIONS
    "amprion": {
        "name": "Amprion",
        "type": "tso",
        "url": "https://www.amprion.net/",
        "quality_score": 0.95
    },
    "tennet": {
        "name": "TenneT",
        "type": "tso",
        "url": "https://www.tennet.eu/",
        "quality_score": 0.90
    },
    "50hertz": {
        "name": "50Hertz",
        "type": "tso",
        "url": "https://www.50hertz.com/",
        "quality_score": 0.90
    },
    "transnet_bw": {
        "name": "TransnetBW",
        "type": "tso",
        "url": "https://www.transnetbw.de/",
        "quality_score": 0.90
    },
    "national_grid": {
        "name": "National Grid ESO",
        "type": "tso",
        "url": "https://www.nationalgrideso.com/",
        "quality_score": 0.90
    },
    "rte_france": {
        "name": "RTE France",
        "type": "tso",
        "url": "https://www.rte-france.com/",
        "quality_score": 0.90
    },
    "elia": {
        "name": "Elia",
        "type": "tso",
        "url": "https://www.elia.be/",
        "quality_score": 0.90
    },
    
    # RESEARCH ORGANIZATIONS
    "iea": {
        "name": "International Energy Agency",
        "type": "research_org",
        "url": "https://www.iea.org/",
        "quality_score": 0.95
    },
    "irena": {
        "name": "IRENA",
        "type": "research_org",
        "url": "https://www.irena.org/",
        "quality_score": 0.95
    },
    "fraunhofer_ise": {
        "name": "Fraunhofer ISE",
        "type": "research_org",
        "url": "https://www.ise.fraunhofer.de/",
        "quality_score": 0.95
    },
    "dena": {
        "name": "Deutsche Energie-Agentur (dena)",
        "type": "research_org",
        "url": "https://www.dena.de/",
        "quality_score": 0.90
    },
    
    # STARTUPS & INNOVATION
    "crunchbase": {
        "name": "Crunchbase",
        "type": "startup",
        "url": "https://www.crunchbase.com/",
        "quality_score": 0.70
    },
    "energy_startups_eu": {
        "name": "EU-Startups Energy",
        "type": "startup",
        "url": "https://www.eu-startups.com/category/energy/",
        "quality_score": 0.70
    }
}


# =============================================================================
# SECTION 10: STAKEHOLDER VIEW CONFIGURATIONS (Page 15-16)
# =============================================================================

STAKEHOLDER_VIEWS = {
    "executive_board": {
        "name": "Executive Board",
        "priority_fields": [
            "priority_score", "strategic_nature", "regulatory_sensitivity", 
            "so_what_summary", "time_to_impact"
        ],
        "default_sort": "priority_score",
        "default_filter": {"time_to_impact": ["<1 year", "1-3 years"]}
    },
    "it_architecture": {
        "name": "IT Architecture",
        "priority_fields": [
            "architectural_layer", "sovereignty_score", "maturity_score", 
            "lifecycle_status", "maturity_type"
        ],
        "default_sort": "maturity_score",
        "default_filter": {"maturity_type": ["TRL"]}
    },
    "grid_planning": {
        "name": "Grid Planning (AM/SO)",
        "priority_fields": [
            "key_amprion_task", "linked_projects", "time_to_impact",
            "customer_impact", "market_impact"
        ],
        "default_sort": "time_to_impact",
        "default_filter": {"key_amprion_task": ["Grid Expansion", "System Security"]}
    },
    "sustainability": {
        "name": "Sustainability Team",
        "priority_fields": [
            "key_amprion_task", "description_short", "customer_impact"
        ],
        "default_sort": "priority_score",
        "default_filter": {"key_amprion_task": ["Decarbonization"]}
    }
}

# =============================================================================
# SECTION 11: API CONFIGURATION
# =============================================================================

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sonar_ai.db")

# API Keys (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Server
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# =============================================================================
# SECTION 12: HELPER FUNCTIONS
# =============================================================================

# Print config summary on import
if __name__ == "__main__":
    print("=" * 60)
    print("SONAR.AI / AGTS Configuration Summary")
    print("=" * 60)
    print(f"\nStrategic Domains: {len(STRATEGIC_DOMAINS)}")
    print(f"Architectural Layers: {len(ARCHITECTURAL_LAYERS)}")
    print(f"Business Areas: {len(AMPRION_BUSINESS_AREAS)}")
    print(f"TSO Categories: {len(TSO_CATEGORIES)}")
    print(f"Data Sources: {len(DATA_SOURCES)}")
    print(f"\nTotal Keywords: {sum(len(v['keywords']) for v in TSO_CATEGORIES.values())}")
