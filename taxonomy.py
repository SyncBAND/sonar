"""
SONAR.AI — Pluggable Taxonomy Definition (TSO / Energy Sector)
==============================================================

This file defines WHAT the classifier knows about.
The classifier engine (services/classifier.py) is domain-agnostic;
this file makes it domain-specific.

To adapt SONAR for another domain (pharma, automotive, fintech …),
create a new taxonomy file with the same structure:
    DOMAIN_DESCRIPTION, CATEGORIES, CATEGORY_TO_AMPRION_TASK, etc.

Each category contains:
    name            – Human-readable label
    description     – 2-3 sentences defining the category (weighted 3× in centroid)
    prototypes      – 8-12 example signal titles/snippets (most impactful for quality)
    boost_keywords  – Domain-specific technical terms for keyword boosting
"""

from typing import Dict, List, Any

# =============================================================================
# DOMAIN DESCRIPTION  — used for on-topic / off-topic filtering
# =============================================================================

DOMAIN_DESCRIPTION = (
    "Energy sector and electric power grid technology. "
    "Transmission system operators, electricity networks, grid infrastructure, "
    "renewable energy integration, energy storage, cybersecurity for critical "
    "energy infrastructure, electric vehicles and grid interaction, hydrogen "
    "and power-to-gas, energy markets and trading, grid stability and frequency "
    "control, offshore wind grid connections, digital twins for power systems, "
    "energy regulation and policy, distributed generation and flexibility, "
    "biogas and biomethane, power generation and decarbonisation."
)

# Domain anchor keywords — if NONE appear, the signal is almost certainly off-topic
DOMAIN_ANCHOR_KEYWORDS = frozenset([
    "grid", "power", "energy", "electricity", "transmission", "distribution",
    "renewable", "solar", "wind", "battery", "storage", "hydrogen", "nuclear",
    "turbine", "generator", "transformer", "substation", "voltage", "frequency",
    "utility", "tso", "dso", "offshore", "hvdc", "inverter", "curtailment",
    "balancing", "redispatch", "interconnector", "charging", "ev", "electrolysis",
    "biogas", "biomethane", "flexibility", "demand response", "vpp", "cybersecurity",
    "scada", "ics", "ot security", "smart grid", "meter", "prosumer", "microgrid",
    "regulation", "bnetzA", "entso-e", "acer", "network code", "grid code",
    "decarbonisation", "decarbonization", "emission", "carbon", "climate",
    "fuel cell", "electrolyzer", "photovoltaic", "pv", "mwh", "gwh", "kwh",
    "megawatt", "gigawatt", "kilowatt", "power plant", "generation",
    "capacity", "load", "peak", "baseload", "dispatch", "congestion",
])

# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

CATEGORIES: Dict[str, Dict[str, Any]] = {

    # =========================================================================
    # CORE GRID TECHNOLOGY
    # =========================================================================

    "grid_infrastructure": {
        "name": "Grid Infrastructure & Transmission",
        "description": (
            "Physical electricity transmission infrastructure including HVDC "
            "lines, overhead lines, underground cables, converter stations, "
            "substations, transformers, and switchgear. Covers grid expansion "
            "projects, corridor planning, permitting, and construction of new "
            "transmission assets."
        ),
        "prototypes": [
            "New 380kV overhead transmission line approved for construction in northern Germany",
            "HVDC converter station commissioned for cross-border power interconnection",
            "Underground cable installation begins for major grid expansion corridor",
            "Transformer replacement program upgrades aging substation equipment",
            "Grid reinforcement project addresses transmission bottleneck for renewable capacity",
            "Gas-insulated switchgear installed at new extra-high voltage substation",
            "Transmission operator plans new 2GW interconnector to neighbouring country",
            "Power line route planning and environmental impact assessment completed",
            "Extra-high voltage cable technology advances enable longer underground sections",
            "Substation modernization project replaces decades-old protection relays",
            "Phase-shifting transformer improves cross-border power flow control",
            "SuedLink HVDC corridor reaches construction milestone in southern Germany",
        ],
        "boost_keywords": [
            "hvdc", "transmission line", "overhead line", "underground cable",
            "converter station", "substation", "transformer", "switchgear",
            "circuit breaker", "grid expansion", "interconnector", "corridor",
            "korridor", "suedlink", "ultranet", "a-nord", "balwin", "neulink",
            "rhein-main link", "supergrid", "380kv", "220kv", "pst",
            "gis switchgear", "ais", "busbar", "insulator", "grid reinforcement",
        ],
    },

    "grid_stability": {
        "name": "Grid Stability & System Services",
        "description": (
            "Frequency control, voltage regulation, reactive power management, "
            "and ancillary services for maintaining power system stability. "
            "Includes STATCOM, grid-forming inverters, synthetic inertia, "
            "redispatch, congestion management, and system restoration."
        ),
        "prototypes": [
            "Grid-forming inverter technology enables stable operation with low synchronous inertia",
            "STATCOM installation improves voltage stability in weak grid area",
            "Frequency containment reserve procurement sees record demand from TSOs",
            "Synthetic inertia from wind turbines helps maintain grid frequency during disturbance",
            "Redispatch costs rise sharply as renewable feed-in exceeds transmission capacity",
            "Black start capability tested successfully at major thermal power station",
            "Rate of change of frequency exceeds threshold during low-inertia conditions",
            "Grid booster battery system provides real-time congestion relief on corridor",
            "Reactive power compensation reduces voltage deviations at grid connection point",
            "System operator implements new real-time stability monitoring and control platform",
            "Fault ride through requirements updated for inverter-based resources",
            "Automatic frequency restoration reserve market shows increasing competition",
        ],
        "boost_keywords": [
            "frequency control", "voltage control", "reactive power", "statcom",
            "grid-forming", "inertia", "synthetic inertia", "frequency response",
            "fcr", "afrr", "mfrr", "balancing", "ancillary services",
            "redispatch", "congestion", "grid booster", "fault ride through",
            "black start", "islanding", "rocof", "svc", "facts", "damping",
            "n-1", "spinning reserve", "operating reserve",
        ],
    },

    "offshore_systems": {
        "name": "Offshore Grid Systems",
        "description": (
            "Offshore wind farm grid connections, submarine and subsea cables, "
            "offshore converter platforms, floating foundations, energy islands, "
            "and meshed offshore HVDC grids. Covers the maritime transmission "
            "infrastructure connecting offshore generation to onshore grids."
        ),
        "prototypes": [
            "Offshore wind farm begins power delivery through new HVDC cable connection",
            "Submarine cable fault causes temporary loss of offshore wind generation",
            "Floating offshore wind platform achieves first power for 15MW turbine",
            "Offshore converter platform installed in North Sea for 900MW wind cluster",
            "Multi-terminal HVDC hub enables meshed offshore grid topology",
            "Offshore substation design completed for 2GW wind energy zone",
            "Baltic Sea offshore wind farm reaches final grid connection milestone",
            "Energy island concept advances for centralized offshore power conversion hub",
            "Offshore wind grid connection planning for 70GW target by 2045",
            "Subsea cable route survey completed for new offshore HVDC interconnector",
        ],
        "boost_keywords": [
            "offshore", "offshore wind", "offshore platform", "offshore substation",
            "submarine cable", "subsea cable", "hvdc offshore", "offshore converter",
            "floating platform", "offshore hub", "energy island", "north sea",
            "baltic sea", "offshore grid", "meshed offshore", "offshore maintenance",
            "balwin", "neulink",
        ],
    },

    # =========================================================================
    # ENERGY TRANSITION
    # =========================================================================

    "renewables_integration": {
        "name": "Renewables Integration",
        "description": (
            "Integration of wind and solar generation into the electricity grid. "
            "Covers variability and intermittency challenges, curtailment, "
            "renewable forecasting, inverter technology, capacity factors, and "
            "the impact of large-scale renewable deployment on grid operations."
        ),
        "prototypes": [
            "Solar PV curtailment increases as transmission grid cannot absorb excess generation",
            "Wind power output variability challenges grid frequency management at high penetration",
            "Renewable energy share reaches record 60% of annual electricity generation",
            "New onshore wind farm commissioning delayed due to grid connection capacity limits",
            "Solar inverter grid code compliance testing for updated technical requirements",
            "Agrivoltaics pilot combines solar panels with agricultural land use in Germany",
            "Renewable integration study shows additional grid flexibility needed for 2030 target",
            "Perovskite tandem solar cells achieve record 33% efficiency in laboratory tests",
            "Wind farm wake effects analysis helps optimize turbine layout for maximum yield",
            "Solar forecast accuracy improvement through machine learning enables better scheduling",
            "Renewable portfolio standard drives new utility-scale wind and solar investment",
            "Solar and wind power generation growth drives electricity supply expansion",
        ],
        "boost_keywords": [
            "solar", "photovoltaic", "pv", "wind", "offshore wind", "onshore wind",
            "wind turbine", "solar inverter", "renewable", "curtailment",
            "intermittency", "variability", "ramp rate", "capacity factor",
            "agrivoltaics", "bifacial", "wind farm", "solar farm", "perovskite",
            "renewable forecast", "renewable integration", "res",
        ],
    },

    "energy_storage": {
        "name": "Energy Storage Systems",
        "description": (
            "Grid-scale and utility-scale energy storage including batteries "
            "(lithium, sodium-ion, flow), pumped hydro, compressed air, and "
            "long-duration storage technologies. Covers storage for grid "
            "balancing, frequency response, energy shifting, and seasonal storage."
        ),
        "prototypes": [
            "Grid-scale lithium battery storage project reaches commercial operation",
            "Long-duration energy storage technology selected for seasonal grid balancing",
            "Battery energy storage system provides fast frequency response services to TSO",
            "Pumped hydro storage expansion approved for increased renewable integration support",
            "Sodium-ion battery technology advances as cost-effective alternative to lithium",
            "Vanadium redox flow battery system installed for 8-hour duration application",
            "Battery storage helps manage transmission congestion during peak renewable output",
            "Compressed air energy storage pilot project demonstrates commercial viability",
            "Utility-scale battery project wins capacity market auction at record low price",
            "Iron-air battery offers low-cost path for 100-hour storage duration",
            "Battery storage displaces gas peaker plants during evening peak demand",
        ],
        "boost_keywords": [
            "battery", "bess", "lithium", "sodium ion", "solid state battery",
            "flow battery", "vanadium", "redox flow", "pumped hydro",
            "compressed air", "caes", "flywheel", "supercapacitor",
            "grid-scale storage", "utility-scale battery", "mwh", "gwh",
            "long duration storage", "ldes", "seasonal storage", "gravity storage",
            "thermal storage", "molten salt", "liquid air", "cryogenic",
            "big battery", "battery project", "energy shifting",
        ],
    },

    "power_generation": {
        "name": "Power Generation",
        "description": (
            "Conventional and thermal power generation including gas turbines, "
            "coal plants, nuclear power, combined cycle, CHP, and carbon capture. "
            "Covers plant commissioning, decommissioning, capacity adequacy, "
            "coal phase-out, and the role of dispatchable generation."
        ),
        "prototypes": [
            "Coal-fired power plant decommissioning schedule confirmed under phase-out plan",
            "Gas turbine power plant provides essential backup during winter peak demand",
            "Nuclear power plant achieves record capacity factor during cold weather period",
            "Combined cycle gas turbine upgrade improves efficiency and reduces emissions",
            "Carbon capture and storage project attached to existing gas-fired power station",
            "Hydrogen-ready gas power plant construction approved for grid reserve capacity",
            "Power plant fleet provides critical generation during extreme cold weather event",
            "Capacity mechanism ensures generation adequacy for forecasted winter peak demand",
            "Power station repowering project replaces aging coal units with gas turbines",
            "Emergency generation reserves activated during unexpected winter supply shortfall",
        ],
        "boost_keywords": [
            "power plant", "thermal power", "combined cycle", "ccgt", "gas turbine",
            "chp", "cogeneration", "peak plant", "baseload", "capacity factor",
            "carbon capture", "ccs", "ccus", "coal phase-out", "decommissioning",
            "nuclear", "repowering", "generation adequacy", "heat rate",
        ],
    },

    # =========================================================================
    # SECTOR COUPLING
    # =========================================================================

    "hydrogen_p2g": {
        "name": "Hydrogen & Power-to-Gas",
        "description": (
            "Green hydrogen production via electrolysis, hydrogen pipelines and "
            "transport, power-to-gas conversion, fuel cells, and sector coupling "
            "between electricity and gas networks. Includes hydrogen hubs, "
            "industrial decarbonisation through hydrogen, and gas grid blending."
        ),
        "prototypes": [
            "Green hydrogen electrolyzer project reaches commercial operation at industrial site",
            "Power-to-gas facility converts excess wind energy to hydrogen for storage",
            "Hydrogen pipeline construction connects production sites with industrial demand centers",
            "PEM electrolyzer capacity expands for large-scale green hydrogen production",
            "European hydrogen backbone network planning advances for cross-border transport",
            "Sector coupling through power-to-hydrogen enables heavy industry decarbonisation",
            "Hydrogen storage in salt caverns provides seasonal energy buffer for grid balancing",
            "Green hydrogen import terminal planned for major European port facility",
            "Electrolyzer provides grid balancing services through rapid flexible operation",
            "Hydrogen blending in natural gas network reaches 20% volumetric limit in pilot",
        ],
        "boost_keywords": [
            "hydrogen", "h2", "green hydrogen", "electrolyzer", "electrolysis",
            "pem electrolyzer", "alkaline electrolyzer", "soec", "fuel cell",
            "hydrogen pipeline", "power-to-gas", "p2g", "methanation",
            "hydrogen hub", "hydrogen valley", "hybridge", "get h2",
            "hydrogen backbone", "hydrogen economy", "hydrogen blending",
            "fcev", "hydrogen refueling",
        ],
    },

    "biogas_biomethane": {
        "name": "Biogas & Biomethane",
        "description": (
            "Biogas production from anaerobic digestion, biomethane upgrading "
            "and grid injection, renewable gas certificates, and waste-to-energy "
            "processes. Covers the role of renewable gases in sector coupling "
            "and gas grid decarbonisation."
        ),
        "prototypes": [
            "Biomethane grid injection reaches record levels in German gas network",
            "New biogas plant begins feeding upgraded biomethane into transmission grid",
            "Anaerobic digestion facility processes agricultural waste for renewable energy",
            "Biogas upgrading technology with membrane separation enables gas grid injection",
            "Renewable gas certificate trading system launched for biomethane tracking",
            "Biogas plant optimization improves methane yield from mixed feedstock",
            "Bio-LNG production facility serves heavy transport fuel demand",
            "Waste-to-energy biogas project reduces organic landfill waste stream",
        ],
        "boost_keywords": [
            "biogas", "biomethane", "anaerobic digestion", "biogas plant",
            "biogas upgrading", "gas grid injection", "renewable gas",
            "bio-cng", "bio-lng", "waste-to-energy", "landfill gas",
            "digestate", "feedstock", "green gas certificate",
        ],
    },

    "e_mobility_v2g": {
        "name": "E-Mobility & Vehicle-to-Grid",
        "description": (
            "Electric vehicle charging infrastructure, vehicle-to-grid (V2G) "
            "bidirectional charging, smart charging management, fleet "
            "electrification, and the impact of EV load on distribution and "
            "transmission grids."
        ),
        "prototypes": [
            "Vehicle-to-grid pilot demonstrates EV fleet providing grid frequency response",
            "Ultra-fast charging network expansion along major highway corridors in Europe",
            "Smart charging management system optimizes EV charging for grid conditions",
            "Bidirectional charging standard ISO 15118 enables automated V2G participation",
            "Electric bus fleet depot charging infrastructure installed for city public transport",
            "EV charging demand forecasting model helps distribution grid planning",
            "Megawatt charging system for electric heavy-duty trucks reaches commercial availability",
            "Vehicle-to-home system allows EV battery to power household during grid outages",
            "Fleet management platform optimizes electric vehicle charging schedules and costs",
            "Charging infrastructure operator deploys 200 new fast charger stations nationwide",
        ],
        "boost_keywords": [
            "ev", "electric vehicle", "ev charging", "fast charging",
            "v2g", "vehicle-to-grid", "v2h", "v2x", "bidirectional charging",
            "smart charging", "charging infrastructure", "charge point", "evse",
            "iso 15118", "plug and charge", "e-bus", "e-truck",
            "megawatt charging", "mcs", "electromobility", "emobility",
        ],
    },

    # =========================================================================
    # FLEXIBILITY & MARKETS
    # =========================================================================

    "flexibility_vpp": {
        "name": "Flexibility & Virtual Power Plants",
        "description": (
            "Virtual power plants (VPPs), demand response, distributed energy "
            "resource aggregation, flexibility markets, load shifting, and "
            "TSO-DSO coordination for flexibility procurement. Covers the "
            "orchestration of distributed assets for grid balancing."
        ),
        "prototypes": [
            "Virtual power plant aggregates 10,000 distributed batteries for grid balancing market",
            "Industrial demand response program reduces peak load by 200MW for transmission operator",
            "Flexibility market platform connects distributed resources with grid operators",
            "Heat pump fleet provides demand-side flexibility for grid frequency support service",
            "DER aggregation platform enables prosumers to participate in balancing market",
            "Automated load shifting program uses smart controls for residential demand management",
            "TSO-DSO coordination platform shares flexibility data between grid levels",
            "Behind-the-meter storage aggregation creates 50MW virtual power plant portfolio",
            "Flexibility procurement mechanism enables market-based congestion management by TSO",
            "Demand response aggregator prequalified for automatic frequency restoration reserve",
        ],
        "boost_keywords": [
            "vpp", "virtual power plant", "aggregation", "demand response",
            "demand side management", "dsm", "load shifting", "peak shaving",
            "flexibility market", "flexibility trading", "der aggregation",
            "distributed energy resources", "prosumer", "behind the meter",
            "flexibility procurement", "aggregator", "balancing responsible party",
        ],
    },

    "energy_trading": {
        "name": "Energy Trading & Markets",
        "description": (
            "Electricity wholesale markets, day-ahead and intraday trading, "
            "balancing markets, capacity markets, market coupling, price "
            "formation, and the evolution of energy market design. Covers "
            "exchanges like EPEX, EEX, and cross-border market integration."
        ),
        "prototypes": [
            "Day-ahead electricity market prices spike during extreme cold weather event",
            "Intraday continuous trading volume reaches new record on European energy exchange",
            "Flow-based market coupling methodology improves cross-border capacity allocation",
            "Balancing market reform introduces faster 15-minute settlement period",
            "Energy exchange launches new flexibility products for demand-side participation",
            "Algorithmic trading adoption increases in European electricity spot markets",
            "Cross-border electricity trading capacity expanded on new interconnector route",
            "Spot market price volatility increases with higher renewable energy penetration",
            "Market coupling algorithm optimizes social welfare across European bidding zones",
            "Capacity auction clears at record price reflecting tight generation supply margin",
            "Spark and dark spreads widen as gas and coal plant profitability improves",
        ],
        "boost_keywords": [
            "energy trading", "power trading", "electricity market",
            "spot market", "futures", "day-ahead", "intraday",
            "balancing market", "capacity market", "market coupling",
            "price formation", "merit order", "marginal price",
            "epex spot", "eex", "nordpool", "auction", "continuous trading",
            "spark spread", "dark spread", "clean spark spread",
            "wholesale price", "electricity price", "price spike",
        ],
    },

    "distributed_generation": {
        "name": "Distributed Generation",
        "description": (
            "Small-scale, decentralised electricity generation including "
            "rooftop solar, community energy, microgrids, prosumer systems, "
            "and local energy markets. Covers the impact of distributed "
            "generation on power flows and TSO-DSO coordination needs."
        ),
        "prototypes": [
            "Rooftop solar PV installations reach record levels in residential sector",
            "Distributed generation growth reverses power flow patterns in distribution grid",
            "Microgrid project demonstrates islanded operation with local solar and battery",
            "Energy community shares locally generated solar power among neighbourhood members",
            "DERMS platform manages growing fleet of distributed energy resources for utility",
            "Prosumer installations with battery storage increase self-consumption rates to 80%",
            "Community battery storage serves neighbourhood of 50 rooftop solar households",
            "Local energy market enables peer-to-peer electricity trading between prosumers",
            "Net metering policy reform impacts economics of small-scale distributed solar",
            "Feed-in tariff reduction changes investment case for new distributed generation",
        ],
        "boost_keywords": [
            "distributed generation", "rooftop solar", "rooftop pv",
            "microgrid", "prosumer", "net metering", "feed-in tariff",
            "self-consumption", "peer-to-peer", "energy community",
            "community solar", "community battery", "derms",
            "behind the meter", "self-generation", "citizen energy",
        ],
    },

    # =========================================================================
    # DIGITAL & SECURITY
    # =========================================================================

    "cybersecurity_ot": {
        "name": "Cybersecurity & OT Security",
        "description": (
            "Cybersecurity for operational technology (OT) in energy and "
            "critical infrastructure. Covers SCADA and ICS security, threat "
            "detection, vulnerability management, incident response, NIS2 "
            "compliance, post-quantum cryptography, and zero-trust architecture "
            "for grid control systems."
        ),
        "prototypes": [
            "Critical vulnerability discovered in industrial control system SCADA software",
            "Ransomware attack targets European energy utility operational technology network",
            "NIS2 directive sets new cybersecurity requirements for critical energy operators",
            "IEC 62443 security certification completed for grid automation components",
            "Threat intelligence report identifies state-sponsored attacks targeting power grids",
            "Zero-trust network segmentation deployed for substation OT communication systems",
            "Post-quantum cryptography migration assessment initiated for grid communications",
            "Incident response exercise simulates coordinated cyber attack on grid control room",
            "CISA issues critical advisory for vulnerabilities in energy sector SCADA systems",
            "Security operations center detects intrusion attempt on grid management network",
            "OT network monitoring platform deployed across transmission operator substations",
        ],
        "boost_keywords": [
            "cybersecurity", "ot security", "scada", "ics security",
            "critical infrastructure", "nerc cip", "nis2", "kritis",
            "zero trust", "threat detection", "siem", "soc",
            "incident response", "vulnerability", "penetration testing",
            "post-quantum", "pqc", "ransomware", "malware", "phishing",
            "iec 62443", "iec 62351", "cyber attack", "threat intelligence",
            "cyber resilience", "network segmentation",
        ],
    },

    "ai_grid_optimization": {
        "name": "AI & Grid Optimization",
        "description": (
            "Artificial intelligence and machine learning applied specifically "
            "to energy and power grid operations. Covers AI for load and "
            "renewable forecasting, grid state estimation, optimal power flow, "
            "predictive maintenance of grid assets, and autonomous grid control. "
            "Does NOT include general AI/tech industry news unrelated to energy."
        ),
        "prototypes": [
            "Machine learning model improves wind power forecasting accuracy by 15% for TSO",
            "Deep reinforcement learning algorithm optimizes power flow across transmission grid",
            "AI-based load forecasting reduces redispatch costs for transmission system operator",
            "Neural network predicts transformer remaining useful life from sensor data",
            "Computer vision automates defect detection in power line drone inspection imagery",
            "Federated learning enables privacy-preserving demand forecasting across utilities",
            "AI optimization of battery storage dispatch reduces balancing costs by 20%",
            "Multi-agent AI system coordinates scheduling of distributed energy resources",
            "Predictive analytics platform identifies aging grid equipment before failure occurs",
            "Natural language processing extracts structured outage data from utility reports",
        ],
        "boost_keywords": [
            "grid optimization", "load forecasting", "demand forecasting",
            "renewable forecasting", "power flow", "optimal power flow",
            "state estimation", "predictive maintenance", "condition monitoring",
            "energy management system", "grid ai", "energy ai",
            "unit commitment", "economic dispatch",
        ],
    },

    "digital_twin_simulation": {
        "name": "Digital Twins & Simulation",
        "description": (
            "Digital twin models of power systems and grid assets, real-time "
            "and offline simulation, hardware-in-the-loop testing, state "
            "estimation, contingency analysis, and scenario modelling for "
            "grid planning and operations."
        ),
        "prototypes": [
            "Digital twin of transmission network enables real-time contingency analysis",
            "Power system simulation validates grid stability under high renewable scenarios",
            "Real-time hardware-in-the-loop testing validates new protection relay settings",
            "Grid model updated with latest network topology for improved state estimation",
            "Probabilistic simulation assesses grid reliability under extreme weather conditions",
            "Digital twin integrates BIM and GIS data for comprehensive asset management",
            "Co-simulation platform models coupled electricity and gas network interactions",
            "Dynamic line rating model uses weather data to maximize transmission capacity",
            "Scenario analysis tool evaluates grid impact of 5GW new generation connections",
            "4D corridor modelling simulates construction progress for major grid project",
        ],
        "boost_keywords": [
            "digital twin", "simulation", "grid model", "state estimation",
            "contingency analysis", "real-time simulation", "hil",
            "hardware in loop", "co-simulation", "scenario analysis",
            "psse", "powerfactory", "pscad", "emtp", "probabilistic simulation",
            "dynamic line rating",
        ],
    },

    # =========================================================================
    # CROSS-CUTTING
    # =========================================================================

    "regulatory_policy": {
        "name": "Regulatory & Policy",
        "description": (
            "Energy regulation, policy frameworks, grid codes, network codes, "
            "and regulatory decisions affecting electricity transmission. "
            "Covers BNetzA, ACER, ENTSO-E policy, EU energy legislation, "
            "network tariffs, and market design reforms."
        ),
        "prototypes": [
            "BNetzA approves updated network development plan for German grid expansion",
            "European clean energy package introduces revised grid connection codes",
            "ACER publishes updated methodology for cross-border capacity calculation",
            "Energy regulation reform changes incentive framework for TSO grid investment",
            "Network tariff structure review impacts allocation of transmission grid costs",
            "ENTSO-E publishes updated ten-year network development plan for Europe",
            "Grid code amendment requires faster frequency response from new generators",
            "Regulatory sandbox enables testing of innovative flexibility market products",
            "EU Fit for 55 legislation accelerates renewable energy deployment targets",
            "Network operator consultation on grid fee reform receives stakeholder input",
        ],
        "boost_keywords": [
            "regulation", "policy", "bnetza", "bundesnetzagentur", "acer",
            "entso-e", "network code", "eu regulation", "clean energy package",
            "red iii", "fit for 55", "grid code", "connection code",
            "aregv", "incentive regulation", "tariff", "grid fee",
            "network development plan", "tyndp", "market design",
            "network tariff", "unbundling",
        ],
    },
}


# =============================================================================
# DETERMINISTIC MAPPINGS  (category → Amprion dimensions)
# =============================================================================

CATEGORY_TO_AMPRION_TASK: Dict[str, str | None] = {
    "grid_infrastructure":      "grid_expansion",
    "grid_stability":           "system_security",
    "offshore_systems":         "grid_expansion",
    "renewables_integration":   "decarbonization",
    "energy_storage":           "system_security",
    "power_generation":         "system_security",
    "hydrogen_p2g":             "decarbonization",
    "biogas_biomethane":        "decarbonization",
    "e_mobility_v2g":           "decarbonization",
    "flexibility_vpp":          "european_trading",
    "energy_trading":           "european_trading",
    "distributed_generation":   "decarbonization",
    "cybersecurity_ot":         "system_security",
    "ai_grid_optimization":     "system_security",
    "digital_twin_simulation":  "system_security",
    "regulatory_policy":        "european_trading",
    "off_topic":                None,
}

CATEGORY_TO_BUSINESS_AREAS: Dict[str, List[str]] = {
    "grid_infrastructure":      ["GP", "AM"],
    "grid_stability":           ["SO", "AM"],
    "offshore_systems":         ["GP", "AM", "CS"],
    "renewables_integration":   ["SO", "M"],
    "energy_storage":           ["SO", "AM", "M"],
    "power_generation":         ["SO"],
    "hydrogen_p2g":             ["GP", "CS", "M"],
    "biogas_biomethane":        ["M", "SO", "CS"],
    "e_mobility_v2g":           ["SO", "GP", "AM"],
    "flexibility_vpp":          ["SO", "M", "ITD"],
    "energy_trading":           ["M", "SO"],
    "distributed_generation":   ["SO", "M"],
    "cybersecurity_ot":         ["ITD", "SO"],
    "ai_grid_optimization":     ["ITD", "SO", "AM"],
    "digital_twin_simulation":  ["ITD", "AM"],
    "regulatory_policy":        ["CS", "M"],
    "off_topic":                [],
}

CATEGORY_TO_STRATEGIC_DOMAIN: Dict[str, str | None] = {
    "grid_infrastructure":      "digital_asset_lifecycle",
    "grid_stability":           "system_operation",
    "offshore_systems":         "system_operation",
    "renewables_integration":   "system_operation",
    "energy_storage":           "energy_markets_flexibility",
    "power_generation":         "system_operation",
    "hydrogen_p2g":             "energy_markets_flexibility",
    "biogas_biomethane":        "energy_markets_flexibility",
    "e_mobility_v2g":           "energy_markets_flexibility",
    "flexibility_vpp":          "energy_markets_flexibility",
    "energy_trading":           "energy_markets_flexibility",
    "distributed_generation":   "energy_markets_flexibility",
    "cybersecurity_ot":         "cybersecurity_trust",
    "ai_grid_optimization":     "system_operation",
    "digital_twin_simulation":  "digital_asset_lifecycle",
    "regulatory_policy":        "system_operation",
    "off_topic":                None,
}


# =============================================================================
# HELPERS
# =============================================================================

ON_TOPIC_CATEGORIES = [c for c in CATEGORIES if c != "off_topic"]

if __name__ == "__main__":
    print(f"Taxonomy: {len(CATEGORIES)} categories")
    for cid, cdata in CATEGORIES.items():
        print(f"  {cid:30s}  {len(cdata['prototypes']):2d} prototypes  "
              f"{len(cdata['boost_keywords']):2d} keywords")
    print(f"\nDomain anchor keywords: {len(DOMAIN_ANCHOR_KEYWORDS)}")
