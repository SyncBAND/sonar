# SONAR.AI / AGTS v2.0

## Agentic Global Trend Scanner for Transmission System Operators

Based on the AGTS Framework specification (Pages 6-22), this system implements:

- **Multi-Agent Architecture**: Scout, Architect, Strategist, Validator, Prioritizer
- **TSO-Specific Taxonomies**: 15+ categories aligned with TSO operations
- **Amprion-Specific Classification**: Business areas, key tasks, mega-projects
- **No-Regret Prioritization**: Weighted scoring framework (Strategic Relevance 40%, Grid Stability 30%, Cost Efficiency 30%)
- **Multi-Dimensional Filtering**: As specified on Pages 17-18
- **Stakeholder Views**: Executive Board, IT Architecture, Grid Planning, Sustainability

---

## 📁 Project Structure

```
sonar_ai_v2/
├── main.py                 # FastAPI application
├── config.py               # Complete AGTS configuration
├── requirements.txt        # Dependencies
├── models/
│   ├── __init__.py
│   └── database.py         # SQLAlchemy models (AGTS-compliant)
└── services/
    ├── __init__.py
    ├── nlp_processor.py    # Multi-dimensional NLP classification
    └── trend_scorer.py     # No-Regret prioritization engine
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd sonar_ai_v2
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the Server

```bash
uvicorn main:app --reload
```

### 3. Access API

- Swagger UI: http://localhost:8000/docs
- Dashboard: http://localhost:8000/api/v1/dashboard
- Trends: http://localhost:8000/api/v1/trends
- Taxonomies: http://localhost:8000/api/v1/taxonomies

---

## 📊 TSO Categories (config.py)

| Category | Description | Relevance Weight |
|----------|-------------|------------------|
| `grid_infrastructure` | HVDC, Transmission Lines, Substations | 1.00 |
| `grid_stability` | Frequency Control, STATCOM, Inertia | 1.00 |
| `offshore_systems` | Offshore Platforms, Submarine Cables | 0.95 |
| `energy_storage` | Batteries, BESS, Pumped Hydro | 0.95 |
| `renewables_integration` | Solar, Wind, Curtailment | 0.95 |
| `flexibility_vpp` | Virtual Power Plants, Aggregation | 0.90 |
| `e_mobility_v2g` | EV Charging, Vehicle-to-Grid | 0.90 |
| `ai_grid_optimization` | Machine Learning, Optimization | 0.90 |
| `hydrogen_p2g` | Electrolyzers, Power-to-Gas | 0.85 |
| `cybersecurity_ot` | SCADA Security, Post-Quantum | 0.85 |
| `digital_twin_simulation` | Simulation, Modeling | 0.85 |
| `biogas_biomethane` | Biogas, Grid Injection | 0.80 |
| `energy_trading` | Markets, Spot, Intraday | 0.80 |
| `heat_district_energy` | District Heating, Heat Pumps | 0.75 |
| `regulatory_policy` | BNetzA, ACER, Network Codes | 0.75 |

---

## 🏛️ Strategic Domains (Page 8-9)

### Domain 1: System Operation & Digital Control Room
- Multi-Agent Orchestration
- Probabilistic Forecasting
- Edge-to-Cloud Continuum

### Domain 2: Digital Asset Life Cycle & Infrastructure
- Predictive Maintenance & IoT
- Digital Twins (BIM & GIS)
- SF6-Free & Circular IT

### Domain 3: Integrated Energy Markets & Flexibility
- TSO-DSO Interoperability
- Automated Market Clearing
- Sector Coupling (P2X)

### Domain 4: Cybersecurity & Trust Foundations
- Preemptive Cybersecurity
- Post-Quantum Cryptography (PQC)
- Sovereign Cloud & Geopatriation

---

## 🏗️ Architectural Layers (Page 9-10)

| Layer | Focus | Examples |
|-------|-------|----------|
| **Perception & Actuator** | The Grid's "Senses" | Drones, Sensors, Smart Materials |
| **Communication & Connectivity** | The Grid's "Nerves" | 5G/6G, SDN, Industrial IoT |
| **Data & Intelligence** | The Grid's "Brain" | Federated Learning, Knowledge Graphs |
| **Service & Orchestration** | The Grid's "Economy" | VPP, Blockchain, Market Coupling |

---

## 🎯 Amprion-Specific Configuration

### Business Areas (Page 10-12)
- **CS**: Corporate Strategy & Development
- **SO**: System Operation
- **AM**: Asset Management & Technology
- **GP**: Grid Projects & Expansion
- **M**: Market & Energy Economy
- **ITD**: Corporate Functions (IT/Digital)

### Key Tasks (Fact Book)
- Grid Expansion & Re-construction
- System Security
- Decarbonization
- European Electricity Trading

### Mega-Projects
- Korridor B (DC31, DC32)
- Rhein-Main Link
- A-Nord
- Ultranet
- SuedLink
- BalWin1 & BalWin2
- NeuLink

---

## 📈 No-Regret Prioritization Framework (Page 19-22)

### Priority Score Formula
```
PS = (SR × 0.40 + GS × 0.30 + CE × 0.30) × Project_Multiplier
```

Where:
- **SR** = Strategic Relevance Score (0-100)
- **GS** = Grid Stability Impact Score (0-100)
- **CE** = Cost Efficiency Score (0-100)

### Project Multipliers
| Project Type | Multiplier |
|--------------|------------|
| Strategic Mega-Project (Korridor B, etc.) | 1.50x |
| Strategic/Stability-Critical | 1.30x |
| Tactical (1-3 year improvements) | 1.10x |
| Operational (day-to-day) | 1.00x |

---

## 📋 Trend List Fields (Page 13-15)

### Core Identity & Classification
- `trend_id`: Unique identifier (e.g., TR-2026-001)
- `name`: Descriptive, non-technical title
- `description_short`: 250-char executive summary
- `description_full`: 2000-char deep dive
- `lifecycle_status`: Scouting / Pilot / Implementation / Standard
- `maturity_type`: TRL / MRL / SRL / RRL
- `maturity_score`: 1-9 scale
- `time_to_impact`: <1 year / 1-3 years / 3-5 years / 5+ years

### Strategic Prioritization
- `priority_score`: 1.0 - 10.0 calculated value
- `strategic_nature`: Accelerator / Disruptor / Transformational
- `key_amprion_task`: Grid Expansion / System Security / etc.
- `field_of_action`: Robust Planning / Grid Operation Evolution / etc.

### Impact & Business Context
- `business_area`: SO / AM / ITD / GP / M / CS
- `linked_projects`: ["Korridor B", "Rhein-Main Link"]
- `regulatory_sensitivity`: High / Medium / Low
- `sovereignty_score`: 0-100

### Technical & Market Intelligence
- `architectural_layer`: Perception / Connectivity / Data / Service
- `customer_impact`: Industrial demand changes
- `market_impact`: Price formation changes
- `so_what_summary`: 2-sentence action/inaction consequence

---

## 👥 Stakeholder Views (Page 15-16)

### Executive Board
Priority fields: priority_score, strategic_nature, regulatory_sensitivity, so_what_summary

### IT Architecture
Priority fields: architectural_layer, sovereignty_score, TRL, lifecycle_status

### Grid Planning (AM/SO)
Priority fields: key_amprion_task, linked_projects, time_to_impact, customer_impact

### Sustainability Team
Priority fields: decarbonization_task, ESG_delta, circular_IT_potential

---

## 🔌 API Endpoints

### Dashboard
```
GET /api/v1/dashboard
```

### Signals
```
GET  /api/v1/signals
POST /api/v1/signals/process
```

### Trends
```
GET  /api/v1/trends
GET  /api/v1/trends/{id}
POST /api/v1/trends/create
POST /api/v1/trends/score
```

### Taxonomies
```
GET /api/v1/taxonomies
GET /api/v1/stakeholder-views/{view_type}
```

### Alerts
```
GET /api/v1/alerts
PUT /api/v1/alerts/{id}/read
```

---

## 🌐 Data Sources

### Research & Patents
- arXiv (eess.SY, cs.AI)
- Google Patents
- WIPO

### Energy News
- Utility Dive
- PV Magazine
- Energy Storage News
- Recharge News
- Clean Energy Wire
- Montel News

### Industry Events
- Enlit Europe
- E-world Energy & Water
- WindEurope
- SolarPower Europe

### Regulatory & TSOs
- BNetzA
- ACER
- ENTSO-E
- Amprion, TenneT, 50Hertz, TransnetBW

### Research Organizations
- IEA
- IRENA
- Fraunhofer ISE
- dena

---

## 📝 Example: Biogas Trend Interpretation

When the system detects increasing biogas signals:

```
Category: biogas_biomethane
Strategic Impact: "Renewable gas can be wheeled through existing gas infrastructure"
TSO Relevance: Sector coupling opportunity
Amprion Task: Decarbonization
Business Area: M (Market & Energy Economy)

Interpretation:
- Biogas/biomethane can be injected into gas grid
- Creates flexibility for power generation
- Reduces need for new transmission capacity
- Supports P2X and sector coupling strategy
```

---

## 📝 Example: V2G/VPP Trend Interpretation

```
Category: flexibility_vpp, e_mobility_v2g
Strategic Impact: "EVs treated as distributed storage providing frequency response"
TSO Relevance: Critical for balancing in high-renewable grid
Amprion Task: System Security, Decarbonization

Interpretation:
- 1000 EVs = 10MW virtual battery
- Aggregation platforms enable TSO access to distributed flexibility
- Reduces need for dedicated balancing plants
- Enables TSO-DSO flexibility trading
```

---

## 🔧 Configuration Tips

### Adjusting Priority Weights
In `config.py`:
```python
PRIORITY_SCORING_WEIGHTS = {
    "strategic_relevance": 0.40,  # Increase for strategy focus
    "grid_stability": 0.30,       # Increase for operations focus
    "cost_efficiency": 0.30       # Increase for finance focus
}
```

### Adding Custom Categories
In `config.py`, add to `TSO_CATEGORIES`:
```python
"new_category": {
    "name": "New Category Name",
    "keywords": ["keyword1", "keyword2"],
    "relevance_weight": 0.85,
    "strategic_impact": "Description of TSO impact"
}
```

---

## 📄 License

Built for Amprion Challenge via prototype.club

Demo Day: February 4, 2026
