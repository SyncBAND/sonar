# SONAR.AI v2 - Complete System Documentation

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Technology Stack](#2-technology-stack)
3. [Data Flow Pipeline](#3-data-flow-pipeline)
4. [Data Sources & How to Add More](#4-data-sources--how-to-add-more)
5. [Taxonomy & Classification System](#5-taxonomy--classification-system)
6. [NLP Processing](#6-nlp-processing)
7. [Scoring Algorithm](#7-scoring-algorithm)
8. [API Endpoints](#8-api-endpoints)
9. [Time-Based Filtering](#9-time-based-filtering)
10. [Configuration & Customization](#10-configuration--customization)
11. [Next Steps & Roadmap](#11-next-steps--roadmap)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SONAR.AI / AGTS Architecture                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │   SCRAPERS   │──▶│     NLP      │──▶│   SCORER     │                 │
│  │  (Scout)     │   │  (Architect) │   │ (Strategist) │                 │
│  └──────────────┘   └──────────────┘   └──────────────┘                 │
│         │                  │                  │                          │
│         ▼                  ▼                  ▼                          │
│  ┌─────────────────────────────────────────────────────┐                │
│  │                    SQLite DATABASE                   │                │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │                │
│  │  │ Signals │  │ Trends  │  │Clusters │  │ Alerts │ │                │
│  │  └─────────┘  └─────────┘  └─────────┘  └────────┘ │                │
│  └─────────────────────────────────────────────────────┘                │
│                           │                                              │
│                           ▼                                              │
│  ┌─────────────────────────────────────────────────────┐                │
│  │                   FastAPI REST API                   │                │
│  │  /signals  /trends  /dashboard  /impact  /search    │                │
│  └─────────────────────────────────────────────────────┘                │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components:

| Component | File | Purpose |
|-----------|------|---------|
| **Scrapers** | `services/scrapers.py` | Collect signals from data sources |
| **NLP Processor** | `services/nlp_processor.py` | Classify signals into categories |
| **Trend Scorer** | `services/trend_scorer.py` | Calculate priority scores |
| **Impact Analyzer** | `services/enhanced_scrapers.py` | TSO business impact analysis |
| **API** | `main.py` | REST API endpoints |
| **Database** | `models/database.py` | SQLAlchemy ORM models |
| **Config** | `config.py` | Taxonomies, keywords, settings |
| **Pipeline** | `full.py` | Command-line pipeline runner |

---

## 2. Technology Stack

### Current Stack:

| Layer | Technology | Version |
|-------|------------|---------|
| **Language** | Python | 3.10+ |
| **Web Framework** | FastAPI | 0.100+ |
| **Database** | SQLite | (SQLAlchemy ORM) |
| **HTTP Client** | aiohttp | Async scraping |
| **RSS Parsing** | feedparser | RSS/Atom feeds |
| **XML Parsing** | lxml, BeautifulSoup | arXiv, web scraping |
| **NLP** | spaCy (optional) | Entity extraction |
| **Server** | Uvicorn | ASGI server |

### File Structure:

```
sonar_ai_v2/
├── config.py                 # All configuration, taxonomies, keywords
├── main.py                   # FastAPI application & API endpoints
├── full.py                   # CLI pipeline runner
├── requirements.txt          # Python dependencies
├── sonar_ai.db              # SQLite database (created on first run)
│
├── models/
│   └── database.py          # SQLAlchemy models (Signal, Trend, etc.)
│
└── services/
    ├── scrapers.py          # Data source scrapers
    ├── nlp_processor.py     # NLP classification
    ├── trend_scorer.py      # No-Regret scoring algorithm
    ├── enhanced_scrapers.py # TSO impact analyzer, link validator
    ├── tso_interpreter.py   # TSO-specific interpretation rules
    └── business_impact.py   # Business impact definitions
```

---

## 3. Data Flow Pipeline

### Step-by-Step Flow:

```
1. SCRAPE          2. PROCESS         3. CLUSTER         4. SCORE           5. SERVE
   │                  │                  │                  │                  │
   ▼                  ▼                  ▼                  ▼                  ▼
┌──────────┐    ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
│  RSS     │    │  Title + │      │  Group   │      │ Priority │      │   API    │
│  arXiv   │───▶│  Content │─────▶│  by      │─────▶│  Score   │─────▶│ Swagger  │
│  Events  │    │  → TSO   │      │ Category │      │ Formula  │      │  JSON    │
│  News    │    │  Category│      │          │      │          │      │          │
└──────────┘    └──────────┘      └──────────┘      └──────────┘      └──────────┘
```

### Commands:

```bash
# Run individual steps:
python full.py --scrape-only      # Step 1: Collect signals
python full.py --process-only     # Step 2: NLP classification
python full.py --cluster-only     # Step 3: Group into trends
python full.py --score-only       # Step 4: Calculate scores
python full.py --report           # Step 5: Generate report

# Run entire pipeline:
python full.py                    # All steps

# Start API server:
uvicorn main:app --reload
```

---

## 4. Data Sources & How to Add More

### Current Data Sources:

| Source | Type | File Location | Signals/Run |
|--------|------|---------------|-------------|
| **arXiv** | Research papers | `scrapers.py` line ~180 | ~256 |
| **Utility Dive** | RSS News | `scrapers.py` line ~80 | ~10 |
| **Energy Storage News** | RSS News | `scrapers.py` line ~80 | ~50 |
| **Electrek** | RSS News | `scrapers.py` line ~80 | ~50 |
| **ACER/EU Regulatory** | Web scraping | `scrapers.py` line ~280 | ~10 |
| **Industry Events** | Static list | `scrapers.py` line ~350 | ~12 |
| **Enlit World** | Web scraping | `scrapers.py` line ~400 | ~8 |

### 🔧 HOW TO ADD NEW RSS FEEDS:

**Location:** `services/scrapers.py` around line 80

```python
RSS_FEEDS = {
    # Existing feeds...
    "utility_dive": {
        "url": "https://www.utilitydive.com/feeds/news/",
        "source_name": "Utility Dive",
        "quality_score": 0.80
    },
    
    # ⬇️ ADD NEW FEEDS HERE ⬇️
    
    # Example: Biogas newsletter
    "european_biogas": {
        "url": "https://www.europeanbiogas.eu/feed/",
        "source_name": "European Biogas Association",
        "quality_score": 0.85
    },
    
    # Example: Offshore wind
    "offshore_wind_biz": {
        "url": "https://www.offshorewind.biz/feed/",
        "source_name": "Offshore Wind Biz",
        "quality_score": 0.80
    },
    
    # Example: Smart Energy
    "smart_energy_intl": {
        "url": "https://www.smart-energy.com/feed/",
        "source_name": "Smart Energy International",
        "quality_score": 0.85
    },
    
    # Example: German energy news
    "clean_energy_wire": {
        "url": "https://www.cleanenergywire.org/rss.xml",
        "source_name": "Clean Energy Wire",
        "quality_score": 0.90
    },
}
```

### 🔧 HOW TO ADD CUSTOM SCRAPERS:

**Location:** `services/scrapers.py`

```python
# Add a new scraper method in MasterScraper class:

async def scrape_custom_source(self) -> List[Dict]:
    """Scrape your custom source."""
    signals = []
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("https://your-source.com/api") as response:
                data = await response.json()
                
                for item in data["articles"]:
                    signal = {
                        "title": item["title"],
                        "content": item["summary"],
                        "url": item["link"],
                        "source_type": "news",  # or "research", "regulatory", etc.
                        "source_name": "Your Source Name",
                        "source_quality_score": 0.80,
                        "published_at": parse_date(item["date"]),
                        "scraped_at": datetime.utcnow()
                    }
                    signals.append(signal)
                    
        except Exception as e:
            logger.error(f"Custom scrape error: {e}")
    
    return signals

# Then add to scrape_all() method:
async def scrape_all(self, sources=None):
    # ... existing code ...
    
    if "custom" in sources:
        signals = await self.scrape_custom_source()
        all_signals.extend(signals)
```

### Recommended Sources to Add:

| Source | URL | Category Coverage |
|--------|-----|-------------------|
| European Biogas Association | europeanbiogas.eu/feed/ | Biogas |
| Offshore Wind Biz | offshorewind.biz/feed/ | Offshore |
| Smart Energy International | smart-energy.com/feed/ | Smart Grid |
| Clean Energy Wire | cleanenergywire.org/rss.xml | German Energy |
| Recharge News | rechargenews.com/rss | Renewables |
| Energy Monitor | energymonitor.ai/feed/ | Analysis |
| ENTSO-E News | entsoe.eu/news/rss | TSO Official |
| WindEurope | windeurope.org/feed/ | Wind |

---

## 5. Taxonomy & Classification System

### TSO Categories (23 Categories):

**Location:** `config.py` - `TSO_CATEGORIES` dictionary

```python
TSO_CATEGORIES = {
    # CRITICAL for TSO (weight 1.0)
    "grid_infrastructure": {
        "name": "Grid Infrastructure & Transmission",
        "keywords": ["hvdc", "transformer", "substation", "transmission line", ...],
        "relevance_weight": 1.0,
        "enlit_mapping": ["Transmission & Distribution", "Grid Management"]
    },
    
    "grid_stability": {
        "name": "Grid Stability & System Services",
        "keywords": ["frequency", "inertia", "grid-forming", "statcom", ...],
        "relevance_weight": 1.0
    },
    
    "cybersecurity_ot": {
        "name": "Cybersecurity & OT Security",
        "keywords": ["scada", "intrusion", "ransomware", "nis2", ...],
        "relevance_weight": 1.0
    },
    
    # HIGH relevance (weight 0.9)
    "energy_storage": {...},
    "flexibility_vpp": {...},
    "offshore_systems": {...},
    
    # MEDIUM relevance (weight 0.8)
    "renewables_integration": {...},
    "e_mobility_v2g": {...},
    "hydrogen_p2g": {...},
    
    # etc...
}
```

### 🔧 HOW TO ADD/MODIFY CATEGORIES:

```python
# In config.py, add to TSO_CATEGORIES:

"your_new_category": {
    "name": "Display Name for UI",
    "keywords": [
        "keyword1", "keyword2", "phrase with spaces",
        "technical term", "acronym"
    ],
    "relevance_weight": 0.85,  # 0.0-1.0, higher = more TSO-relevant
    "enlit_mapping": ["Enlit Category 1", "Enlit Category 2"]  # Optional
}
```

### Multi-Dimensional Classification:

Each signal is classified on **6 dimensions**:

| Dimension | Options | Purpose |
|-----------|---------|---------|
| **TSO Category** | 23 categories | Technical domain |
| **Strategic Domain** | 4 domains | Business alignment |
| **Architectural Layer** | 5 layers | IT architecture |
| **Amprion Task** | 4 tasks | Key business task |
| **Business Area** | 6 areas | Department relevance |
| **Impact Type** | 3 types | Change nature |

### Strategic Domains (Page 8-9 of AGTS):

```
1. System Operation          - Real-time grid control
2. Digital Asset Lifecycle   - Asset management, digital twins
3. Energy Markets & Flexibility - Trading, balancing, flexibility
4. Cybersecurity & Trust     - Security, privacy, compliance
```

### Architectural Layers (Page 9-10 of AGTS):

```
1. Perception & Actuator     - Sensors, IoT, edge devices
2. Communication & Connectivity - Networks, protocols
3. Data & Integration        - Data platforms, APIs
4. Service & Orchestration   - Applications, automation
5. Business & Ecosystem      - Strategy, partnerships
```

### Impact Types:

```
Accelerator      - Enhances existing capabilities
Disruptor        - Creates new paradigms
Transformational - Both accelerating and disrupting
```

---

## 6. NLP Processing

### How Classification Works:

**Location:** `services/nlp_processor.py`

```
Input: Signal (title + content)
                │
                ▼
┌───────────────────────────────────────┐
│  1. ABSOLUTE OVERRIDES                │
│     "intrusion detection" → cybersecurity (100%)
│     "phishing" → cybersecurity (100%)  │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  2. PRIORITY KEYWORDS                 │
│     Check cybersecurity keywords first │
│     Check grid_stability keywords      │
│     Score >= 6 → Return immediately   │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  3. STANDARD MATCHING                 │
│     For each category:                │
│       Count keyword matches           │
│       Apply relevance_weight          │
│       Track score                     │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  4. DISAMBIGUATION                    │
│     If "security" in text:            │
│       Reduce renewables score         │
│       Boost cybersecurity score       │
│     If AI but no energy context:      │
│       → ai_grid_optimization          │
└───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  5. FALLBACK RULES                    │
│     If score < 1.0:                   │
│       Match to broad categories       │
│       or return "other"               │
└───────────────────────────────────────┘
                │
                ▼
Output: (category, confidence_score)
```

### Classification Algorithm:

```python
def _classify_tso_category(self, text: str) -> Tuple[str, float]:
    
    # 1. Absolute overrides (always match)
    CYBER_ABSOLUTE = ["intrusion detection", "phishing", "malware", ...]
    for kw in CYBER_ABSOLUTE:
        if kw in text:
            return "cybersecurity_ot", 1.0  # 100% confidence
    
    # 2. Priority keywords (strong signals)
    cyber_score = sum(3 for kw in CYBER_STRONG if kw in text)
    if cyber_score >= 6:
        return "cybersecurity_ot", min(cyber_score/10, 1.0)
    
    # 3. Standard matching
    for category, data in TSO_CATEGORIES.items():
        score = 0
        for keyword in data["keywords"]:
            if keyword in text:
                score += 2 if exact_match else 1
        score *= data["relevance_weight"]
        category_scores[category] = score
    
    # 4. Disambiguation
    if has_security_context:
        category_scores["renewables_integration"] *= 0.3
        category_scores["cybersecurity_ot"] += 2
    
    # 5. Return best match
    best = max(category_scores, key=category_scores.get)
    return best, category_scores[best] / 10.0
```

### Will Results Change Over Time?

**YES** - Results change based on:

| Factor | How it affects results |
|--------|----------------------|
| **New signals** | Each scrape adds new signals |
| **Signal age** | Older signals have less weight in growth calculation |
| **Growth rate** | Calculated from recent vs. older signals |
| **Trend momentum** | Based on signal publication dates |

**NO** - These are deterministic:
- Classification of individual signals (same input → same output)
- Keyword matching rules
- Scoring formula weights

---

## 7. Scoring Algorithm

### No-Regret Priority Score Formula:

**Location:** `services/trend_scorer.py`

```
Priority Score = (SR × 0.40 + GS × 0.30 + CE × 0.30) × Project_Multiplier

Where:
  SR = Strategic Relevance (0-100)
  GS = Grid Stability Impact (0-100)
  CE = Cost Efficiency Impact (0-100)
  Project_Multiplier = 1.0 to 1.5 (if linked to mega-project)
```

### Component Breakdown:

#### 1. Strategic Relevance (SR) - 40% weight

```python
def calculate_strategic_relevance():
    score = 50  # Base score
    
    # Keyword analysis (+/- 20 points)
    score += high_relevance_keywords * 5  # max +20
    score += medium_keywords * 2          # max +10
    score -= low_relevance_keywords * 3   # max -15
    
    # TSO Category weight (+15 points max)
    category_weights = {
        "grid_infrastructure": +15,
        "grid_stability": +15,
        "cybersecurity_ot": +15,  # BOOSTED
        "offshore_systems": +15,
        "energy_storage": +12,
        "flexibility_vpp": +12,   # BOOSTED
        "ai_grid_optimization": +10,
        ...
    }
    score += category_weights[category]
    
    # Amprion Task alignment (+10 points)
    if task in ["grid_expansion", "system_security"]:
        score += 10
    
    # Mega-project linkage (+15 points max)
    score += len(linked_projects) * 5
    
    return min(max(score, 0), 100)
```

#### 2. Grid Stability Impact (GS) - 30% weight

```python
def calculate_grid_stability_impact():
    score = 40  # Base score
    
    # Critical stability keywords (+30 max)
    critical_keywords = ["frequency", "inertia", "blackout", "grid-forming",
                        "cyber attack", "scada", "kritis"]  # CYBER ADDED
    score += critical_count * 6
    
    # Category stability relevance (+15 max)
    stability_categories = {
        "grid_stability": +15,
        "cybersecurity_ot": +15,  # ADDED - cyber = stability risk
        "grid_infrastructure": +12,
        "energy_storage": +10,
        "flexibility_vpp": +10,
        "offshore_systems": +8
    }
    
    return min(max(score, 0), 100)
```

#### 3. Cost Efficiency Impact (CE) - 30% weight

```python
def calculate_cost_efficiency_impact():
    score = 60  # Base score
    
    # Positive keywords (cost reduction)
    positive = ["cost reduction", "efficiency", "automated", "savings"]
    score += positive_count * 3
    
    # Negative keywords (cost increase)
    negative = ["expensive", "complex", "specialized"]
    score -= negative_count * 3
    
    # Volume bonus (more signals = more validation)
    score += min(signal_count * 0.5, 15)
    
    # Quality bonus
    score += avg_quality_score * 20
    
    return min(max(score, 0), 100)
```

#### 4. Project Multiplier

```python
PROJECT_MULTIPLIERS = {
    "Korridor B": 1.3,      # Highest priority project
    "Rhein-Main Link": 1.25,
    "A-Nord": 1.2,
    "Ultranet": 1.2,
    "BalWin1": 1.15,
    "BalWin2": 1.15,
    "NeuLink": 1.15,
    "hybridge": 1.1,        # Hydrogen project
    "Systemmarkt": 1.1,     # Flexibility platform
    "Grid Boosters": 1.3,   # Critical for stability
    "Auto-Trader": 1.1      # AI trading
}
```

### Example Score Calculation:

```
Signal: "Grid-forming inverters for Korridor B enhance frequency stability"

Strategic Relevance:
  Base: 50
  + "grid-forming" (high): +5
  + "frequency" (high): +5
  + "korridor" (high): +5
  + Category (grid_stability): +15
  + Task (system_security): +10
  + Project (Korridor B): +5
  = 95/100

Grid Stability:
  Base: 40
  + "frequency" (critical): +6
  + "grid-forming" (critical): +6
  + "stability" (high): +3
  + Category (grid_stability): +15
  = 70/100 → capped to 100

Cost Efficiency:
  Base: 60
  + Signal count (15): +7.5
  + Quality (0.85): +17
  = 84.5/100

Final Score:
  (95 × 0.40 + 100 × 0.30 + 84.5 × 0.30) × 1.3 (Korridor B multiplier)
  = (38 + 30 + 25.35) × 1.3
  = 93.35 × 1.3
  = 121.4 → capped to 10.0

Priority: 10.0 (maximum)
```

---

## 8. API Endpoints

### Core Endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/dashboard` | GET | System statistics |
| `/api/v1/signals` | GET | List all signals |
| `/api/v1/signals/{id}` | GET | Single signal |
| `/api/v1/trends` | GET | List all trends |
| `/api/v1/trends/{id}` | GET | Single trend with details |
| `/api/v1/trends/create` | POST | Trigger trend creation |
| `/api/v1/trends/score` | POST | Recalculate all scores |

### What Does `/api/v1/trends/create` Do?

**Location:** `main.py` line ~380

```python
@app.post("/api/v1/trends/create")
async def create_trends(min_signals: int = 3):
    """
    Create trends from processed signals.
    
    Steps:
    1. Get all processed signals
    2. Group by TSO category
    3. Skip categories with < min_signals
    4. Create TrendCluster for each category
    5. Create Trend with:
       - Aggregated metadata (domain, layer, task)
       - Calculated scores
       - Signal count and growth rate
    6. Return created trends
    """
```

**When to use:**
- After adding new signals
- After re-processing signals
- To regenerate trends from scratch

### Filtering & Querying Endpoints:

```bash
# Filter by category
GET /api/v1/trends?category=cybersecurity_ot

# Filter by strategic domain
GET /api/v1/trends?strategic_domain=system_operation

# Filter by minimum priority
GET /api/v1/trends?min_priority=8.0

# Filter by time to impact
GET /api/v1/trends?time_to_impact=1-3%20years

# Sort by different fields
GET /api/v1/trends?sort_by=growth_rate&sort_order=desc

# Pagination
GET /api/v1/signals?skip=0&limit=50
```

### TSO Impact Analysis Endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/impact/signal/{id}` | Impact analysis for single signal |
| `GET /api/v1/impact/trend/{id}` | Full impact report for trend |
| `GET /api/v1/impact/category/{cat}` | Impact rules for category |
| `GET /api/v1/impact/all-categories` | Summary of all impacts |

### Stakeholder View Endpoints:

```bash
# Executive Board view (high-level)
GET /api/v1/stakeholder-views/executive

# IT Architecture view (technical)
GET /api/v1/stakeholder-views/it_architecture

# Grid Planning view (infrastructure)
GET /api/v1/stakeholder-views/grid_planning

# Sustainability view (ESG)
GET /api/v1/stakeholder-views/sustainability
```

---

## 9. Time-Based Filtering

### How Time Affects Results:

#### Growth Rate Calculation:

```python
def calculate_growth_rate(signals):
    # Split signals into time periods
    recent = [s for s in signals if s.published_at > (now - 30 days)]
    older = [s for s in signals if s.published_at <= (now - 30 days)]
    
    if older:
        growth_rate = (len(recent) / len(older) - 1) * 100
    else:
        growth_rate = 100.0  # All signals are recent
    
    return growth_rate
```

#### Time-to-Impact Classification:

```python
TIME_TO_IMPACT_RULES = {
    "<1 year": ["deployed", "operational", "launched", "production"],
    "1-3 years": ["pilot", "demonstration", "field test", "scaling"],
    "3-5 years": ["prototype", "pre-commercial", "regulatory approval"],
    "5+ years": ["research", "concept", "experimental", "laboratory"]
}
```

### 🔧 HOW TO FILTER BY TIME:

**Via API:**
```bash
# Get signals from last 7 days
GET /api/v1/signals?days_back=7

# Get signals in date range (if implemented)
GET /api/v1/signals?from_date=2026-01-01&to_date=2026-01-31
```

**Via Code (in full.py):**
```python
# Filter signals by date before processing
from datetime import datetime, timedelta

cutoff_date = datetime.utcnow() - timedelta(days=30)
recent_signals = [s for s in signals if s.published_at >= cutoff_date]
```

### Current Behavior:

| Aspect | Current Implementation |
|--------|----------------------|
| Signal scraping | Gets latest from each source |
| arXiv | Last ~30 days of papers |
| RSS feeds | Last ~50 items per feed |
| Growth calculation | 30-day rolling window |
| Trend scoring | Includes all signals in category |

### To Add Custom Time Filtering:

```python
# In main.py, modify get_signals endpoint:

@app.get("/api/v1/signals")
async def get_signals(
    days_back: Optional[int] = None,  # ADD THIS
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    ...
):
    query = db.query(Signal)
    
    # Time filtering
    if days_back:
        cutoff = datetime.utcnow() - timedelta(days=days_back)
        query = query.filter(Signal.published_at >= cutoff)
    
    if from_date:
        query = query.filter(Signal.published_at >= from_date)
    
    if to_date:
        query = query.filter(Signal.published_at <= to_date)
```

---

## 10. Configuration & Customization

### Key Configuration Files:

#### `config.py` - Main Settings

```python
# Database
DATABASE_URL = "sqlite:///./sonar_ai.db"

# Scoring weights (MODIFY THESE to change priorities)
PRIORITY_SCORING_WEIGHTS = {
    "strategic_relevance": 0.40,  # 40% weight
    "grid_stability": 0.30,       # 30% weight
    "cost_efficiency": 0.30       # 30% weight
}

# Project multipliers
PROJECT_CRITICALITY_MULTIPLIERS = {
    "Korridor B": 1.3,
    "Grid Boosters": 1.3,
    ...
}

# TSO Categories (add/modify categories here)
TSO_CATEGORIES = {...}

# Strategic Domains
STRATEGIC_DOMAINS = {...}

# Enlit Europe mapping
ENLIT_TSO_MAPPING = {...}
```

#### Environment Variables (optional):

```bash
# Create .env file:
DATABASE_URL=sqlite:///./sonar_ai.db
OPENAI_API_KEY=sk-...  # For AI summaries (future)
DEBUG=true
```

### Customization Examples:

#### Change Scoring Weights:

```python
# In config.py:
PRIORITY_SCORING_WEIGHTS = {
    "strategic_relevance": 0.50,  # Increase strategic weight
    "grid_stability": 0.35,       # Increase stability weight
    "cost_efficiency": 0.15       # Decrease cost weight
}
```

#### Add Custom Keywords:

```python
# In config.py, find the category and add keywords:
TSO_CATEGORIES["cybersecurity_ot"]["keywords"].extend([
    "zero trust",
    "supply chain attack",
    "firmware vulnerability"
])
```

#### Change arXiv Search Topics:

```python
# In services/scrapers.py, find ARXIV_QUERIES:
ARXIV_QUERIES = [
    # Existing queries...
    
    # Add new topics:
    "offshore wind grid connection",
    "battery management system",
    "hydrogen pipeline"
]
```

---

## 11. Next Steps & Roadmap

### Immediate (Before Demo Day - Feb 4):

| Task | Priority | Effort |
|------|----------|--------|
| ✅ Fix classification | DONE | - |
| ✅ Fix scoring | DONE | - |
| Add more RSS sources | HIGH | 30 min |
| Test API endpoints | HIGH | 1 hour |
| Create demo script | MEDIUM | 1 hour |

### Short-Term (Week 1-2 Post-Demo):

| Task | Priority | Description |
|------|----------|-------------|
| Frontend dashboard | HIGH | React/Vue UI for visualization |
| Add offshore wind sources | HIGH | Fill category gap |
| Add biogas sources | MEDIUM | Fill category gap |
| Email/Slack alerts | MEDIUM | Notify on new high-priority trends |
| Scheduled scraping | MEDIUM | Cron job for daily updates |

### Medium-Term (Month 1-2):

| Task | Priority | Description |
|------|----------|-------------|
| AI summaries | HIGH | GPT/Claude for trend summaries |
| User authentication | MEDIUM | Multi-user support |
| PostgreSQL migration | MEDIUM | Better for production |
| Historical analysis | MEDIUM | Track trend evolution over time |
| Export features | LOW | PDF/Excel reports |

### Long-Term (Month 3+):

| Task | Description |
|------|-------------|
| Predictive scoring | ML model to predict trend importance |
| Sentiment analysis | Positive/negative trend sentiment |
| Entity resolution | Link signals to specific companies/projects |
| Multi-language | German sources support |
| API integrations | Connect to internal Amprion systems |

---

## Quick Reference Commands

```bash
# Full pipeline
python full.py

# Individual steps
python full.py --scrape-only
python full.py --process-only
python full.py --cluster-only
python full.py --score-only
python full.py --report

# Start API server
uvicorn main:app --reload

# Reset database
rm sonar_ai.db && python full.py

# Test classification
python -c "
from services.nlp_processor import get_nlp_processor
p = get_nlp_processor()
result = p.classify_signal('Your test headline here')
print(result['tso_category'], result['tso_category_confidence'])
"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Empty API response | Restart server after running full.py |
| arXiv errors | `pip install lxml` |
| Import errors | `pip install -r requirements.txt` |
| Database locked | Close other connections, delete .db file |
| Low cybersecurity score | Check config.py weights |
| Wrong classification | Add keywords to config.py |

---

## Contact & Resources

- **AGTS Framework**: Pages 6-22 of original document
- **Enlit Europe Categories**: https://www.enlit-europe.com/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **SQLAlchemy Docs**: https://docs.sqlalchemy.org/

---

*Last Updated: January 31, 2026*
*Version: 2.0*
