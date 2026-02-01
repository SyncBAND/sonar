# SONAR.AI / AGTS — Demo Guide

## What It Does (Elevator Pitch)

SONAR.AI is an **Agentic Global Trend Scanner** built for Amprion (a German TSO — Transmission System Operator). It automatically monitors 35 RSS feeds + arXiv research + regulatory scrapers, classifies every signal against a 16-category energy taxonomy, and produces a **priority-ranked dashboard** of technology trends scored using a Bayesian multi-criteria framework calibrated to Amprion's strategic priorities.

Think of it as an AI-powered radar for the energy transition — it tells you what's accelerating, what's critical, and where your blind spots are.

---

## The 5-Step Pipeline

Run everything: `python full.py`
Or run steps individually:

### Step 1: Scrape → `python full.py --scrape-only`

Pulls signals from 35+ sources in parallel:

| Source Type | Count | Examples |
|---|---|---|
| RSS news feeds | 35 | Utility Dive, PV Magazine, Dark Reading, Electrek |
| arXiv research | 24 queries | "grid-forming inverter", "SCADA cybersecurity", "green hydrogen" |
| TSO publications | 4 scrapers | ENTSO-E, Amprion, TenneT, 50Hertz news via Google News |
| Regulatory bodies | 1 scraper | BNetzA, EU Commission, ACER via Google News |
| Industry events | 2 scrapers | Enlit World, energy conference agendas |

Each signal gets: title, content, URL, source_name, source_type, published_at, scraped_at.

**Where to add new RSS feeds:** `services/scrapers.py`, line ~195 — the `RSS_FEEDS` dictionary inside the `RSSNewsScraper` class. Just add a new line:

```python
"your_feed_id": "https://example.com/feed/",
```

Also add a category hint in `main.py`, line ~1445 — the `_source_category_hint()` function:

```python
"your_feed_id": "grid_stability",  # or whatever category
```

The category hint is a fallback if the classifier is uncertain. The classifier will override it if it's confident.

### Step 2: Process → `python full.py --process-only`

Runs the **Semantic Classifier** (`services/classifier.py`) on every unprocessed signal:

1. Encodes the title+content into a 384-dimensional vector using a SentenceTransformer model
2. Compares it against 16 category centroids (pre-computed from curated seed texts)
3. Applies keyword boost for anchor terms (e.g. "HVDC" → offshore_systems, "SCADA" → cybersecurity_ot)
4. Checks domain relevance — how close is this to the "energy/grid" domain overall
5. If domain_relevance < threshold AND no anchor keywords → classified as `off_topic`

Output per signal: `tso_category`, `strategic_domain`, `architectural_layer`, `amprion_task`, `quality_score`, keywords.

In the last run: 1066 signals processed, 154 flagged off-topic (14.4%).

### Step 3: Cluster → `python full.py --cluster-only`

Groups signals by their `tso_category` into Trends. Currently 1 trend per category = 16 trends. Each trend becomes a database entry (Trend model) with aggregated metadata.

### Step 4: Score → `python full.py --score-only`

This is the core algorithm. Each trend gets a **priority score (0–10)** computed by the `TrendScorer` (`services/trend_scorer.py`):

```
Priority = base(5.0) + strategic_importance + evidence_strength + growth_momentum + maturity_readiness + project_bonus
```

The five scoring dimensions:

| Dimension | Weight | What It Measures | Source |
|---|---|---|---|
| **Strategic Importance** | 35% | How critical is this to Amprion? | Fixed priors from Amprion framework (weight 50–95) |
| **Evidence Strength** | 25% | How much evidence do we have? | Signal count, source diversity, quality scores |
| **Growth Momentum** | 20% | Is signal volume accelerating? | Ratio of recent vs. older signals (time-series) |
| **Maturity Readiness** | 20% | How deployment-ready is the technology? | TRL computed from source types + title keywords |
| **Project Bonus** | additive | Links to Amprion mega-projects? | SuedLink, BalWin1/2, Grid Boosters, hybridge, etc. |

**Amprion Strategic Priors** (the key differentiator):

| Tier | Categories | Weight | Meaning |
|---|---|---|---|
| **EXISTENTIAL** | Grid Stability, Cybersecurity | 95 | Failure is unacceptable for a TSO |
| **CRITICAL** | Grid Infrastructure, Offshore, Flexibility/VPP | 88–90 | Core mission, actively planned |
| **HIGH** | Renewables, Storage, E-Mobility, AI | 72–80 | Strategically important |
| **MEDIUM-HIGH** | Hydrogen, Regulatory, Trading | 65–70 | Important but less immediate |
| **MEDIUM** | Distributed Gen, Digital Twins, Biogas, Power Gen | 50–60 | Watch and track |

This means Cybersecurity (EXISTENTIAL, weight 95) will always score higher than Biogas (MEDIUM, weight 55) even if Biogas has more signals — because the organizational prior says cybersecurity matters more for a TSO.

**Cold-Start Protection:** On the first run, all signals arrive at once (batch scrape), so there's no real time-series to measure growth. The scorer detects this (scrape window < 2 days) and caps growth contribution to a neutral range (≤65/100) instead of inflating it to 95. Growth will differentiate naturally on the second run after 7+ days.

**Monitoring Alerts:** Tier-aware blind-spot detection:
- EXISTENTIAL/CRITICAL categories: alert if < 15 signals OR evidence < 50
- HIGH categories: alert if < 10 signals AND evidence < 40

### Step 5: Report → `python full.py --report`

Generates a console report with:
- Full 16-trend ranking (score, TRL, lifecycle, nature, impact, signal count, tier)
- ⚠ Blind spot alerts for under-covered critical categories
- Amprion Framework cross-reference (all tiers mapped)
- Breakdown by strategic nature, time-to-impact, lifecycle stage
- Score diagnostics (range, uniqueness, TRL spread)

---

## The API (FastAPI)

Start: `python main.py` (runs on `localhost:8000`)

Key endpoints:

| Endpoint | What It Does |
|---|---|
| `GET /api/v1/dashboard` | Summary stats + top trends |
| `GET /api/v1/trends` | All 16 trends with filters |
| `GET /api/v1/trends/{id}` | Single trend with all signals |
| `POST /api/v1/trends/score` | Re-score all trends |
| `GET /api/v1/explain/trend/{id}` | **SHAP waterfall** — shows exactly why a trend scored what it did |
| `GET /api/v1/explain/compare?n=5` | Side-by-side SHAP comparison of top N trends |
| `GET /api/v1/signals` | Browse all signals with filters |
| `GET /api/v1/alerts` | Monitoring alerts |
| `GET /api/v1/sources` | All configured sources and their status |
| `GET /api/v1/taxonomies` | Full taxonomy (categories, domains, layers, business areas) |
| `GET /api/v1/stakeholder-views/{type}` | Filtered view for executive_board, it_architecture, grid_planning, sustainability |

---

## SHAP Explainability

Every score is fully decomposable. The `/api/v1/explain/trend/{id}` endpoint returns a waterfall chart showing exactly how each factor contributed:

```
base_value:            5.00  ─────────── starting point
+ strategic_importance: +1.58 ─────────── EXISTENTIAL tier (weight 95)
+ evidence_strength:    +0.82 ─────────── 169 signals, 14 sources
+ growth_momentum:      +0.25 ─────────── cold-start damped
+ maturity_readiness:   +0.11 ─────────── TRL 5 (Pilot stage)
+ project_bonus:        +0.15 ─────────── partial project linkage
= priority_score:       7.90  ─────────── Cybersecurity & OT Security
```

This is critical for demo credibility — you can explain *why* any trend ranks where it does, and it's not a black box.

---

## Key Files

| File | Purpose |
|---|---|
| `full.py` | CLI pipeline (scrape → process → cluster → score → report) |
| `main.py` | FastAPI REST API (all endpoints) |
| `services/scrapers.py` | **← ADD RSS FEEDS HERE** (line ~195, `RSS_FEEDS` dict) |
| `services/classifier.py` | Semantic classifier (SentenceTransformer + keyword boost) |
| `services/trend_scorer.py` | Scoring algorithm (Bayesian MCDA + Amprion priors) |
| `models/database.py` | SQLAlchemy models (Signal, Trend, Alert) |
| `taxonomy.py` | Full Amprion taxonomy definition |
| `config.py` | Database path, logging, feature flags |
| `test_scorer.py` | 10-test validation suite for scoring algorithm |

---

## Quick Reference: Adding New Sources

### Adding an RSS feed

1. Open `services/scrapers.py`
2. Find the `RSS_FEEDS = {` dictionary (line ~195)
3. Add your feed under the appropriate section comment:

```python
# ─── GRID STABILITY (need more coverage!) ───────────────
"entso_e_news": "https://www.entsoe.eu/news/feed/",  # 🆕
"ieee_pes": "https://resourcecenter.ieee-pes.org/feed",  # 🆕
```

4. Optionally add a category hint in `main.py` → `_source_category_hint()` (line ~1445):

```python
"entso_e_news": "grid_stability",
"ieee_pes": "grid_stability",
```

5. Delete `sonar_ai.db` and re-run: `python full.py`

### Adding arXiv search queries

Open `services/scrapers.py`, find the `ArxivScraper` class (line ~71), and add queries to the `QUERIES` list.

### What happens after adding feeds

The classifier will automatically categorize new signals. If a feed is well-targeted (e.g. a grid stability journal), most signals will land in the right category. The scorer will then incorporate them into evidence strength, and blind-spot alerts will update accordingly.

---

## Current Production Results (Jan 31, 2026)

- **1066 signals** from 35 RSS + arXiv + regulatory scrapers
- **154 off-topic** filtered (14.4%)
- **16 trends** scored 6.0–8.4 (healthy spread, no compression)
- **2 blind spots** detected: Grid Stability (6 signals), Flexibility/VPP (10 signals)
- **Top 3**: Renewables Integration (8.4), Grid Infrastructure (8.3), Offshore Systems (8.2)
- Growth: currently uniform at 62.5 (cold-start). Will differentiate on run 2.
