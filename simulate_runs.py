#!/usr/bin/env python3
"""
SONAR.AI Run Simulator
======================
Simulates what the database would look like after N weekly scans.
Updates trends with realistic lifecycle progression, TRL advancement,
growth rates, signal accumulation, and re-scored priorities.

Usage:
    python simulate_runs.py --runs 4     # After 1 month
    python simulate_runs.py --runs 10    # After ~2.5 months
    python simulate_runs.py --runs 15    # After ~4 months
    python simulate_runs.py --runs 20    # After 5 months

Requires: An existing sonar_ai.db with trends (run a Full Scan first).
"""

import argparse
import random
import sys
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import our models
from models.database import Base, Signal, Trend, TrendCluster, Alert

# ─────────────────────────────────────────────────────────────────────────────
# TREND EVOLUTION PROFILES
# ─────────────────────────────────────────────────────────────────────────────
# Each trend has a realistic trajectory over 20 runs based on real-world dynamics.
# Format per run milestone: {trl, lifecycle, growth_rate, signal_boost, nature, time_to_impact}

PROFILES = {
    "grid_infrastructure": {
        "label": "Grid Infrastructure & Transmission",
        # Already mature, steady — Amprion's core business
        4:  {"trl": 7, "lifecycle": "Evaluating",  "growth": 45,  "sig_boost": 60,  "nature": "Accelerator", "tti": "<1 year"},
        10: {"trl": 8, "lifecycle": "Piloting",     "growth": 30,  "sig_boost": 180, "nature": "Accelerator", "tti": "<1 year"},
        15: {"trl": 8, "lifecycle": "Scaling",       "growth": 15,  "sig_boost": 300, "nature": "Accelerator", "tti": "<1 year"},
        20: {"trl": 9, "lifecycle": "Deployed",      "growth": 8,   "sig_boost": 420, "nature": "Accelerator", "tti": "<1 year"},
    },
    "offshore_systems": {
        "label": "Offshore Grid Systems",
        # Growing fast — DolWin/BorWin projects driving activity
        4:  {"trl": 7, "lifecycle": "Evaluating",  "growth": 120, "sig_boost": 80,  "nature": "Accelerator", "tti": "1-3 years"},
        10: {"trl": 8, "lifecycle": "Piloting",     "growth": 85,  "sig_boost": 250, "nature": "Accelerator", "tti": "<1 year"},
        15: {"trl": 8, "lifecycle": "Scaling",       "growth": 40,  "sig_boost": 380, "nature": "Accelerator", "tti": "<1 year"},
        20: {"trl": 9, "lifecycle": "Deployed",      "growth": 12,  "sig_boost": 480, "nature": "Accelerator", "tti": "<1 year"},
    },
    "renewables_integration": {
        "label": "Renewables Integration",
        # Massive signal volume, steady maturation
        4:  {"trl": 6, "lifecycle": "Watching",    "growth": 55,  "sig_boost": 120, "nature": "Transformational", "tti": "<1 year"},
        10: {"trl": 7, "lifecycle": "Evaluating",  "growth": 70,  "sig_boost": 350, "nature": "Transformational", "tti": "<1 year"},
        15: {"trl": 8, "lifecycle": "Piloting",     "growth": 35,  "sig_boost": 500, "nature": "Transformational", "tti": "<1 year"},
        20: {"trl": 8, "lifecycle": "Scaling",       "growth": 20,  "sig_boost": 650, "nature": "Transformational", "tti": "<1 year"},
    },
    "cybersecurity_ot": {
        "label": "Cybersecurity & OT Security",
        # High urgency, rapid escalation after incidents
        4:  {"trl": 6, "lifecycle": "Watching",    "growth": 90,  "sig_boost": 200, "nature": "Accelerator", "tti": "<1 year"},
        10: {"trl": 7, "lifecycle": "Evaluating",  "growth": 150, "sig_boost": 600, "nature": "Accelerator", "tti": "<1 year"},
        15: {"trl": 7, "lifecycle": "Piloting",     "growth": 60,  "sig_boost": 900, "nature": "Disruptor",   "tti": "<1 year"},
        20: {"trl": 8, "lifecycle": "Scaling",       "growth": 25,  "sig_boost": 1100,"nature": "Disruptor",   "tti": "<1 year"},
    },
    "energy_storage": {
        "label": "Energy Storage Systems",
        # Hype cycle — explosive growth then stabilization
        4:  {"trl": 5, "lifecycle": "Watching",    "growth": 180, "sig_boost": 100, "nature": "Accelerator", "tti": "1-3 years"},
        10: {"trl": 6, "lifecycle": "Evaluating",  "growth": 250, "sig_boost": 400, "nature": "Transformational", "tti": "1-3 years"},
        15: {"trl": 7, "lifecycle": "Evaluating",  "growth": 80,  "sig_boost": 550, "nature": "Transformational", "tti": "<1 year"},
        20: {"trl": 7, "lifecycle": "Piloting",     "growth": 35,  "sig_boost": 680, "nature": "Transformational", "tti": "<1 year"},
    },
    "grid_stability": {
        "label": "Grid Stability & System Services",
        # Starts as blind spot, dramatically improves with better sources
        4:  {"trl": 7, "lifecycle": "Watching",    "growth": 300, "sig_boost": 25,  "nature": "Accelerator", "tti": "<1 year"},
        10: {"trl": 8, "lifecycle": "Evaluating",  "growth": 120, "sig_boost": 90,  "nature": "Accelerator", "tti": "<1 year"},
        15: {"trl": 8, "lifecycle": "Piloting",     "growth": 40,  "sig_boost": 160, "nature": "Accelerator", "tti": "<1 year"},
        20: {"trl": 9, "lifecycle": "Scaling",       "growth": 10,  "sig_boost": 220, "nature": "Accelerator", "tti": "<1 year"},
    },
    "flexibility_vpp": {
        "label": "Flexibility & Virtual Power Plants",
        # Disruptive — slow start then rapid acceleration
        4:  {"trl": 4, "lifecycle": "Watching",    "growth": 200, "sig_boost": 20,  "nature": "Disruptor", "tti": "3-5 years"},
        10: {"trl": 5, "lifecycle": "Evaluating",  "growth": 350, "sig_boost": 110, "nature": "Disruptor", "tti": "1-3 years"},
        15: {"trl": 6, "lifecycle": "Piloting",     "growth": 150, "sig_boost": 250, "nature": "Disruptor", "tti": "1-3 years"},
        20: {"trl": 7, "lifecycle": "Piloting",     "growth": 70,  "sig_boost": 380, "nature": "Disruptor", "tti": "<1 year"},
    },
    "energy_trading": {
        "label": "Energy Trading & Markets",
        # Steady, regulated — slow evolution
        4:  {"trl": 7, "lifecycle": "Watching",    "growth": 25,  "sig_boost": 50,  "nature": "Transformational", "tti": "<1 year"},
        10: {"trl": 8, "lifecycle": "Evaluating",  "growth": 15,  "sig_boost": 120, "nature": "Transformational", "tti": "<1 year"},
        15: {"trl": 8, "lifecycle": "Piloting",     "growth": 10,  "sig_boost": 180, "nature": "Transformational", "tti": "<1 year"},
        20: {"trl": 9, "lifecycle": "Deployed",      "growth": 5,   "sig_boost": 230, "nature": "Transformational", "tti": "<1 year"},
    },
    "regulatory_policy": {
        "label": "Regulatory & Policy",
        # Spiky — jumps when new regulations drop
        4:  {"trl": 6, "lifecycle": "Watching",    "growth": 40,  "sig_boost": 80,  "nature": "Accelerator", "tti": "<1 year"},
        10: {"trl": 7, "lifecycle": "Evaluating",  "growth": 180, "sig_boost": 300, "nature": "Accelerator", "tti": "<1 year"},
        15: {"trl": 7, "lifecycle": "Evaluating",  "growth": 60,  "sig_boost": 420, "nature": "Accelerator", "tti": "<1 year"},
        20: {"trl": 8, "lifecycle": "Piloting",     "growth": 20,  "sig_boost": 520, "nature": "Accelerator", "tti": "<1 year"},
    },
    "hydrogen_p2g": {
        "label": "Hydrogen & Power-to-Gas",
        # Long-term play — lots of research, slow maturation
        4:  {"trl": 3, "lifecycle": "Watching",    "growth": 65,  "sig_boost": 80,  "nature": "Transformational", "tti": "3-5 years"},
        10: {"trl": 4, "lifecycle": "Evaluating",  "growth": 95,  "sig_boost": 250, "nature": "Transformational", "tti": "3-5 years"},
        15: {"trl": 5, "lifecycle": "Evaluating",  "growth": 50,  "sig_boost": 400, "nature": "Transformational", "tti": "1-3 years"},
        20: {"trl": 6, "lifecycle": "Piloting",     "growth": 40,  "sig_boost": 520, "nature": "Transformational", "tti": "1-3 years"},
    },
    "e_mobility_v2g": {
        "label": "E-Mobility & Vehicle-to-Grid",
        # Disruptive — consumer-driven, hard to predict
        4:  {"trl": 4, "lifecycle": "Watching",    "growth": 110, "sig_boost": 100, "nature": "Disruptor", "tti": "3-5 years"},
        10: {"trl": 5, "lifecycle": "Evaluating",  "growth": 200, "sig_boost": 350, "nature": "Disruptor", "tti": "1-3 years"},
        15: {"trl": 6, "lifecycle": "Piloting",     "growth": 90,  "sig_boost": 500, "nature": "Disruptor", "tti": "1-3 years"},
        20: {"trl": 7, "lifecycle": "Piloting",     "growth": 45,  "sig_boost": 650, "nature": "Disruptor", "tti": "<1 year"},
    },
    "power_generation": {
        "label": "Power Generation",
        # Mature, declining — legacy tech
        4:  {"trl": 8, "lifecycle": "Watching",    "growth": -5,  "sig_boost": 20,  "nature": "Accelerator", "tti": "<1 year"},
        10: {"trl": 9, "lifecycle": "Evaluating",  "growth": -15, "sig_boost": 30,  "nature": "Accelerator", "tti": "<1 year"},
        15: {"trl": 9, "lifecycle": "Deployed",      "growth": -20, "sig_boost": 35,  "nature": "Accelerator", "tti": "<1 year"},
        20: {"trl": 9, "lifecycle": "Deployed",      "growth": -25, "sig_boost": 38,  "nature": "Accelerator", "tti": "<1 year"},
    },
    "ai_grid_optimization": {
        "label": "AI & Grid Optimization",
        # High buzz, slow to mature — classic hype cycle
        4:  {"trl": 3, "lifecycle": "Watching",    "growth": 320, "sig_boost": 200, "nature": "Transformational", "tti": "5+ years"},
        10: {"trl": 4, "lifecycle": "Evaluating",  "growth": 180, "sig_boost": 600, "nature": "Transformational", "tti": "3-5 years"},
        15: {"trl": 5, "lifecycle": "Evaluating",  "growth": 50,  "sig_boost": 850, "nature": "Transformational", "tti": "3-5 years"},
        20: {"trl": 6, "lifecycle": "Piloting",     "growth": 30,  "sig_boost": 1000,"nature": "Transformational", "tti": "1-3 years"},
    },
    "distributed_generation": {
        "label": "Distributed Generation",
        # Slow burn — regulatory dependent
        4:  {"trl": 5, "lifecycle": "Watching",    "growth": 35,  "sig_boost": 15,  "nature": "Disruptor", "tti": "1-3 years"},
        10: {"trl": 6, "lifecycle": "Evaluating",  "growth": 55,  "sig_boost": 50,  "nature": "Disruptor", "tti": "1-3 years"},
        15: {"trl": 6, "lifecycle": "Evaluating",  "growth": 30,  "sig_boost": 85,  "nature": "Disruptor", "tti": "1-3 years"},
        20: {"trl": 7, "lifecycle": "Piloting",     "growth": 20,  "sig_boost": 120, "nature": "Disruptor", "tti": "<1 year"},
    },
    "digital_twin_simulation": {
        "label": "Digital Twins & Simulation",
        # Niche but growing — Amprion exploring
        4:  {"trl": 4, "lifecycle": "Watching",    "growth": 75,  "sig_boost": 30,  "nature": "Transformational", "tti": "3-5 years"},
        10: {"trl": 5, "lifecycle": "Evaluating",  "growth": 100, "sig_boost": 100, "nature": "Transformational", "tti": "3-5 years"},
        15: {"trl": 6, "lifecycle": "Piloting",     "growth": 60,  "sig_boost": 180, "nature": "Transformational", "tti": "1-3 years"},
        20: {"trl": 7, "lifecycle": "Piloting",     "growth": 35,  "sig_boost": 250, "nature": "Transformational", "tti": "1-3 years"},
    },
    "biogas_biomethane": {
        "label": "Biogas & Biomethane",
        # Low priority, steady — doesn't change much
        4:  {"trl": 7, "lifecycle": "Watching",    "growth": 10,  "sig_boost": 8,   "nature": "Transformational", "tti": "1-3 years"},
        10: {"trl": 7, "lifecycle": "Watching",    "growth": 5,   "sig_boost": 15,  "nature": "Transformational", "tti": "1-3 years"},
        15: {"trl": 8, "lifecycle": "Evaluating",  "growth": 15,  "sig_boost": 25,  "nature": "Transformational", "tti": "1-3 years"},
        20: {"trl": 8, "lifecycle": "Evaluating",  "growth": 8,   "sig_boost": 35,  "nature": "Transformational", "tti": "1-3 years"},
    },
}


def get_profile_at_run(category: str, run: int) -> dict:
    """Interpolate profile values for any run number."""
    profile = PROFILES.get(category)
    if not profile:
        return None

    milestones = sorted([k for k in profile.keys() if isinstance(k, int)])

    # Find bracketing milestones
    if run <= milestones[0]:
        return profile[milestones[0]]
    if run >= milestones[-1]:
        return profile[milestones[-1]]

    for i in range(len(milestones) - 1):
        lo, hi = milestones[i], milestones[i + 1]
        if lo <= run <= hi:
            t = (run - lo) / (hi - lo)  # 0.0 to 1.0
            lo_p, hi_p = profile[lo], profile[hi]
            return {
                "trl": lo_p["trl"] if t < 0.7 else hi_p["trl"],
                "lifecycle": lo_p["lifecycle"] if t < 0.5 else hi_p["lifecycle"],
                "growth": round(lo_p["growth"] + t * (hi_p["growth"] - lo_p["growth"])),
                "sig_boost": round(lo_p["sig_boost"] + t * (hi_p["sig_boost"] - lo_p["sig_boost"])),
                "nature": lo_p["nature"] if t < 0.5 else hi_p["nature"],
                "tti": lo_p["tti"] if t < 0.5 else hi_p["tti"],
            }
    return profile[milestones[-1]]


def rescore_trend(trend, sig_count: int, growth_rate: float):
    """
    Re-calculate priority score using the same Bayesian MCDA formula as trend_scorer.py.
    This is a simplified inline version to avoid importing the full scorer.
    """
    from config import TSO_CATEGORIES

    cat_config = TSO_CATEGORIES.get(trend.tso_category, {})
    tier = cat_config.get("tier", "medium")
    weight = cat_config.get("strategic_weight", 50)
    linked_projects = trend.linked_projects or cat_config.get("default_projects", [])

    BASE = 5.0

    # Strategic importance (35%)
    sr_raw = weight / 100.0
    sr_c = sr_raw * 0.35 * (10 - BASE)

    # Evidence strength (25%)
    vol = min(sig_count / 100, 1.0)
    src_div = min(sig_count / 50, 1.0) * 0.8 + 0.2
    ev_raw = 0.6 * vol + 0.4 * src_div
    ev_c = ev_raw * 0.25 * (10 - BASE)

    # Growth momentum (20%)
    if growth_rate > 100:
        gr_raw = 1.0
    elif growth_rate > 0:
        gr_raw = min(growth_rate / 100, 1.0)
    elif growth_rate == 0:
        gr_raw = 0.5
    else:
        gr_raw = max(0, 0.5 + growth_rate / 200)
    gr_c = gr_raw * 0.20 * (10 - BASE)

    # Maturity readiness (20%)
    trl = trend.maturity_score or 5
    if trl <= 3:
        mt_raw = 0.2
    elif trl <= 5:
        mt_raw = 0.5 + (trl - 4) * 0.15
    elif trl <= 7:
        mt_raw = 0.7 + (trl - 6) * 0.15
    else:
        mt_raw = 0.9 + (trl - 8) * 0.1
    # Apply deployment penalty for very early or very late
    if trl >= 9:
        mt_raw = 0.85  # deployed = less "opportunity"
    mt_c = mt_raw * 0.20 * (10 - BASE)

    # Project bonus
    pj_c = min(len(linked_projects) * 0.15, 0.5) if linked_projects else 0.0

    final = round(BASE + sr_c + ev_c + gr_c + mt_c + pj_c, 1)
    final = max(1.0, min(10.0, final))

    return final


def simulate(runs: int, db_path: str = "sonar_ai.db"):
    """Apply simulation to existing database."""
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    trends = session.query(Trend).filter(Trend.is_active == True).all()
    if not trends:
        print("❌ No trends found. Run a Full Scan first, then simulate.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  SONAR.AI — Simulating {runs} weekly runs")
    print(f"  Simulated timespan: {runs} weeks ({runs/4:.1f} months)")
    print(f"  Found {len(trends)} active trends to evolve")
    print(f"{'='*70}\n")

    updated = 0
    results = []

    for trend in trends:
        cat = trend.tso_category
        if cat == "off_topic":
            continue

        p = get_profile_at_run(cat, runs)
        if not p:
            continue

        old_score = trend.priority_score
        old_lifecycle = trend.lifecycle_status
        old_trl = trend.maturity_score
        old_signals = trend.signal_count or 0

        # Apply evolution
        trend.maturity_score = p["trl"]
        trend.lifecycle_status = p["lifecycle"]
        trend.growth_rate = p["growth"] + random.randint(-10, 10)  # slight noise
        trend.strategic_nature = p["nature"]
        trend.time_to_impact = p["tti"]
        trend.signal_count = old_signals + p["sig_boost"]
        trend.signal_count_7d = max(5, p["sig_boost"] // runs + random.randint(-3, 8))
        trend.signal_count_30d = trend.signal_count_7d * 4

        # Advance first_seen back in time to simulate history
        trend.first_seen = datetime.utcnow() - timedelta(weeks=runs)
        trend.last_updated = datetime.utcnow()

        # Mark as no longer emerging after enough runs
        if runs >= 10 and p["lifecycle"] in ("Piloting", "Scaling", "Deployed"):
            trend.is_emerging = False
        if runs >= 15:
            trend.is_validated = True  # "human reviewed" after this many runs

        # Re-score with new parameters
        new_score = rescore_trend(trend, trend.signal_count, trend.growth_rate)
        trend.priority_score = new_score

        results.append({
            "name": trend.name,
            "cat": cat,
            "old_score": old_score,
            "new_score": new_score,
            "old_trl": old_trl,
            "new_trl": p["trl"],
            "old_lc": old_lifecycle,
            "new_lc": p["lifecycle"],
            "signals": trend.signal_count,
            "growth": trend.growth_rate,
            "nature": p["nature"],
            "tti": p["tti"],
        })
        updated += 1

    # Sort by new score descending
    results.sort(key=lambda r: r["new_score"], reverse=True)

    # Print evolution report
    print(f"  {'TREND':<38} {'SCORE':>12} {'TRL':>10} {'LIFECYCLE':>22} {'SIG':>6} {'GROWTH':>8}")
    print(f"  {'─'*38} {'─'*12} {'─'*10} {'─'*22} {'─'*6} {'─'*8}")

    for i, r in enumerate(results, 1):
        score_delta = r["new_score"] - r["old_score"]
        score_arrow = f"{'↑' if score_delta > 0 else '↓' if score_delta < 0 else '='}{abs(score_delta):.1f}"
        trl_change = f"{r['old_trl']}→{r['new_trl']}" if r['old_trl'] != r['new_trl'] else f"  {r['new_trl']}  "
        lc_change = f"{r['old_lc']}→{r['new_lc']}" if r['old_lc'] != r['new_lc'] else r['new_lc']

        print(f"  {i:2}. {r['name']:<35} {r['new_score']:5.1f} ({score_arrow:>4}) "
              f" {trl_change:>8}  {lc_change:<20} {r['signals']:5} {r['growth']:>+6}%")

    # Commit
    session.commit()

    # Count lifecycle distribution
    lc_dist = {}
    for r in results:
        lc = r["new_lc"]
        lc_dist[lc] = lc_dist.get(lc, 0) + 1

    print(f"\n  {'─'*70}")
    print(f"  LIFECYCLE DISTRIBUTION after {runs} runs:")
    for lc in ["Scouting", "Watching", "Evaluating", "Piloting", "Scaling", "Deployed"]:
        count = lc_dist.get(lc, 0)
        bar = "█" * (count * 3)
        if count:
            print(f"    {lc:<14} {bar} {count}")

    # Blind spots check
    from config import TSO_CATEGORIES
    blind = [r for r in results
             if TSO_CATEGORIES.get(r["cat"], {}).get("tier") in ("existential", "critical")
             and r["signals"] < 15]
    if blind:
        print(f"\n  ⚠ BLIND SPOTS REMAINING ({len(blind)}):")
        for b in blind:
            print(f"    • {b['name']}: {b['signals']} signals ({TSO_CATEGORIES[b['cat']]['tier']})")
    else:
        print(f"\n  ✅ No blind spots — all critical categories have sufficient coverage")

    print(f"\n  ✅ Updated {updated} trends. Open dashboard to view.")
    print(f"     http://localhost:8000\n")

    session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate SONAR.AI weekly runs")
    parser.add_argument("--runs", type=int, required=True,
                        help="Number of weekly runs to simulate (4, 10, 15, 20)")
    parser.add_argument("--db", type=str, default="sonar_ai.db",
                        help="Path to database file")
    args = parser.parse_args()

    if args.runs < 1 or args.runs > 50:
        print("❌ Runs must be between 1 and 50")
        sys.exit(1)

    simulate(args.runs, args.db)
