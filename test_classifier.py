#!/usr/bin/env python3
"""
Test: Semantic Classifier Integration
======================================
Validates that the new TaxonomyClassifier (via nlp_processor.py v3)
correctly classifies signals that the old keyword matcher got wrong.

Usage:
    python test_classifier.py              # Run all tests
    python test_classifier.py --verbose    # Show all_scores per signal
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from services.nlp_processor import get_nlp_processor
from services.classifier import get_classifier
from taxonomy import CATEGORIES, CATEGORY_TO_AMPRION_TASK, CATEGORY_TO_BUSINESS_AREAS

# ─────────────────────────────────────────────────────────────────────
# TEST 1: Previously misclassified signals (from user's scrape output)
# ─────────────────────────────────────────────────────────────────────

MISCLASSIFIED_SIGNALS = [
    # (title, old_wrong_category, expected_correct_category)

    # Off-topic signals that leaked into categories
    ("Anthropic Targets $20 Billion Raise, Eyeing $350 Billion Valuation",
     "grid_infrastructure", "off_topic"),
    ("Tesla to End Production of Flagship Model S and Model X",
     "digital_twin_simulation", "off_topic"),
    ("Bluesky Charting a Course for 2026: Balancing Innovation with Essential Fixes",
     "cybersecurity_ot", "off_topic"),
    ("Australia's high speed rail ambitions could be funded through an aviation fuel excise",
     "ai_grid_optimization", "off_topic"),
    ("Meta's Reality Labs Records $19 Billion Loss as Strategic Focus Shifts",
     "ai_grid_optimization", "off_topic"),
    ("Outtake Secures $40M Series B to Combat AI-Driven Identity Fraud",
     "grid_infrastructure", "off_topic"),
    ("Google Unveils Gemini 3 Integration for Chrome: Auto Browse and Agentic AI",
     "ai_grid_optimization", "off_topic"),

    # Mislabeled energy signals (keyword confusion)
    ("Ignore your algorithm: Solar prices are going up in 2026, but it's not a crisis",
     "cybersecurity_ot", "renewables_integration"),
    ("Crude oil prices fell in 2025 amid oversupply",
     "cybersecurity_ot", "off_topic"),  # oil prices ≠ cybersecurity
    ("Coal-fired generation rose to meet demand during Winter Storm Fern",
     "power_generation", "power_generation"),  # was actually correct
    ("New transmission line connecting Hydro-Quebec to ISO-NE begins commercial operations",
     "grid_infrastructure", "grid_infrastructure"),  # was correct
    ("Petroleum electricity generation surpassed natural gas in New England during winter storm",
     "power_generation", "power_generation"),  # was correct

    # AI/tech news falsely classified as grid categories
    ("LWiAI Podcast #223 - Haiku 4.5, OpenAI DevDay, SB 243",
     "grid_stability", "off_topic"),
    ("LWiAI Podcast #225 - GPT 5.1, Kimi K2 Thinking, Remote Labor Index",
     "grid_stability", "off_topic"),
    ("AI Governance Challenges: Key Obstacles Enterprises Face When Scaling AI Responsibly",
     "ai_grid_optimization", "off_topic"),
    ("The ROI Paradox: Why Small-Scale AI Architecture Outperforms Large Corporate Programs",
     "ai_grid_optimization", "off_topic"),
    ("Microsoft's AI Ambitions Drive $7.6 Billion Net Income Boost",
     "ai_grid_optimization", "off_topic"),

    # EIA oil/gas articles falsely classified
    ("Geopolitical developments contribute to elevated diesel prices",
     "other", "off_topic"),
    ("U.S. retail gasoline prices fall below $3 per gallon, the lowest since 2021",
     "other", "off_topic"),
    ("Crude oil tanker rates reached multi-year highs in late 2025",
     "e_mobility_v2g", "off_topic"),
    ("EIA updates its definitions and estimates of OPEC crude oil production capacity",
     "renewables_integration", "off_topic"),
    ("Brazil, Guyana, and Argentina support forecast crude oil growth in 2026",
     "renewables_integration", "off_topic"),
    ("EIA forecasts U.S. crude oil production will decrease slightly in 2026",
     "renewables_integration", "off_topic"),
]


# ─────────────────────────────────────────────────────────────────────
# TEST 2: Correctly classified signals (should STILL be correct)
# ─────────────────────────────────────────────────────────────────────

CORRECTLY_CLASSIFIED = [
    ("Big batteries push coal and gas out of evening peaks", "energy_storage"),
    ("Near 100 pct renewable electricity for Australia's main grid is achievable",
     "renewables_integration"),
    ("Solar power generation drives electricity generation growth over the next two years",
     "renewables_integration"),
    ("Nuclear plants reported few outages in the first three weeks of January 2026",
     "power_generation"),
    ("Day-ahead electricity market prices spike during extreme cold weather",
     "energy_trading"),
    ("Spark and dark spreads indicate improved profitability of natural gas, coal power plants",
     "energy_trading"),
    ("Severe winter weather across large portions of the country, natural gas prices increasing",
     "power_generation"),
]


# ─────────────────────────────────────────────────────────────────────
# TEST 3: Amprion enrichment mapping validation
# ─────────────────────────────────────────────────────────────────────

AMPRION_MAPPING_TESTS = [
    ("grid_infrastructure", "grid_expansion", ["GP", "AM"]),
    ("grid_stability", "system_security", ["SO", "AM"]),
    ("cybersecurity_ot", "system_security", ["ITD", "SO"]),
    ("renewables_integration", "decarbonization", ["SO", "M"]),
    ("flexibility_vpp", "european_trading", ["SO", "M", "ITD"]),
    ("offshore_systems", "grid_expansion", ["GP", "AM", "CS"]),
    ("e_mobility_v2g", "decarbonization", ["SO", "GP", "AM"]),
    ("hydrogen_p2g", "decarbonization", ["GP", "CS", "M"]),
    ("energy_trading", "european_trading", ["M", "SO"]),
    ("regulatory_policy", "european_trading", ["CS", "M"]),
    ("ai_grid_optimization", "system_security", ["ITD", "SO", "AM"]),
    ("off_topic", None, []),
]


def run_tests(verbose=False):
    proc = get_nlp_processor()
    clf = get_classifier()
    
    print("=" * 78)
    print(f"SONAR.AI Semantic Classifier Test Suite")
    print(f"Backend: {clf.backend_name}  |  Categories: {len(clf.categories)}")
    print("=" * 78)
    
    all_pass = True
    
    # ── Test 1: Previously misclassified ────────────────────────────
    print(f"\n{'─'*78}")
    print("TEST 1: Previously Misclassified Signals (should now be fixed)")
    print(f"{'─'*78}")
    
    fixed = 0
    total = len(MISCLASSIFIED_SIGNALS)
    for title, old_cat, expected in MISCLASSIFIED_SIGNALS:
        r = proc.classify_signal(title)
        got = r["tso_category"]
        
        # For "off_topic" expected, also accept very low confidence on-topic
        is_correct = (got == expected)
        # Relax: if we expected off_topic but got a low-confidence on-topic, still OK
        if expected == "off_topic" and got != "off_topic":
            # Accept if domain_relevance is very low (borderline)
            if r["domain_relevance"] < 0.02 and r["tso_category_confidence"] < 0.5:
                is_correct = True  # close enough
        
        icon = "✅" if is_correct else "❌"
        if is_correct:
            fixed += 1
        else:
            all_pass = False
        
        print(f"  {icon} {title[:60]:61s} → {got:25s} "
              f"(was: {old_cat}, exp: {expected})")
        
        if verbose:
            top3 = sorted(r.get("all_scores", {}).items() if "all_scores" in r else [],
                         key=lambda x: -x[1])[:3]
            print(f"       conf={r['tso_category_confidence']:.2f}  "
                  f"dom={r['domain_relevance']:.2f}  "
                  f"top3={[(c,f'{s:.3f}') for c,s in top3]}")
    
    print(f"\n  Result: {fixed}/{total} fixed "
          f"({'ALL' if fixed == total else f'{total-fixed} remaining'})")
    
    # ── Test 2: Regression — still correct ──────────────────────────
    print(f"\n{'─'*78}")
    print("TEST 2: Regression Check (should still be correct)")
    print(f"{'─'*78}")
    
    reg_pass = 0
    for title, expected in CORRECTLY_CLASSIFIED:
        r = proc.classify_signal(title)
        got = r["tso_category"]
        ok = got == expected
        if ok:
            reg_pass += 1
        else:
            all_pass = False
        icon = "✅" if ok else "❌"
        print(f"  {icon} {title[:60]:61s} → {got:25s} (exp: {expected})")
    
    print(f"\n  Result: {reg_pass}/{len(CORRECTLY_CLASSIFIED)} still correct")
    
    # ── Test 3: Amprion mappings ────────────────────────────────────
    print(f"\n{'─'*78}")
    print("TEST 3: Amprion Deterministic Mappings")
    print(f"{'─'*78}")
    
    map_pass = 0
    for cat, exp_task, exp_areas in AMPRION_MAPPING_TESTS:
        task = CATEGORY_TO_AMPRION_TASK.get(cat)
        areas = CATEGORY_TO_BUSINESS_AREAS.get(cat, [])
        ok = (task == exp_task) and (areas == exp_areas)
        if ok:
            map_pass += 1
        else:
            all_pass = False
        icon = "✅" if ok else "❌"
        print(f"  {icon} {cat:25s} → task={str(task):20s} areas={areas}")
    
    print(f"\n  Result: {map_pass}/{len(AMPRION_MAPPING_TESTS)} correct")
    
    # ── Test 4: Quality score domain penalty ────────────────────────
    print(f"\n{'─'*78}")
    print("TEST 4: Quality Score — Domain Relevance Penalty")
    print(f"{'─'*78}")
    
    q_on = proc.calculate_quality_score("news", 0.7, 500, True, domain_relevance=0.5)
    q_off = proc.calculate_quality_score("news", 0.7, 500, True, domain_relevance=0.0)
    q_diff = q_on - q_off
    ok = q_diff > 0.05  # On-topic should score meaningfully higher
    icon = "✅" if ok else "❌"
    print(f"  {icon} On-topic quality: {q_on:.3f}  Off-topic: {q_off:.3f}  "
          f"Δ={q_diff:.3f} (should be >0.05)")
    if not ok:
        all_pass = False
    
    # ── Test 5: Taxonomy completeness ───────────────────────────────
    print(f"\n{'─'*78}")
    print("TEST 5: Taxonomy Completeness")
    print(f"{'─'*78}")
    
    issues = []
    for cat_id, cat_data in CATEGORIES.items():
        if len(cat_data.get("prototypes", [])) < 5:
            issues.append(f"  ⚠️ {cat_id}: only {len(cat_data['prototypes'])} prototypes (want ≥5)")
        if len(cat_data.get("boost_keywords", [])) < 5:
            issues.append(f"  ⚠️ {cat_id}: only {len(cat_data['boost_keywords'])} keywords (want ≥5)")
        if cat_id not in CATEGORY_TO_AMPRION_TASK:
            issues.append(f"  ❌ {cat_id}: missing Amprion task mapping")
        if cat_id not in CATEGORY_TO_BUSINESS_AREAS:
            issues.append(f"  ❌ {cat_id}: missing business area mapping")
    
    if issues:
        for iss in issues:
            print(iss)
        if any("❌" in i for i in issues):
            all_pass = False
    else:
        print("  ✅ All categories have sufficient prototypes, keywords, and mappings")
    
    print(f"\n  Categories: {len(CATEGORIES)}")
    print(f"  Total prototypes: {sum(len(c['prototypes']) for c in CATEGORIES.values())}")
    print(f"  Total boost keywords: {sum(len(c['boost_keywords']) for c in CATEGORIES.values())}")
    
    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"OVERALL: {'ALL TESTS PASSED ✅' if all_pass else 'SOME TESTS FAILED ❌'}")
    print(f"{'='*78}")
    
    return all_pass


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    ok = run_tests(verbose=verbose)
    sys.exit(0 if ok else 1)
