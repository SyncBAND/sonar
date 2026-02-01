"""
Test Suite for SONAR.AI Trend Scorer v2
========================================
Validates:
  1. Score spread (no ceiling compression)
  2. Maturity computed (not hardcoded at 5)
  3. Criticality correctly assessed
  4. 6 signals ≠ 84 signals
  5. Backward compatibility with full.py fields
  6. SHAP explanation present
  7. Monitoring alerts for blind spots
"""

import sys
sys.path.insert(0, ".")

from datetime import datetime, timedelta
from services.trend_scorer import TrendScorer, AMPRION_STRATEGIC_PRIORS


def make_signals(n, source_type="news", quality=0.65, days_spread=30):
    """Generate realistic test signals."""
    now = datetime.utcnow()
    signals = []
    for i in range(n):
        signals.append({
            "title": f"Test signal {i} about grid technology",
            "content": f"Content for signal {i}",
            "quality_score": quality,
            "source_type": source_type,
            "source_name": f"Source_{i % max(n // 3, 1)}",
            "published_at": (now - timedelta(days=i % days_spread)).isoformat(),
        })
    return signals


def test_score_spread():
    """Priority scores should spread across 1–10, not compress at ceiling."""
    scorer = TrendScorer()

    categories = [
        ("grid_stability", 6), ("cybersecurity_ot", 169),
        ("grid_infrastructure", 17), ("offshore_systems", 43),
        ("renewables_integration", 84), ("energy_storage", 73),
        ("flexibility_vpp", 10), ("e_mobility_v2g", 97),
        ("ai_grid_optimization", 133), ("hydrogen_p2g", 64),
        ("regulatory_policy", 85), ("energy_trading", 57),
        ("distributed_generation", 14), ("digital_twin_simulation", 24),
        ("biogas_biomethane", 9), ("power_generation", 28),
    ]

    scores = []
    for cat, count in categories:
        signals = make_signals(count)
        result = scorer.score_trend(
            {"tso_category": cat, "amprion_task": "system_security",
             "linked_projects": [], "strategic_domain": "system_operation"},
            signals,
        )
        scores.append((cat, result["priority_score"]))

    values = [s for _, s in scores]
    score_range = max(values) - min(values)
    unique_scores = len(set(round(v, 1) for v in values))

    print(f"\n{'='*60}")
    print("TEST: Score Spread")
    print(f"{'='*60}")
    print(f"  Range: {min(values):.1f} – {max(values):.1f} (span={score_range:.1f})")
    print(f"  Unique: {unique_scores} / {len(values)}")

    assert score_range >= 1.5, f"Score range too narrow: {score_range:.1f} (need ≥1.5)"
    assert unique_scores >= 10, f"Too few unique scores: {unique_scores} (need ≥10)"
    assert max(values) < 10.0, f"Ceiling compression: max={max(values)}"
    print("  ✅ PASSED")


def test_no_ceiling_compression():
    """No more than 2 trends should score exactly 10.0."""
    scorer = TrendScorer()
    at_ceiling = 0
    for cat, prior in AMPRION_STRATEGIC_PRIORS.items():
        signals = make_signals(100)
        result = scorer.score_trend(
            {"tso_category": cat, "amprion_task": "system_security",
             "linked_projects": [], "strategic_domain": "system_operation"},
            signals,
        )
        if result["priority_score"] >= 10.0:
            at_ceiling += 1

    print(f"\n{'='*60}")
    print("TEST: Ceiling Compression")
    print(f"{'='*60}")
    print(f"  Trends at 10.0: {at_ceiling} / {len(AMPRION_STRATEGIC_PRIORS)}")
    assert at_ceiling <= 2, f"Ceiling compression: {at_ceiling} trends at 10.0"
    print("  ✅ PASSED")


def test_maturity_computed():
    """Maturity should vary by category and signal content, not be hardcoded."""
    scorer = TrendScorer()

    # Research-heavy category
    research_signals = make_signals(50, source_type="research")
    r1 = scorer.score_trend(
        {"tso_category": "ai_grid_optimization", "amprion_task": "system_security",
         "linked_projects": [], "strategic_domain": "system_operation"},
        research_signals,
    )

    # Deployment-heavy category
    deploy_signals = []
    for i in range(50):
        deploy_signals.append({
            "title": f"New grid infrastructure deployed and commissioned",
            "quality_score": 0.7,
            "source_type": "news",
            "source_name": f"Source_{i%5}",
            "published_at": (datetime.utcnow() - timedelta(days=i%30)).isoformat(),
        })
    r2 = scorer.score_trend(
        {"tso_category": "grid_infrastructure", "amprion_task": "grid_expansion",
         "linked_projects": [], "strategic_domain": "digital_asset_lifecycle"},
        deploy_signals,
    )

    print(f"\n{'='*60}")
    print("TEST: Maturity Computed")
    print(f"{'='*60}")
    print(f"  AI/Research signals: TRL={r1['maturity_score']}, lifecycle={r1['lifecycle_status']}")
    print(f"  Grid/Deploy signals: TRL={r2['maturity_score']}, lifecycle={r2['lifecycle_status']}")

    assert r1["maturity_score"] != 5 or r2["maturity_score"] != 5, \
        "Both maturity scores are 5 — still hardcoded!"
    assert r1["maturity_score"] <= r2["maturity_score"], \
        f"Research ({r1['maturity_score']}) should be ≤ deployment ({r2['maturity_score']})"
    assert r1["lifecycle_status"] != r2["lifecycle_status"] or \
        r1["maturity_score"] != r2["maturity_score"], \
        "No differentiation between research and deployment"
    print("  ✅ PASSED")


def test_six_vs_eighty_four():
    """6 signals (Grid Stability) must NOT score identically to 84 (Renewables)."""
    scorer = TrendScorer()

    r_gs = scorer.score_trend(
        {"tso_category": "grid_stability", "amprion_task": "system_security",
         "linked_projects": [], "strategic_domain": "system_operation"},
        make_signals(6),
    )
    r_ri = scorer.score_trend(
        {"tso_category": "renewables_integration", "amprion_task": "decarbonization",
         "linked_projects": [], "strategic_domain": "system_operation"},
        make_signals(84),
    )

    print(f"\n{'='*60}")
    print("TEST: 6 signals ≠ 84 signals")
    print(f"{'='*60}")
    print(f"  Grid Stability (6 sig):  {r_gs['priority_score']:.1f}, "
          f"evidence={r_gs['evidence_strength_score']:.1f}")
    print(f"  Renewables (84 sig):     {r_ri['priority_score']:.1f}, "
          f"evidence={r_ri['evidence_strength_score']:.1f}")
    print(f"  Δ priority: {abs(r_gs['priority_score'] - r_ri['priority_score']):.1f}")

    assert r_gs["priority_score"] != r_ri["priority_score"], \
        "6 signals scored IDENTICALLY to 84 — evidence not contributing"
    assert r_gs["evidence_strength_score"] < r_ri["evidence_strength_score"], \
        "6 signals should have LOWER evidence strength than 84"
    print("  ✅ PASSED")


def test_criticality_ranking():
    """EXISTENTIAL categories should rank higher than MEDIUM categories
    given comparable evidence."""
    scorer = TrendScorer()

    # Give both categories the same amount of evidence
    r_critical = scorer.score_trend(
        {"tso_category": "grid_stability", "amprion_task": "system_security",
         "linked_projects": [], "strategic_domain": "system_operation"},
        make_signals(50),
    )
    r_medium = scorer.score_trend(
        {"tso_category": "biogas_biomethane", "amprion_task": "decarbonization",
         "linked_projects": [], "strategic_domain": "energy_markets_flexibility"},
        make_signals(50),
    )

    print(f"\n{'='*60}")
    print("TEST: Criticality Ranking")
    print(f"{'='*60}")
    print(f"  Grid Stability (EXISTENTIAL, 50 sig): {r_critical['priority_score']:.1f}")
    print(f"  Biogas (MEDIUM, 50 sig):              {r_medium['priority_score']:.1f}")
    print(f"  Δ: {r_critical['priority_score'] - r_medium['priority_score']:.1f}")

    assert r_critical["priority_score"] > r_medium["priority_score"], \
        "EXISTENTIAL category must outrank MEDIUM with equal evidence"
    assert (r_critical["priority_score"] - r_medium["priority_score"]) >= 0.5, \
        "Gap between EXISTENTIAL and MEDIUM should be ≥0.5"
    print("  ✅ PASSED")


def test_backward_compatibility():
    """Scorer output must contain all fields that full.py reads."""
    scorer = TrendScorer()
    result = scorer.score_trend(
        {"tso_category": "grid_stability", "amprion_task": "system_security",
         "linked_projects": [], "strategic_domain": "system_operation"},
        make_signals(10),
    )

    required_fields = [
        "priority_score", "strategic_relevance_score",
        "grid_stability_score", "cost_efficiency_score",
        "volume_score", "growth_score", "quality_score",
        "growth_rate", "project_multiplier",
        "strategic_nature", "time_to_impact",
        "maturity_score", "lifecycle_status", "maturity_type",
        "shapley_explanation",
    ]

    print(f"\n{'='*60}")
    print("TEST: Backward Compatibility")
    print(f"{'='*60}")

    missing = [f for f in required_fields if f not in result]
    if missing:
        print(f"  ❌ MISSING: {missing}")
        assert False, f"Missing fields: {missing}"
    print(f"  All {len(required_fields)} required fields present")
    print("  ✅ PASSED")


def test_monitoring_alert():
    """High-importance + low-evidence should trigger monitoring alert."""
    scorer = TrendScorer()

    # EXISTENTIAL category with very few signals
    r = scorer.score_trend(
        {"tso_category": "grid_stability", "amprion_task": "system_security",
         "linked_projects": [], "strategic_domain": "system_operation"},
        make_signals(2, quality=0.5),  # Very few signals
    )

    print(f"\n{'='*60}")
    print("TEST: Monitoring Alert")
    print(f"{'='*60}")
    print(f"  Grid Stability (2 signals): alert={r['monitoring_alert']}, "
          f"evidence={r['evidence_strength_score']:.1f}")

    assert r["monitoring_alert"] is True, \
        "EXISTENTIAL category with 2 signals should trigger monitoring alert"
    print("  ✅ PASSED")


def test_shap_explanation():
    """SHAP explanation should be additive and contain waterfall."""
    scorer = TrendScorer()
    result = scorer.score_trend(
        {"tso_category": "cybersecurity_ot", "amprion_task": "system_security",
         "linked_projects": [], "strategic_domain": "cybersecurity_trust"},
        make_signals(50),
    )

    exp = result["shapley_explanation"]

    print(f"\n{'='*60}")
    print("TEST: SHAP Explanation")
    print(f"{'='*60}")

    assert "waterfall" in exp, "Missing waterfall"
    assert "feature_importance" in exp, "Missing feature_importance"
    assert "top_drivers" in exp, "Missing top_drivers"
    assert len(exp["waterfall"]) >= 5, "Waterfall too short"

    # Check additivity
    wf = exp["waterfall"]
    final = wf[-1]["cumulative"]
    print(f"  Waterfall final: {final:.2f}")
    print(f"  Priority score: {result['priority_score']:.1f}")

    # CRITICAL: waterfall must match the actual priority score
    assert abs(final - result["priority_score"]) <= 0.15, \
        f"Waterfall ({final:.2f}) doesn't match priority ({result['priority_score']:.1f})"

    print(f"  Top drivers: {len(exp['top_drivers'])}")
    print(f"  Features: {list(exp['feature_importance'].keys())}")
    print("  ✅ PASSED")


def test_lifecycle_variation():
    """Different categories should produce different lifecycle statuses."""
    scorer = TrendScorer()

    lifecycles = set()
    for cat in AMPRION_STRATEGIC_PRIORS:
        r = scorer.score_trend(
            {"tso_category": cat, "amprion_task": "system_security",
             "linked_projects": [], "strategic_domain": "system_operation"},
            make_signals(30),
        )
        lifecycles.add(r["lifecycle_status"])

    print(f"\n{'='*60}")
    print("TEST: Lifecycle Variation")
    print(f"{'='*60}")
    print(f"  Unique lifecycle statuses: {lifecycles}")

    assert len(lifecycles) >= 2, \
        f"Need ≥2 different lifecycle statuses, got: {lifecycles}"
    print("  ✅ PASSED")


def test_cold_start_growth():
    """First-run (cold start) should NOT inflate growth to 95."""
    scorer = TrendScorer()
    now = datetime.utcnow()

    # Simulate cold start: all signals scraped at same time,
    # with natural RSS published_at distribution
    signals = []
    for i in range(50):
        signals.append({
            "title": f"Cold start signal {i}",
            "content": "Deployed commissioned operational",
            "quality_score": 0.68,
            "source_name": f"Source_{i % 5}",
            "source_type": "news",
            "published_at": (now - timedelta(days=i % 5)).isoformat(),
            "scraped_at": now.isoformat(),
        })
    for i in range(5):
        signals.append({
            "title": f"Older signal {i}",
            "content": "Analysis study review",
            "quality_score": 0.65,
            "source_name": "EIA",
            "source_type": "news",
            "published_at": (now - timedelta(days=20 + i * 3)).isoformat(),
            "scraped_at": now.isoformat(),
        })

    r = scorer.score_trend(
        {"tso_category": "renewables_integration", "linked_projects": ["BalWin1"]},
        signals,
    )

    print(f"\n{'='*60}")
    print("TEST: Cold-Start Growth Damping")
    print(f"{'='*60}")
    print(f"  growth_score: {r['growth_score']} (should be ≤ 65, was 95 before fix)")
    print(f"  growth_rate: {r['growth_rate']}%")
    print(f"  priority: {r['priority_score']}")

    assert r["growth_score"] <= 65, \
        f"Cold-start growth should be ≤ 65, got {r['growth_score']}"
    assert r["growth_score"] >= 30, \
        f"Cold-start growth should be ≥ 30, got {r['growth_score']}"
    print("  ✅ PASSED")


# =========================================================================

if __name__ == "__main__":
    tests = [
        test_score_spread,
        test_no_ceiling_compression,
        test_maturity_computed,
        test_six_vs_eighty_four,
        test_criticality_ranking,
        test_backward_compatibility,
        test_monitoring_alert,
        test_shap_explanation,
        test_lifecycle_variation,
        test_cold_start_growth,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)
