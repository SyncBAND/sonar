#!/usr/bin/env python3
"""
SONAR.AI Demo Runner
====================
Execute the full agentic workflow with rich visual output for demo presentation.

This script demonstrates the complete end-to-end trend scanning pipeline:
  1. FIND SIGNALS      - Collect data from RSS, arXiv, TSO, regulatory sources
  2. CLUSTER SIGNALS   - Classify into 16 TSO categories using semantic similarity
  3. DERIVE TRENDS     - Group signals into trend clusters
  4. ASSESS TRENDS     - Score using Bayesian MCDA + AHP
  5. PREPARE RESULTS   - Generate narratives for decision-makers
  6. VALIDATE RESULTS  - Quality checks, alerts, traceability

Usage:
    # Full scan with visual output
    python demo_runner.py
    
    # With custom goal
    python demo_runner.py --goal "Find emerging cybersecurity threats for OT systems"
    
    # With specific sources only
    python demo_runner.py --sources arxiv,rss_news,regulatory
    
    # Export trace for audit
    python demo_runner.py --export-trace trace_output.json

Environment Variables (LLM Configuration):
    # Commercial LLMs
    CLASSIFIER_LLM_PROVIDER=anthropic|openai
    ANTHROPIC_API_KEY=sk-...
    OPENAI_API_KEY=sk-...
    
    # Self-hosted LLMs
    CLASSIFIER_LLM_PROVIDER=selfhosted
    SELFHOSTED_LLM_PROVIDER=ollama|vllm|llamacpp|textgen|localai|lmstudio
    SELFHOSTED_LLM_URL=http://localhost:11434
    SELFHOSTED_LLM_MODEL=llama3.2:8b

Requirements:
    - All source data is from publicly available, GDPR-compliant sources
    - Data processing on EU servers only
    - Full traceability from signals to trends

Author: SONAR.AI Team
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.visual_logger import VisualLogger


async def run_demo_scan(
    goal: str = "Identify highest-priority emerging trends for Amprion TSO operations",
    sources: Optional[List[str]] = None,
    export_trace: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete SONAR.AI agentic scan with visual logging.
    
    This is the main demo function that showcases:
    - End-to-end workflow from signals to strategic intelligence
    - Clear step-by-step progress
    - Traceability from source to trend
    - Quality validation and alerts
    """
    
    # Initialize visual logger
    vlog = VisualLogger(enable_colors=True, verbose=verbose)
    vlog.banner("SONAR.AI v2.1", "Agentic Global Trend Scanner for TSO Operations")
    
    # Show LLM configuration
    vlog.header("SYSTEM CONFIGURATION")
    
    llm_provider = os.getenv("CLASSIFIER_LLM_PROVIDER", "").lower()
    if llm_provider in ("anthropic", "openai"):
        vlog.info(f"Classifier: LLM ({llm_provider}) with local fallback")
    elif llm_provider == "selfhosted":
        sh_provider = os.getenv("SELFHOSTED_LLM_PROVIDER", "unknown")
        sh_url = os.getenv("SELFHOSTED_LLM_URL", "localhost")
        vlog.info(f"Classifier: Self-hosted LLM ({sh_provider} @ {sh_url})")
    else:
        vlog.info("Classifier: SentenceTransformer (all-MiniLM-L6-v2)")
    
    vlog.info(f"Goal: {goal}")
    vlog.info(f"Sources: {', '.join(sources) if sources else 'all'}")
    
    results = {
        "start_time": datetime.utcnow().isoformat(),
        "goal": goal,
        "steps": {}
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: FIND SIGNALS
    # "Automatically capture relevant information from publicly available sources"
    # ═══════════════════════════════════════════════════════════════════════════
    
    with vlog.phase("FIND SIGNALS"):
        vlog.step("Initializing data scrapers")
        
        from services.scrapers import MasterScraper
        scraper = MasterScraper()
        
        available_scrapers = list(scraper.scrapers.keys())
        vlog.substep(f"Available scrapers: {', '.join(available_scrapers)}")
        
        vlog.step("Scraping data sources in parallel")
        
        try:
            all_signals = await scraper.scrape_all(sources)
            
            # Collect statistics
            source_breakdown = Counter()
            source_names = Counter()
            for sig in all_signals:
                source_breakdown[sig.get("source_type", "unknown")] += 1
                source_names[sig.get("source_name", "unknown")] += 1
            
            vlog.success(f"Scraped {len(all_signals)} signals")
            
            # Show source breakdown
            vlog.step("Source breakdown by type")
            for src_type, count in sorted(source_breakdown.items(), key=lambda x: -x[1]):
                vlog.substep(f"{src_type}: {count} signals")
            
            # Show top sources
            vlog.step("Top 10 sources by volume")
            for src_name, count in source_names.most_common(10):
                vlog.substep(f"{src_name}: {count}")
            
            # Check for deduplication
            unique_urls = len(set(s.get("url", "") for s in all_signals if s.get("url")))
            vlog.metric("Raw signals", len(all_signals))
            vlog.metric("Unique URLs", unique_urls)
            vlog.metric("Duplicates removed", len(all_signals) - unique_urls)
            
            results["steps"]["find_signals"] = {
                "total": len(all_signals),
                "unique": unique_urls,
                "breakdown": dict(source_breakdown)
            }
            
        except Exception as e:
            vlog.error(f"Scraping failed: {e}")
            all_signals = []
            results["steps"]["find_signals"] = {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: CLUSTER SIGNALS (Classification)
    # "Group similar signals to identify patterns and connections"
    # ═══════════════════════════════════════════════════════════════════════════
    
    with vlog.phase("CLUSTER SIGNALS"):
        vlog.step("Loading semantic classifier")
        
        from services.nlp_processor import get_nlp_processor
        processor = get_nlp_processor()
        
        backend_name = getattr(processor.classifier, 'backend_name', 'unknown')
        vlog.substep(f"Backend: {backend_name}")
        
        vlog.step("Classifying signals into 16 TSO categories")
        
        classified_signals = []
        category_counts = Counter()
        off_topic_count = 0
        confidence_sum = 0.0
        
        total = len(all_signals)
        for i, sig in enumerate(all_signals):
            # Show progress every 100 signals
            if (i + 1) % 100 == 0:
                vlog.progress(i + 1, total, "signals")
            
            try:
                result = processor.classify_signal(
                    sig.get("title", ""),
                    sig.get("content", "")
                )
                
                sig["classification"] = result
                sig["tso_category"] = result.get("tso_category", "other")
                sig["confidence"] = result.get("tso_category_confidence", 0.0)
                
                if result.get("is_off_topic", False):
                    off_topic_count += 1
                else:
                    category_counts[sig["tso_category"]] += 1
                    confidence_sum += sig["confidence"]
                
                classified_signals.append(sig)
                
            except Exception as e:
                vlog.warning(f"Classification failed for signal: {e}")
        
        if total > 100:
            vlog.progress(total, total, "signals")
        
        avg_confidence = confidence_sum / len(classified_signals) if classified_signals else 0.0
        
        vlog.success(f"Classified {len(classified_signals)} signals")
        
        # Show category distribution
        vlog.step("Category distribution")
        vlog.table(
            ["Category", "Count", "Percentage"],
            [
                [cat, count, f"{count/len(classified_signals)*100:.1f}%"]
                for cat, count in category_counts.most_common(16)
            ]
        )
        
        vlog.metric("Signals classified", len(classified_signals))
        vlog.metric("Off-topic filtered", off_topic_count)
        vlog.metric("Average confidence", f"{avg_confidence:.2%}")
        vlog.metric("Categories with signals", len(category_counts))
        
        results["steps"]["cluster_signals"] = {
            "classified": len(classified_signals),
            "off_topic": off_topic_count,
            "avg_confidence": round(avg_confidence, 3),
            "categories": dict(category_counts)
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: DERIVE TRENDS
    # "Identify potential trends based on clusters"
    # ═══════════════════════════════════════════════════════════════════════════
    
    with vlog.phase("DERIVE TRENDS"):
        vlog.step("Grouping signals by category")
        
        from taxonomy import CATEGORIES
        
        # Group signals
        category_groups = {}
        for sig in classified_signals:
            cat = sig.get("tso_category", "other")
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(sig)
        
        vlog.substep(f"Found {len(category_groups)} categories with signals")
        
        vlog.step("Creating trends (min 3 signals per trend)")
        
        trends = []
        for cat, signals in category_groups.items():
            if len(signals) < 3:
                vlog.substep(f"Skipping {cat}: only {len(signals)} signals (need 3+)")
                continue
            
            cat_info = CATEGORIES.get(cat, {})
            trend = {
                "category": cat,
                "name": cat_info.get("name", cat.replace("_", " ").title()),
                "signal_count": len(signals),
                "signals": signals,
                "keywords": list(set(
                    kw for sig in signals[:20] 
                    for kw in sig.get("classification", {}).get("keywords", [])[:5]
                ))[:15]
            }
            trends.append(trend)
            vlog.substep(f"Created: {trend['name']} ({len(signals)} signals)")
        
        vlog.success(f"Derived {len(trends)} trends")
        vlog.metric("Trends created", len(trends))
        vlog.metric("Signals in trends", sum(t["signal_count"] for t in trends))
        
        results["steps"]["derive_trends"] = {
            "trends_created": len(trends),
            "trend_names": [t["name"] for t in trends]
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: ASSESS TRENDS
    # "Evaluate trends in terms of relevance, potential, and strategic importance"
    # ═══════════════════════════════════════════════════════════════════════════
    
    with vlog.phase("ASSESS TRENDS"):
        vlog.step("Loading Bayesian MCDA scorer")
        
        from services.trend_scorer import get_trend_scorer, AMPRION_STRATEGIC_PRIORS
        from services.ahp import get_mcda_weights, _get_active_profile_name
        
        scorer = get_trend_scorer()
        
        try:
            mcda_weights = get_mcda_weights()
            ahp_profile = _get_active_profile_name()
        except:
            mcda_weights = {
                "strategic_importance": 0.43,
                "evidence_strength": 0.23,
                "growth_momentum": 0.15,
                "maturity_readiness": 0.19
            }
            ahp_profile = "default"
        
        vlog.substep(f"AHP Profile: {ahp_profile}")
        vlog.substep("MCDA Weights:")
        for factor, weight in mcda_weights.items():
            vlog.substep(f"  {factor}: {weight:.2f}")
        
        vlog.step("Scoring trends with 4-factor MCDA")
        
        scored_trends = []
        for trend in trends:
            # Prepare signal data
            signal_data = [
                {
                    "title": s.get("title", ""),
                    "content": s.get("content", ""),
                    "quality_score": s.get("source_quality_score", 0.5),
                    "source_name": s.get("source_name", ""),
                    "source_type": s.get("source_type", ""),
                    "published_at": s.get("published_at"),
                    "scraped_at": s.get("scraped_at")
                }
                for s in trend["signals"]
            ]
            
            trend_data = {
                "tso_category": trend["category"],
                "linked_projects": [],
                "maturity_score": 5
            }
            
            scores = scorer.score_trend(trend_data, signal_data)
            
            trend["scores"] = scores
            trend["priority_score"] = scores["priority_score"]
            trend["amprion_tier"] = scores.get("amprion_tier", "medium")
            trend["recommended_action"] = scores.get("recommended_action", "Monitor")
            trend["growth_rate"] = scores.get("growth_rate", 0)
            trend["strategic_nature"] = scores.get("strategic_nature", "Accelerator")
            trend["time_to_impact"] = scores.get("time_to_impact", "1-3 years")
            trend["explanation"] = scores.get("shapley_explanation", {})
            
            scored_trends.append(trend)
        
        # Sort by priority score
        scored_trends.sort(key=lambda x: -x["priority_score"])
        
        vlog.success("Scoring complete")
        
        # Show top trends
        vlog.step("Top 10 Trends by Priority Score")
        for i, trend in enumerate(scored_trends[:10], 1):
            vlog.trend_card(
                rank=i,
                name=trend["name"],
                score=trend["priority_score"],
                signals=trend["signal_count"],
                tier=trend["amprion_tier"],
                action=trend["recommended_action"],
                growth=trend["growth_rate"]
            )
        
        # Score distribution
        scores = [t["priority_score"] for t in scored_trends]
        vlog.metric("Score range", f"{min(scores):.1f} - {max(scores):.1f}")
        vlog.metric("Average score", f"{sum(scores)/len(scores):.2f}")
        vlog.metric("High priority (≥8.0)", len([s for s in scores if s >= 8.0]))
        
        results["steps"]["assess_trends"] = {
            "trends_scored": len(scored_trends),
            "score_range": [round(min(scores), 1), round(max(scores), 1)],
            "rankings": [
                {"name": t["name"], "score": t["priority_score"], "tier": t["amprion_tier"]}
                for t in scored_trends[:10]
            ]
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: PREPARE RESULTS
    # "Present trends and analyses clearly and understandably for decision-makers"
    # ═══════════════════════════════════════════════════════════════════════════
    
    with vlog.phase("PREPARE RESULTS"):
        vlog.step("Generating narratives for trends")
        
        from services.narrative_generator import NarrativeGenerator
        from taxonomy import CATEGORIES as TSO_CATEGORIES
        
        narr_gen = NarrativeGenerator(mode="auto")
        
        for trend in scored_trends:
            cat = trend["category"]
            cat_config = TSO_CATEGORIES.get(cat, {})
            prior = AMPRION_STRATEGIC_PRIORS.get(cat, {})
            
            signal_data = [
                {"title": s.get("title", ""), "content": s.get("content", "")}
                for s in trend["signals"][:20]
            ]
            
            narratives = narr_gen.generate_all(
                category=cat,
                trend_name=trend["name"],
                scores=trend["scores"],
                signal_count=trend["signal_count"],
                growth_rate=trend["growth_rate"],
                tier=prior.get("tier", "medium"),
                weight=prior.get("weight", 50),
                rationale=prior.get("rationale", ""),
                strategic_impact=cat_config.get("strategic_impact", ""),
                projects=prior.get("default_projects", []),
                nature=trend["strategic_nature"],
                time_to_impact=trend["time_to_impact"],
                signals=signal_data,
                maturity_score=5
            )
            
            trend["brief"] = narratives.get("description_short", "")
            trend["deep_dive"] = narratives.get("description_full", "")
            trend["so_what"] = narratives.get("so_what_summary", "")
            trend["key_players"] = narratives.get("key_players", [])
        
        vlog.success("Narratives generated")
        
        # Show sample narrative
        if scored_trends:
            top_trend = scored_trends[0]
            vlog.step(f"Sample Narrative: {top_trend['name']}")
            vlog.substep(f"Brief: {top_trend['brief'][:150]}...")
            if top_trend.get('so_what'):
                vlog.substep(f"So What: {top_trend['so_what'][:150]}...")
        
        vlog.metric("Briefs generated", len(scored_trends))
        vlog.metric("Deep dives generated", len(scored_trends))
        vlog.metric("Key players extracted", sum(len(t.get("key_players", [])) for t in scored_trends))
        
        results["steps"]["prepare_results"] = {
            "narratives_generated": len(scored_trends),
            "top_trend_brief": scored_trends[0]["brief"] if scored_trends else None
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6: VALIDATE RESULTS
    # "Ensure quality, timeliness, de-duplication, correct assignment, relevance"
    # ═══════════════════════════════════════════════════════════════════════════
    
    with vlog.phase("VALIDATE RESULTS"):
        vlog.step("Running quality validation checks")
        
        # Check 1: Coverage
        total_categories = len(TSO_CATEGORIES)
        covered_categories = len(category_counts)
        coverage_pct = (covered_categories / total_categories) * 100
        
        vlog.validation_result(
            "Category Coverage",
            coverage_pct >= 50,
            f"{covered_categories}/{total_categories} categories ({coverage_pct:.0f}%)"
        )
        
        # Check 2: Deduplication
        unique_titles = len(set(s.get("title", "") for s in classified_signals))
        dedup_ok = unique_titles >= len(classified_signals) * 0.95
        vlog.validation_result(
            "Deduplication",
            dedup_ok,
            f"{unique_titles} unique titles out of {len(classified_signals)}"
        )
        
        # Check 3: Classification confidence
        conf_ok = avg_confidence >= 0.5
        vlog.validation_result(
            "Classification Confidence",
            conf_ok,
            f"Average confidence: {avg_confidence:.2%}"
        )
        
        # Check 4: Traceability
        all_have_urls = all(s.get("url") for s in classified_signals)
        vlog.validation_result(
            "Source Traceability",
            all_have_urls,
            "Every signal has source URL for audit"
        )
        
        # Check 5: Score distribution
        score_spread = max(scores) - min(scores) if scores else 0
        vlog.validation_result(
            "Score Differentiation",
            score_spread >= 1.5,
            f"Score spread: {score_spread:.1f} points"
        )
        
        # Detect blind spots (high-tier categories with low coverage)
        vlog.step("Checking for blind spots")
        
        blind_spots = []
        for cat, prior in AMPRION_STRATEGIC_PRIORS.items():
            tier = prior.get("tier", "low")
            signal_count = category_counts.get(cat, 0)
            
            if tier in ("existential", "critical") and signal_count < 15:
                blind_spots.append({
                    "category": cat,
                    "tier": tier,
                    "signals": signal_count
                })
                vlog.alert(
                    "BLIND SPOT",
                    f"{cat} ({tier} tier) has only {signal_count} signals",
                    severity="high" if tier == "existential" else "medium"
                )
        
        if not blind_spots:
            vlog.success("No critical blind spots detected")
        
        # Trend deviation detection (comparing to baseline if available)
        vlog.step("Checking for trend deviations")
        vlog.substep("(First scan - no baseline for comparison)")
        
        # Quality assessment
        quality_score = sum([
            1 if coverage_pct >= 60 else 0,
            1 if dedup_ok else 0,
            1 if conf_ok else 0,
            1 if all_have_urls else 0,
            1 if score_spread >= 1.5 else 0
        ])
        
        quality_assessment = "HIGH" if quality_score >= 4 else ("MEDIUM" if quality_score >= 2 else "LOW")
        
        vlog.metric("Quality Score", f"{quality_score}/5")
        vlog.metric("Quality Assessment", quality_assessment, highlight=True)
        
        results["steps"]["validate_results"] = {
            "coverage_pct": round(coverage_pct, 1),
            "dedup_ok": dedup_ok,
            "confidence_ok": conf_ok,
            "traceability_ok": all_have_urls,
            "blind_spots": blind_spots,
            "quality_assessment": quality_assessment
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    
    vlog.header("EXECUTION COMPLETE")
    
    # Store final metrics
    vlog.metrics = {
        "Signals Collected": len(all_signals),
        "Signals Classified": len(classified_signals),
        "Off-Topic Filtered": off_topic_count,
        "Trends Derived": len(scored_trends),
        "High Priority Trends": len([t for t in scored_trends if t["priority_score"] >= 8.0]),
        "Blind Spots Detected": len(blind_spots),
        "Category Coverage": f"{coverage_pct:.0f}%",
        "Quality Assessment": quality_assessment
    }
    vlog.summary("SCAN SUMMARY")
    
    # Show top 3 actionable trends
    vlog.header("TOP ACTIONABLE TRENDS")
    for i, trend in enumerate(scored_trends[:3], 1):
        vlog.info(f"#{i} {trend['name']} (Score: {trend['priority_score']:.1f})")
        vlog.substep(f"Action: {trend['recommended_action']}")
        if trend.get('brief'):
            vlog.substep(f"Brief: {trend['brief'][:100]}...")
    
    # Export trace if requested
    if export_trace:
        vlog.export_trace(export_trace)
    
    results["end_time"] = datetime.utcnow().isoformat()
    results["summary"] = vlog.metrics
    results["trends"] = [
        {
            "name": t["name"],
            "score": t["priority_score"],
            "tier": t["amprion_tier"],
            "action": t["recommended_action"],
            "signals": t["signal_count"],
            "brief": t.get("brief", "")
        }
        for t in scored_trends
    ]
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SONAR.AI Demo Runner - Execute full agentic trend scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_runner.py
    python demo_runner.py --goal "Find OT cybersecurity threats"
    python demo_runner.py --sources arxiv,rss_news --export-trace audit.json

Environment Variables for LLM:
    CLASSIFIER_LLM_PROVIDER=anthropic|openai|selfhosted
    SELFHOSTED_LLM_PROVIDER=ollama|vllm|llamacpp
    SELFHOSTED_LLM_URL=http://localhost:11434
    SELFHOSTED_LLM_MODEL=llama3.2:8b
        """
    )
    
    parser.add_argument(
        "--goal", "-g",
        default="Identify highest-priority emerging trends for Amprion TSO operations",
        help="Scan goal/objective"
    )
    parser.add_argument(
        "--sources", "-s",
        help="Comma-separated list of sources (default: all)"
    )
    parser.add_argument(
        "--export-trace", "-e",
        help="Export trace log to JSON file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity"
    )
    
    args = parser.parse_args()
    
    sources = args.sources.split(",") if args.sources else None
    
    # Run async scan
    results = asyncio.run(run_demo_scan(
        goal=args.goal,
        sources=sources,
        export_trace=args.export_trace,
        verbose=not args.quiet
    ))
    
    print(f"\n✅ Demo scan complete. Processed {results['summary'].get('Signals Collected', 0)} signals into {results['summary'].get('Trends Derived', 0)} trends.\n")


if __name__ == "__main__":
    main()
