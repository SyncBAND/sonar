#!/usr/bin/env python3
"""
SONAR.AI / AGTS Full Pipeline
==============================
Complete workflow: Scrape → Process → Cluster → Score → Report

Usage:
    python full.py                    # Run full pipeline
    python full.py --scrape-only      # Only scrape data
    python full.py --process-only     # Only process existing signals
    python full.py --report           # Generate trend report
"""

import asyncio
import argparse
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path
sys.path.insert(0, '.')

from config import TSO_CATEGORIES, STRATEGIC_DOMAINS, AMPRION_BUSINESS_AREAS
from models.database import init_db, SessionLocal, Signal, Trend, TrendCluster
from services.nlp_processor import get_nlp_processor
from services.trend_scorer import get_trend_scorer
from services.scrapers import MasterScraper

# Taxonomy-based category names (preferred over config.py TSO_CATEGORIES)
try:
    from taxonomy import CATEGORIES as TAX_CATEGORIES
except ImportError:
    TAX_CATEGORIES = {}


def _cat_name(category: str) -> str:
    """Get human-readable category name from taxonomy or config fallback."""
    if category in TAX_CATEGORIES:
        return TAX_CATEGORIES[category]["name"]
    if category in TSO_CATEGORIES:
        return TSO_CATEGORIES[category].get("name", category)
    return category.replace("_", " ").title()

# =============================================================================
# PIPELINE STEPS
# =============================================================================

async def step_scrape(sources: List[str] = None) -> List[Dict]:
    """Step 1: Scrape data from all sources."""
    print("\n" + "=" * 60)
    print("STEP 1: SCRAPING DATA SOURCES")
    print("=" * 60)
    
    scraper = MasterScraper()
    signals = await scraper.scrape_all(sources)
    
    print(f"\n✅ Scraped {len(signals)} total signals")
    
    # Store in database
    db = SessionLocal()
    stored_count = 0
    
    for signal_data in signals:
        # Check for duplicate
        existing = db.query(Signal).filter(
            Signal.url == signal_data.get("url")
        ).first()
        
        if existing:
            continue
        
        signal = Signal(
            title=signal_data.get("title", "")[:500],
            content=signal_data.get("content", "")[:10000],
            url=signal_data.get("url", ""),
            source_type=signal_data.get("source_type"),
            source_name=signal_data.get("source_name"),
            source_quality_score=signal_data.get("source_quality_score", 0.5),
            published_at=signal_data.get("published_at"),
            scraped_at=signal_data.get("scraped_at", datetime.utcnow()),
            is_processed=False
        )
        db.add(signal)
        stored_count += 1
    
    db.commit()
    db.close()
    
    print(f"✅ Stored {stored_count} new signals in database")
    return signals


def step_process() -> int:
    """Step 2: Process signals with NLP."""
    print("\n" + "=" * 60)
    print("STEP 2: NLP PROCESSING")
    print("=" * 60)
    
    processor = get_nlp_processor()
    db = SessionLocal()
    
    # Get unprocessed signals
    signals = db.query(Signal).filter(Signal.is_processed == False).all()
    print(f"Found {len(signals)} unprocessed signals")
    
    processed_count = 0
    category_counts = {}
    off_topic_count = 0
    
    for i, signal in enumerate(signals):
        text = f"{signal.title} {signal.content or ''}"
        
        # Classify signal
        classification = processor.classify_signal(signal.title, signal.content or "")
        
        # Update signal
        signal.tso_category = classification["tso_category"]
        signal.strategic_domain = classification["strategic_domain"]
        signal.architectural_layer = classification["architectural_layer"]
        signal.amprion_task = classification["amprion_task"]
        signal.amprion_business_area = classification["business_area_code"]
        signal.linked_projects = classification["linked_projects"]
        signal.keywords = classification["keywords"]
        signal.entities = classification["entities"]
        
        # Calculate quality score (now includes domain relevance)
        signal.quality_score = processor.calculate_quality_score(
            signal.source_type or "news",
            signal.source_quality_score or 0.5,
            len(text),
            bool(classification["entities"]),
            domain_relevance=classification.get("domain_relevance", 1.0),
        )
        
        signal.is_processed = True
        processed_count += 1
        
        # Track categories and off-topic count
        cat = classification["tso_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if not classification.get("is_on_topic", True):
            off_topic_count += 1
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(signals)}...")
    
    db.commit()
    db.close()

    print(f"\n✅ Processed {processed_count} signals")
    if off_topic_count > 0:
        print(f"   ⚠️ {off_topic_count} signals flagged as off-topic (low domain relevance)")
    print("\n📊 Category Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"   {_cat_name(cat)}: {count}")
    
    return processed_count


def step_cluster(min_signals: int = 3) -> int:
    """Step 3: Create trend clusters."""
    print("\n" + "=" * 60)
    print("STEP 3: TREND CLUSTERING")
    print("=" * 60)
    
    db = SessionLocal()
    scorer = get_trend_scorer()
    
    # Get processed signals grouped by category
    signals = db.query(Signal).filter(Signal.is_processed == True).all()
    
    # Group by category
    category_signals = {}
    off_topic_filtered = 0
    for signal in signals:
        cat = signal.tso_category or "other"
        # Skip off-topic signals — they should not form trends
        if cat == "off_topic":
            off_topic_filtered += 1
            continue
        if cat not in category_signals:
            category_signals[cat] = []
        category_signals[cat].append(signal)
    
    print(f"Found {len(category_signals)} categories with processed signals")
    if off_topic_filtered:
        print(f"  (filtered {off_topic_filtered} off-topic signals)")
    
    # Get existing trend count for ID generation
    trend_counter = db.query(Trend).count()
    created_count = 0
    
    for category, cat_signals in category_signals.items():
        if len(cat_signals) < min_signals:
            print(f"  Skipping {category}: only {len(cat_signals)} signals")
            continue
        
        # Check if cluster already exists for this category
        existing_cluster = db.query(TrendCluster).filter(
            TrendCluster.tso_category == category
        ).first()
        
        if existing_cluster:
            # Update existing cluster
            cluster = existing_cluster
            # Update signal assignments
            for signal in cat_signals:
                signal.cluster_id = cluster.id
        else:
            # Create new cluster
            cluster = TrendCluster(
                name=_cat_name(category),
                description=f"Technology trends in {_cat_name(category)}",
                tso_category=category,
                keywords=list(set([
                    kw for s in cat_signals[:20] 
                    for kw in (s.keywords or [])[:5]
                ]))[:20]
            )
            db.add(cluster)
            db.flush()
            
            # Assign signals
            for signal in cat_signals:
                signal.cluster_id = cluster.id
        
        # Check if trend exists
        existing_trend = db.query(Trend).filter(
            Trend.cluster_id == cluster.id
        ).first()
        
        if existing_trend:
            trend = existing_trend
        else:
            trend_counter += 1
            trend_id = f"TR-{datetime.utcnow().year}-{trend_counter:03d}"
            
            # Collect metadata
            domains = [s.strategic_domain for s in cat_signals if s.strategic_domain]
            layers = [s.architectural_layer for s in cat_signals if s.architectural_layer]
            tasks = [s.amprion_task for s in cat_signals if s.amprion_task]
            projects = [p for s in cat_signals for p in (s.linked_projects or [])]
            
            # Prepare trend data for scoring
            trend_data = {
                "tso_category": category,
                "strategic_domain": max(set(domains), key=domains.count) if domains else None,
                "amprion_task": max(set(tasks), key=tasks.count) if tasks else None,
                "linked_projects": list(set(projects))[:5],
                "maturity_score": 5
            }
            
            signal_data = [
                {
                    "title": s.title,
                    "content": s.content,
                    "quality_score": s.quality_score,
                    "source_name": s.source_name,
                    "source_type": s.source_type,
                    "published_at": s.published_at,
                    "scraped_at": s.scraped_at
                }
                for s in cat_signals
            ]
            
            # Score the trend
            scores = scorer.score_trend(trend_data, signal_data)
            
            # Create trend
            trend = Trend(
                trend_id=trend_id,
                name=cluster.name,
                description_short=f"Emerging trends in {_cat_name(category)}",
                tso_category=category,
                strategic_domain=trend_data["strategic_domain"],
                architectural_layer=max(set(layers), key=layers.count) if layers else None,
                key_amprion_task=trend_data["amprion_task"],
                linked_projects=trend_data["linked_projects"],
                cluster_id=cluster.id,
                signal_count=len(cat_signals),
                
                # Scores
                priority_score=scores["priority_score"],
                strategic_relevance_score=scores["strategic_relevance_score"],
                grid_stability_score=scores["grid_stability_score"],
                cost_efficiency_score=scores["cost_efficiency_score"],
                volume_score=scores["volume_score"],
                growth_score=scores["growth_score"],
                quality_score=scores["quality_score"],
                growth_rate=scores["growth_rate"],
                project_multiplier=scores["project_multiplier"],
                
                # Classification (computed by scorer v2, not hardcoded)
                strategic_nature=scores["strategic_nature"],
                time_to_impact=scores["time_to_impact"],
                maturity_score=scores.get("maturity_score", 5),
                maturity_type=scores.get("maturity_type", "TRL"),
                lifecycle_status=scores.get("lifecycle_status", "Scouting"),
                
                is_active=True,
                is_emerging=len(cat_signals) < 20
            )
            db.add(trend)
            created_count += 1
        
        # Update existing trend signals count
        trend.signal_count = len(cat_signals)
    
    db.commit()
    db.close()
    
    print(f"\n✅ Created/updated {created_count} trends")
    return created_count


def step_score() -> int:
    """Step 4: Re-score all trends."""
    print("\n" + "=" * 60)
    print("STEP 4: TREND SCORING (No-Regret Framework)")
    print("=" * 60)
    
    db = SessionLocal()
    scorer = get_trend_scorer()
    
    trends = db.query(Trend).filter(Trend.is_active == True).all()
    print(f"Scoring {len(trends)} active trends")
    
    for trend in trends:
        signals = db.query(Signal).filter(Signal.cluster_id == trend.cluster_id).all()
        
        if not signals:
            continue
        
        signal_data = [
            {
                "title": s.title,
                "content": s.content,
                "quality_score": s.quality_score,
                "source_name": s.source_name,
                "source_type": s.source_type,
                "published_at": s.published_at,
                "scraped_at": s.scraped_at
            }
            for s in signals
        ]
        
        trend_data = {
            "tso_category": trend.tso_category,
            "strategic_domain": trend.strategic_domain,
            "amprion_task": trend.key_amprion_task,
            "linked_projects": trend.linked_projects or [],
            "maturity_score": trend.maturity_score or 5
        }
        
        scores = scorer.score_trend(trend_data, signal_data)
        
        # Update all scores (including computed maturity/lifecycle)
        trend.priority_score = scores["priority_score"]
        trend.strategic_relevance_score = scores["strategic_relevance_score"]
        trend.grid_stability_score = scores["grid_stability_score"]
        trend.cost_efficiency_score = scores["cost_efficiency_score"]
        trend.volume_score = scores["volume_score"]
        trend.growth_score = scores["growth_score"]
        trend.quality_score = scores["quality_score"]
        trend.growth_rate = scores["growth_rate"]
        trend.project_multiplier = scores["project_multiplier"]
        trend.strategic_nature = scores["strategic_nature"]
        trend.time_to_impact = scores["time_to_impact"]
        trend.maturity_score = scores.get("maturity_score", trend.maturity_score)
        trend.maturity_type = scores.get("maturity_type", "TRL")
        trend.lifecycle_status = scores.get("lifecycle_status", trend.lifecycle_status)
        trend.signal_count = len(signals)
        trend.last_updated = datetime.utcnow()
    
    db.commit()
    db.close()
    
    print(f"✅ Scored {len(trends)} trends")
    return len(trends)


def step_report():
    """Step 5: Generate trend report."""
    print("\n" + "=" * 60)
    print("STEP 5: TREND REPORT")
    print("=" * 60)
    
    db = SessionLocal()
    
    # Get statistics
    total_signals = db.query(Signal).count()
    processed_signals = db.query(Signal).filter(Signal.is_processed == True).count()
    total_trends = db.query(Trend).filter(Trend.is_active == True).count()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Total Signals: {total_signals}")
    print(f"   Processed: {processed_signals}")
    print(f"   Active Trends: {total_trends}")
    
    # Import scorer for monitoring analysis
    from services.trend_scorer import AMPRION_STRATEGIC_PRIORS, TrendScorer

    # Full trend ranking (all trends, not just top 10)
    all_ranked = db.query(Trend).filter(
        Trend.is_active == True
    ).order_by(Trend.priority_score.desc()).all()
    
    print(f"\n🏆 Full Trend Ranking ({len(all_ranked)} trends):")
    print("-" * 120)
    print(f"{'#':<3} {'Trend Name':<30} {'Score':>5} {'TRL':>4} "
          f"{'Lifecycle':<15} {'Nature':<16} {'Impact':<10} {'Sig':>4} {'Tier':<12}")
    print("-" * 120)
    
    scorer = TrendScorer()
    blind_spots = []
    
    for i, trend in enumerate(all_ranked, 1):
        cat = trend.tso_category or "other"
        prior = AMPRION_STRATEGIC_PRIORS.get(cat, {})
        tier = prior.get("tier", "–")
        
        # Detect blind spots: tier-aware thresholds matching scorer logic
        sig_count = trend.signal_count or 0
        if tier in ("existential", "critical"):
            is_blind = sig_count < 15
        elif tier == "high":
            is_blind = sig_count < 10
        else:
            is_blind = False
        flag = " ⚠" if is_blind else ""
        if is_blind:
            blind_spots.append((trend.name, sig_count, tier))
        
        print(f"{i:<3} {trend.name[:29]:<30} {trend.priority_score:>5.1f} "
              f"{trend.maturity_score or 5:>4} "
              f"{(trend.lifecycle_status or 'Scouting'):<15} "
              f"{(trend.strategic_nature or 'N/A'):<16} "
              f"{(trend.time_to_impact or 'N/A'):<10} "
              f"{trend.signal_count or 0:>4} "
              f"{tier:<12}{flag}")
    
    # Blind spot alerts
    if blind_spots:
        print(f"\n⚠️  BLIND SPOTS ({len(blind_spots)} detected):")
        print("   Strategic categories with insufficient signal coverage:")
        for name, count, tier in blind_spots:
            print(f"   ⚠ {name} ({tier.upper()}) — only {count} signals. "
                  "Add dedicated RSS feeds or increase monitoring.")
    
    # Amprion framework cross-reference
    print(f"\n🔍 Amprion Framework Cross-Reference:")
    print("-" * 90)
    print(f"   {'Category':<30} {'Tier':<12} {'AGTS Score':>10} {'Signals':>8} {'Status':<15}")
    print("-" * 90)
    
    for tier_name in ["existential", "critical", "high", "medium-high", "medium"]:
        for cat, prior in AMPRION_STRATEGIC_PRIORS.items():
            if prior.get("tier") != tier_name:
                continue
            trend = next((t for t in all_ranked if t.tso_category == cat), None)
            if trend:
                print(f"   {trend.name[:29]:<30} {tier_name:<12} "
                      f"{trend.priority_score:>10.1f} {trend.signal_count or 0:>8} "
                      f"{trend.lifecycle_status or 'N/A':<15}")
            else:
                print(f"   {cat:<30} {tier_name:<12} {'N/A':>10} {'0':>8} "
                      f"{'No trend':<15}")
    
    # Category breakdown (compact)
    print(f"\n📈 Trends by Category:")
    for cat_id, cat_data in sorted(TSO_CATEGORIES.items(), key=lambda x: x[1].get("name", "")):
        count = db.query(Trend).filter(
            Trend.tso_category == cat_id,
            Trend.is_active == True
        ).count()
        if count > 0:
            print(f"   {cat_data['name']}: {count} trends")
    
    # Strategic nature breakdown
    print(f"\n⚡ By Strategic Nature:")
    for nature in ["Accelerator", "Disruptor", "Transformational"]:
        count = db.query(Trend).filter(
            Trend.strategic_nature == nature,
            Trend.is_active == True
        ).count()
        print(f"   {nature}: {count}")
    
    # Time to impact
    print(f"\n⏱️ By Time to Impact:")
    for impact in ["<1 year", "1-3 years", "3-5 years", "5+ years"]:
        count = db.query(Trend).filter(
            Trend.time_to_impact == impact,
            Trend.is_active == True
        ).count()
        print(f"   {impact}: {count}")
    
    # Lifecycle status breakdown
    print(f"\n🔬 By Lifecycle (computed from TRL):")
    for status in ["Scouting", "Pilot", "Implementation", "Standard"]:
        count = db.query(Trend).filter(
            Trend.lifecycle_status == status,
            Trend.is_active == True
        ).count()
        if count > 0:
            print(f"   {status}: {count}")
    
    # Score spread summary
    if all_ranked:
        scores = [t.priority_score for t in all_ranked if t.priority_score]
        trls = [t.maturity_score for t in all_ranked if t.maturity_score]
        if scores:
            print(f"\n📊 Score Diagnostics:")
            print(f"   Priority range: {min(scores):.1f} – {max(scores):.1f}")
            print(f"   Unique scores: {len(set(round(s, 1) for s in scores))} / {len(scores)}")
        if trls:
            print(f"   TRL range: {min(trls)} – {max(trls)}")
    
    db.close()
    
    print("\n" + "=" * 60)
    print("REPORT COMPLETE")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="SONAR.AI / AGTS Pipeline")
    parser.add_argument("--scrape-only", action="store_true", help="Only run scraping")
    parser.add_argument("--process-only", action="store_true", help="Only run NLP processing")
    parser.add_argument("--cluster-only", action="store_true", help="Only run clustering")
    parser.add_argument("--score-only", action="store_true", help="Only run scoring")
    parser.add_argument("--report", action="store_true", help="Only generate report")
    parser.add_argument("--sources", nargs="+", help="Specific sources to scrape")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🚀 SONAR.AI / AGTS - Agentic Global Trend Scanner")
    print("=" * 60)
    print(f"Started at: {datetime.utcnow().isoformat()}")
    
    # Initialize database
    init_db()
    
    if args.report:
        step_report()
        return
    
    if args.scrape_only:
        await step_scrape(args.sources)
        return
    
    if args.process_only:
        step_process()
        return
    
    if args.cluster_only:
        step_cluster()
        return
    
    if args.score_only:
        step_score()
        return
    
    # Full pipeline
    print("\n🔄 Running full pipeline...")
    
    await step_scrape(args.sources)
    step_process()
    step_cluster()
    step_score()
    step_report()
    
    print(f"\n✅ Pipeline completed at: {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
