#!/usr/bin/env python3
"""
TSO Coverage Upgrade — Verification Script
============================================
Run this to verify all TSO scraper changes work on your machine.

Usage:
    python test_tso_upgrade.py              # Full test (requires internet)
    python test_tso_upgrade.py --offline    # Structural tests only
"""

import asyncio
import sys
import argparse

sys.path.insert(0, '.')

from services.scrapers import (
    TSOPublicationsScraper, RSSNewsScraper, MasterScraper
)


def test_structural():
    """Test structural correctness (no network needed)."""
    print("=" * 60)
    print("STRUCTURAL TESTS (offline)")
    print("=" * 60)
    passed = 0
    failed = 0
    
    # 1. SMARD not duplicated
    rss = RSSNewsScraper()
    if 'smard_bnetza' not in rss.RSS_FEEDS:
        print("✅ smard_bnetza removed from RSS_FEEDS (no duplication)")
        passed += 1
    else:
        print("❌ smard_bnetza still in RSS_FEEDS!")
        failed += 1
    
    # 2. TSO scraper has all expected sources
    tso = TSOPublicationsScraper()
    expected_news = {"amprion", "tennet_de", "50hertz", "transnetbw", "entsoe"}
    actual_news = set(tso.TSO_NEWS_SOURCES.keys())
    if expected_news == actual_news:
        print(f"✅ TSO_NEWS_SOURCES has all 5 sources: {', '.join(sorted(actual_news))}")
        passed += 1
    else:
        print(f"❌ TSO_NEWS_SOURCES mismatch: expected {expected_news}, got {actual_news}")
        failed += 1
    
    # 3. BNetzA Netzausbau in MARKET_RSS_FEEDS
    if 'bnetza_netzausbau' in tso.MARKET_RSS_FEEDS:
        print("✅ BNetzA Netzausbau RSS added to MARKET_RSS_FEEDS")
        passed += 1
    else:
        print("❌ bnetza_netzausbau missing from MARKET_RSS_FEEDS!")
        failed += 1
    
    # 4. SMARD still in MARKET_RSS_FEEDS
    if 'smard' in tso.MARKET_RSS_FEEDS:
        print("✅ SMARD still in MARKET_RSS_FEEDS (single source of truth)")
        passed += 1
    else:
        print("❌ SMARD missing from MARKET_RSS_FEEDS!")
        failed += 1
    
    # 5. All German TSOs have Google News queries
    for sid, cfg in tso.TSO_NEWS_SOURCES.items():
        if cfg.get('google_news_query'):
            print(f"  ✅ {sid} has Google News query")
            passed += 1
        else:
            print(f"  ❌ {sid} missing Google News query!")
            failed += 1
    
    # 6. ENTSO-E has both direct RSS and Google News fallback
    entsoe = tso.TSO_NEWS_SOURCES.get("entsoe", {})
    if entsoe.get('rss_url') and entsoe.get('google_news_query'):
        print("✅ ENTSO-E has both direct RSS and Google News fallback")
        passed += 1
    else:
        print("❌ ENTSO-E missing strategy!")
        failed += 1
    
    # 7. Dedup works
    ms = MasterScraper()
    sigs = [
        {'url': 'https://a.com', 'title': 'A'},
        {'url': 'https://b.com', 'title': 'B'},
        {'url': 'https://a.com', 'title': 'A dup'},
    ]
    deduped = ms._dedup_by_url(sigs)
    if len(deduped) == 2:
        print("✅ Cross-scraper URL dedup works (3→2)")
        passed += 1
    else:
        print(f"❌ Dedup returned {len(deduped)}, expected 2")
        failed += 1
    
    # 8. MasterScraper has TSO scraper
    if 'tso' in ms.scrapers and isinstance(ms.scrapers['tso'], TSOPublicationsScraper):
        print("✅ MasterScraper includes TSO scraper")
        passed += 1
    else:
        print("❌ TSO scraper missing from MasterScraper!")
        failed += 1
    
    # 9. RSS feed count
    print(f"\n  RSS feeds: {len(rss.RSS_FEEDS)}")
    print(f"  TSO news sources: {len(tso.TSO_NEWS_SOURCES)}")
    print(f"  Market RSS feeds: {len(tso.MARKET_RSS_FEEDS)}")
    
    print(f"\n{'='*60}")
    print(f"Structural: {passed} passed, {failed} failed")
    return failed == 0


async def test_network():
    """Test actual network scraping."""
    print("\n" + "=" * 60)
    print("NETWORK TESTS (requires internet)")
    print("=" * 60)
    
    tso = TSOPublicationsScraper()
    import aiohttp
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        
        # 1. Test BNetzA Netzausbau RSS (verified working)
        print("\n--- BNetzA Netzausbau RSS ---")
        signals = await tso._fetch_rss(
            session,
            "https://www.netzausbau.de/SiteGlobals/Functions/RSSFeed/DE/RSSAllgemein/RSSGesamt.xml",
            "bnetza_netzausbau", "BNetzA Netzausbau", 0.95, "tso_market"
        )
        if signals:
            print(f"  ✅ {len(signals)} items")
            print(f"  Sample: {signals[0]['title'][:70]}")
        else:
            print("  ❌ No items (check your internet)")
        
        # 2. Test SMARD RSS
        print("\n--- SMARD RSS ---")
        signals = await tso._fetch_rss(
            session,
            "https://www.smard.de/service/rss/en/feed.rss",
            "smard", "SMARD (BNetzA)", 0.95, "tso_market"
        )
        if signals:
            print(f"  ✅ {len(signals)} items")
        else:
            print("  ❌ No items")
        
        # 3. Test Google News RSS for Amprion
        print("\n--- Google News RSS (Amprion) ---")
        gn_query = tso.TSO_NEWS_SOURCES["amprion"]["google_news_query"].replace(" ", "+")
        gn_url = tso.GOOGLE_NEWS_RSS.format(query=gn_query)
        signals = await tso._fetch_rss(
            session, gn_url, "amprion_gnews",
            "Amprion", 0.85, "tso_news", max_items=10
        )
        if signals:
            print(f"  ✅ {len(signals)} items")
            print(f"  Sample: {signals[0]['title'][:70]}")
        else:
            print("  ⚠️  No items (Google News may throttle — try again)")
        
        # 4. Test curl fallback (power_magazine)
        print("\n--- Curl fallback (power_magazine) ---")
        content = await tso._curl_fallback("https://www.powermag.com/feed/", "power_magazine")
        if content and "<rss" in content[:500]:
            import feedparser
            feed = feedparser.parse(content)
            print(f"  ✅ curl got {len(feed.entries)} items from power_magazine")
        else:
            print("  ❌ curl fallback failed for power_magazine")
    
    # 5. Full TSO scraper test
    print("\n--- Full TSO Scraper ---")
    all_signals = await tso.scrape()
    print(f"  Total TSO signals: {len(all_signals)}")
    
    # Count by source
    sources = {}
    for s in all_signals:
        name = s.get('source_name', 'unknown')
        sources[name] = sources.get(name, 0) + 1
    
    for name, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}")
    
    return len(all_signals) > 0


async def test_full_scrape():
    """Run full scrape and show TSO signal counts."""
    print("\n" + "=" * 60)
    print("FULL SCRAPE TEST")
    print("=" * 60)
    
    scraper = MasterScraper()
    signals = await scraper.scrape_all()
    
    print(f"\nTotal signals: {len(signals)}")
    
    # Count TSO-related
    tso_types = {"tso", "tso_news", "tso_market"}
    tso_signals = [s for s in signals if s.get("source_type") in tso_types]
    print(f"TSO signals: {len(tso_signals)} ({100*len(tso_signals)/max(len(signals),1):.1f}%)")
    
    # Breakdown
    by_source = {}
    for s in tso_signals:
        name = s.get("source_name", "unknown")
        by_source[name] = by_source.get(name, 0) + 1
    
    print("\nTSO signal breakdown:")
    for name, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Structural tests only")
    parser.add_argument("--full", action="store_true", help="Run full scrape")
    args = parser.parse_args()
    
    ok = test_structural()
    
    if not args.offline:
        net_ok = asyncio.run(test_network())
        if args.full:
            asyncio.run(test_full_scrape())
    
    print("\n" + "=" * 60)
    print("DONE" if ok else "SOME TESTS FAILED")
    print("=" * 60)
