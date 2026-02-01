"""
SONAR.AI / AGTS Data Scrapers
==============================
Multi-source data collection for TSO trend intelligence.

Sources:
- Research: arXiv
- News: RSS feeds (Utility Dive, PV Magazine, Energy Storage News, etc.)
- Regulatory: ENTSO-E, BNetzA
- TSO Publications: Amprion, TenneT, 50Hertz
- Industry Events: Enlit, E-world
- Startups: Crunchbase
"""

import asyncio
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re
import json
import logging

from config import DATA_SOURCES, TSO_CATEGORIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scrapers")

# =============================================================================
# BASE SCRAPER
# =============================================================================

class BaseScraper:
    """Base class for all scrapers."""
    
    def __init__(self, source_id: str):
        self.source_id = source_id
        self.source_config = DATA_SOURCES.get(source_id, {})
        self.source_name = self.source_config.get("name", source_id)
        self.source_type = self.source_config.get("type", "news")
        self.quality_score = self.source_config.get("quality_score", 0.5)
    
    async def scrape(self) -> List[Dict[str, Any]]:
        """Override in subclasses."""
        raise NotImplementedError
    
    def _create_signal(
        self,
        title: str,
        content: str = "",
        url: str = "",
        published_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Create standardized signal dict."""
        return {
            "title": title,
            "content": content,
            "url": url,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_quality_score": self.quality_score,
            "published_at": published_at,
            "scraped_at": datetime.utcnow()
        }

# =============================================================================
# ARXIV SCRAPER (Research)
# =============================================================================

class ArxivScraper(BaseScraper):
    """Scrape arXiv for energy/grid research papers."""
    
    def __init__(self):
        super().__init__("arxiv")
        self.base_url = "http://export.arxiv.org/api/query"
        
        # TSO-relevant search queries
        self.search_queries = [
            # Grid & Power Systems
            "power grid stability",
            "transmission system operator",
            "HVDC transmission",
            "grid-forming inverter",
            "frequency control power system",
            "voltage stability grid",
            "renewable integration grid",
            "energy storage grid",
            
            # AI for Energy
            "machine learning power system",
            "deep learning grid",
            "reinforcement learning energy",
            "neural network forecasting energy",
            "AI grid optimization",
            
            # Flexibility & Markets
            "demand response",
            "virtual power plant",
            "flexibility aggregation",
            "electricity market",
            "vehicle to grid",
            
            # Sector Coupling
            "power to gas",
            "hydrogen electrolysis",
            "green hydrogen",
            
            # Cybersecurity
            "smart grid security",
            "SCADA cybersecurity",
            "post-quantum cryptography"
        ]
    
    async def scrape(self, max_results_per_query: int = 20) -> List[Dict[str, Any]]:
        """Scrape arXiv API for relevant papers."""
        signals = []
        
        async with aiohttp.ClientSession() as session:
            for query in self.search_queries:
                try:
                    params = {
                        "search_query": f"all:{query}",
                        "start": 0,
                        "max_results": max_results_per_query,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending"
                    }
                    
                    async with session.get(self.base_url, params=params) as response:
                        if response.status == 200:
                            content = await response.text()
                            signals.extend(self._parse_arxiv_response(content))
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"arXiv scrape error for '{query}': {e}")
        
        # Deduplicate by URL
        seen_urls = set()
        unique_signals = []
        for signal in signals:
            if signal["url"] not in seen_urls:
                seen_urls.add(signal["url"])
                unique_signals.append(signal)
        
        logger.info(f"arXiv: Scraped {len(unique_signals)} papers")
        return unique_signals
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv API XML response."""
        signals = []
        soup = BeautifulSoup(xml_content, "xml")
        
        for entry in soup.find_all("entry"):
            try:
                title = entry.find("title").text.strip().replace("\n", " ")
                summary = entry.find("summary").text.strip().replace("\n", " ")
                
                # Get arXiv ID and URL
                arxiv_id = entry.find("id").text.strip()
                url = arxiv_id
                
                # Parse date
                published = entry.find("published").text.strip()
                published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))
                
                # Get categories
                categories = [cat.get("term") for cat in entry.find_all("category")]
                
                # Get authors
                authors = [author.find("name").text for author in entry.find_all("author")]
                
                signal = self._create_signal(
                    title=title,
                    content=f"{summary}\n\nAuthors: {', '.join(authors[:5])}\nCategories: {', '.join(categories)}",
                    url=url,
                    published_at=published_at
                )
                signals.append(signal)
                
            except Exception as e:
                logger.debug(f"Error parsing arXiv entry: {e}")
        
        return signals

# =============================================================================
# RSS SCRAPER (News)
# =============================================================================

class RSSNewsScraper(BaseScraper):
    """Scrape RSS feeds from energy news sources."""
    
    # Energy news RSS feeds - ALL VERIFIED WORKING
    # 
    # Status: ✅ = confirmed working in test run
    #         🔧 = URL fixed from broken version
    #         🆕 = newly added source
    #         ❌ = removed (dead/paywalled/DNS error)
    #
    # Removed (non-working):
    #   recharge_news (404), power_magazine (403), bioenergy_news (0 items),
    #   cyberalerts (404), sustainable_bus (0 items), acea_auto (0 items),
    #   euractiv_energy (403), politico_eu_tech (0 items), tdworld (404),
    #   smart_energy_intl (503), renewables_now (403), eu_champions_alliance (DNS),
    #   jao_messages (404 - no public RSS), nord_pool_umm (0 items)
    #
    RSS_FEEDS = {
        # ─── GENERAL ENERGY NEWS (all ✅ verified) ────────────────
        "utility_dive": "https://www.utilitydive.com/feeds/news/",
        "pv_magazine": "https://www.pv-magazine.com/feed/",
        "energy_storage_news": "https://www.energy-storage.news/feed/",
        "electrek": "https://electrek.co/feed/",
        "clean_technica": "https://cleantechnica.com/feed/",
        
        # ─── OFFSHORE WIND (all ✅ verified) ──────────────────────
        "offshore_wind_biz": "https://www.offshorewind.biz/feed/",
        "windpower_monthly": "https://www.windpowermonthly.com/rss",
        "wind_europe": "https://windeurope.org/feed/",
        
        # ─── GERMAN ENERGY / TSO-SPECIFIC ─────────────────────────
        "clean_energy_wire": "https://www.cleanenergywire.org/rss.xml",  # ✅
        # smard_bnetza moved to TSOPublicationsScraper.MARKET_RSS_FEEDS (avoid duplication)
        
        # ─── BIOGAS & BIOMETHANE ──────────────────────────────────
        "european_biogas": "https://www.europeanbiogas.eu/feed/",  # ✅
        "bioenergy_intl": "https://bioenergyinternational.com/feed/",  # ✅
        
        # ─── HYDROGEN & P2G ──────────────────────────────────────
        "fuel_cells_works": "https://fuelcellsworks.com/feed/",  # ✅
        "hydrogen_central": "https://hydrogen-central.com/feed/",  # ✅
        
        # ─── CYBERSECURITY OT/ICS (HIGH VALUE FOR TSO!) ──────────
        "ics_cert": "https://www.cisa.gov/cybersecurity-advisories/ics-advisories.xml",  # 🔧 fixed URL
        "cisa_all": "https://www.cisa.gov/cybersecurity-advisories/all.xml",  # 🆕 all CISA advisories
        "dark_reading": "https://www.darkreading.com/rss.xml",  # ✅
        "the_record": "https://therecord.media/feed",  # ✅
        "ncsc_nl": "https://advisories.ncsc.nl/rss/advisories",  # 🔧 fixed URL (was wrong domain)
        "cert_eu": "https://cert.europa.eu/publications/security-advisories-rss",  # 🔧 fixed URL
        "cert_eu_threat": "https://cert.europa.eu/publications/threat-intelligence-rss",  # 🆕 threat intel
        "decent_cybersecurity": "https://decentcybersecurity.eu/feed/",  # ✅
        "industrial_cyber": "https://industrialcyber.co/feed/",  # 🆕 OT/ICS news
        
        # ─── E-MOBILITY & V2G ────────────────────────────────────
        "charged_evs": "https://chargedevs.com/feed/",  # ✅
        "electrive": "https://www.electrive.com/feed/",  # ✅
        "e_motec": "https://e-motec.net/feed/",  # ✅
        "the_driven": "https://thedriven.io/feed/",  # 🆕 EV news (Australia/Europe)
        "insideevs": "https://insideevs.com/rss/news/all/",  # 🆕 global EV coverage
        
        # ─── REGULATORY & POLICY ─────────────────────────────────
        "carbon_brief": "https://www.carbonbrief.org/feed/",  # ✅
        "energy_transition": "https://energytransition.org/feed/",  # 🔧 replaces euractiv_sections (Cloudflare 403)
        "gie_news": "https://www.gie.eu/feed/",  # 🆕 Gas Infrastructure Europe
        
        # ─── SMART GRID / POWER / T&D ────────────────────────────
        "power_magazine": "https://powermag.com/feed/",  # 🔧 replaces power_technology (Cloudflare 403)
        "ecowatch_energy": "https://ecowatch.com/energy-news/feed",  # 🔧 replaces renew_energy_world (Cloudflare 403)
        "mit_energy": "https://energy.mit.edu/news/feed/",  # 🔧 replaces canary_media (Cloudflare 404)
        
        # ─── AI & TECHNOLOGY ─────────────────────────────────────
        # (Datafloq and LastWeekInAI removed — general tech aggregators
        #  that produced 30-40 off-topic signals misclassified into
        #  ai_grid_optimization, regulatory_policy, flexibility_vpp)
        
        # ─── TSO / ENERGY MARKET NEWS ─────────────────────────────
        "eia_today": "https://www.eia.gov/rss/todayinenergy.xml",  # 🔧 replaces energy_central (404)
        "renew_economy": "https://reneweconomy.com.au/feed/",  # 🆕 renewables/grid
    }
    
    def __init__(self, source_id: str = "rss_news"):
        super().__init__(source_id)
        self.source_type = "news"
        self.quality_score = 0.75
    
    async def _curl_fallback(self, feed_url: str, feed_id: str) -> Optional[str]:
        """Use curl as fallback for Cloudflare-protected feeds that block aiohttp TLS."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "curl", "-sL", "--max-time", "30", "--compressed",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "-H", "Accept: application/rss+xml, application/xml, text/xml, */*",
                feed_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=35)
            if proc.returncode == 0 and stdout:
                content = stdout.decode("utf-8", errors="replace")
                if "<rss" in content[:500] or "<feed" in content[:500] or "<channel" in content[:1000]:
                    return content
        except Exception as e:
            logger.debug(f"curl fallback failed for {feed_id}: {e}")
        return None
    
    async def scrape(self, feeds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scrape multiple RSS feeds."""
        signals = []
        feeds_to_scrape = feeds or list(self.RSS_FEEDS.keys())
        
        # Browser-like User-Agent prevents 403 blocks from sites that reject bots
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        # Slow feeds (large XML) get extra time
        SLOW_FEEDS = {"ics_cert", "cisa_all"}
        default_timeout = aiohttp.ClientTimeout(total=45, connect=15)
        slow_timeout = aiohttp.ClientTimeout(total=90, connect=20)
        
        async with aiohttp.ClientSession(headers=headers) as session:
            for feed_id in feeds_to_scrape:
                if feed_id not in self.RSS_FEEDS:
                    continue
                
                feed_url = self.RSS_FEEDS[feed_id]
                feed_timeout = slow_timeout if feed_id in SLOW_FEEDS else default_timeout
                
                try:
                    async with session.get(feed_url, timeout=feed_timeout) as response:
                        if response.status == 200:
                            content = await response.text()
                            feed_signals = self._parse_feed(content, feed_id)
                            signals.extend(feed_signals)
                            logger.info(f"RSS ({feed_id}): {len(feed_signals)} items")
                        elif response.status == 403:
                            # Cloudflare TLS fingerprinting — fall back to curl
                            content = await self._curl_fallback(feed_url, feed_id)
                            if content:
                                feed_signals = self._parse_feed(content, feed_id)
                                signals.extend(feed_signals)
                                logger.info(f"RSS ({feed_id}): {len(feed_signals)} items (curl fallback)")
                            else:
                                logger.warning(f"RSS ({feed_id}): HTTP 403 (blocked, curl fallback also failed)")
                        elif response.status == 301 or response.status == 302:
                            # Follow redirect manually if aiohttp didn't
                            redirect_url = response.headers.get("Location", "")
                            if redirect_url:
                                async with session.get(redirect_url, timeout=feed_timeout) as r2:
                                    if r2.status == 200:
                                        content = await r2.text()
                                        feed_signals = self._parse_feed(content, feed_id)
                                        signals.extend(feed_signals)
                                        logger.info(f"RSS ({feed_id}): {len(feed_signals)} items (redirected)")
                        else:
                            logger.warning(f"RSS ({feed_id}): HTTP {response.status}")
                
                except asyncio.TimeoutError:
                    logger.warning(f"RSS ({feed_id}): Timeout after {90 if feed_id in SLOW_FEEDS else 45}s")
                except Exception as e:
                    logger.error(f"RSS scrape error for {feed_id}: {type(e).__name__}: {e}")
                
                await asyncio.sleep(0.3)
        
        logger.info(f"RSS Total: {len(signals)} signals")
        return signals
    
    def _parse_feed(self, content: str, feed_id: str) -> List[Dict[str, Any]]:
        """Parse RSS feed content."""
        signals = []
        
        try:
            feed = feedparser.parse(content)
            
            for entry in feed.entries[:50]:  # Limit per feed
                title = entry.get("title", "")
                
                # Get content
                content_text = ""
                if "summary" in entry:
                    content_text = entry.summary
                elif "description" in entry:
                    content_text = entry.description
                
                # Clean HTML
                content_text = BeautifulSoup(content_text, "html.parser").get_text()
                
                # Parse date
                published_at = None
                if "published_parsed" in entry and entry.published_parsed:
                    published_at = datetime(*entry.published_parsed[:6])
                elif "updated_parsed" in entry and entry.updated_parsed:
                    published_at = datetime(*entry.updated_parsed[:6])
                
                signal = {
                    "title": title,
                    "content": content_text[:2000],
                    "url": entry.get("link", ""),
                    "source_type": "news",
                    "source_name": feed_id.replace("_", " ").title(),
                    "source_quality_score": 0.75,
                    "published_at": published_at,
                    "scraped_at": datetime.utcnow()
                }
                signals.append(signal)
        
        except Exception as e:
            logger.error(f"Feed parse error ({feed_id}): {e}")
        
        return signals

# =============================================================================
# TSO PUBLICATIONS SCRAPER
# =============================================================================

class TSOPublicationsScraper(BaseScraper):
    """
    Scrape TSO news via multiple strategies (reusable, config-driven).
    
    Strategy order per source:
      1. Direct RSS feed (if available — fastest, most reliable)
      2. Google News RSS proxy (works for any entity — just add a search query)
      3. curl fallback for Cloudflare-protected RSS (same as RSSNewsScraper)
    
    To add a new TSO or entity:
      - Add an entry to TSO_NEWS_SOURCES with at least a 'name' and one strategy
      - Strategies: 'rss_url' for direct RSS, 'google_news_query' for Google News proxy
    """
    
    # ─── REUSABLE CONFIG: add/remove entities by editing this dict ──────
    TSO_NEWS_SOURCES = {
        # ── German TSOs (Google News proxy — their press pages are JS-rendered SPAs) ──
        "amprion": {
            "name": "Amprion",
            "google_news_query": '"Amprion" grid OR transmission OR Netzausbau OR offshore OR Konverter',
            "quality_score": 0.85,
        },
        "tennet_de": {
            "name": "TenneT Germany",
            "google_news_query": '"TenneT" Germany grid OR transmission OR offshore OR Netzausbau',
            "quality_score": 0.85,
        },
        "50hertz": {
            "name": "50Hertz",
            "google_news_query": '"50Hertz" grid OR transmission OR offshore OR renewable OR STATCOM',
            "quality_score": 0.85,
        },
        "transnetbw": {
            "name": "TransnetBW",
            "google_news_query": '"TransnetBW" grid OR transmission OR Netzausbau OR SuedLink',
            "quality_score": 0.85,
        },
        # ── European TSO bodies (direct RSS where available) ──
        "entsoe": {
            "name": "ENTSO-E",
            "rss_url": "https://www.entsoe.eu/rss/news.xml",
            "google_news_query": '"ENTSO-E" grid OR electricity OR transmission OR regulation',
            "quality_score": 0.95,
        },
    }
    
    # ─── TSO market data & regulatory RSS feeds (direct, verified) ──────
    # Add new sources by adding entries here — the scraper loop handles them all.
    MARKET_RSS_FEEDS = {
        "smard": {
            "name": "SMARD (BNetzA)",
            "rss_url": "https://www.smard.de/service/rss/en/feed.rss",
            "quality_score": 0.95,
            "description": "German electricity market data, renewables, wholesale prices"
        },
        "bnetza_netzausbau": {
            "name": "BNetzA Netzausbau",
            "rss_url": "https://www.netzausbau.de/SiteGlobals/Functions/RSSFeed/DE/RSSAllgemein/RSSGesamt.xml",
            "quality_score": 0.95,
            "description": "German grid expansion approvals, corridor decisions, construction milestones"
        },
    }
    
    # Google News RSS base URL — works globally, just change hl/gl/ceid for locale
    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en&gl=DE&ceid=DE:en"
    
    def __init__(self):
        super().__init__("tso_publications")
        self.source_type = "tso"
    
    async def _fetch_rss(self, session: aiohttp.ClientSession, url: str, source_id: str,
                         source_name: str, quality_score: float, source_type: str = "tso",
                         max_items: int = 30) -> List[Dict[str, Any]]:
        """
        Generic RSS fetch+parse. Reusable for any RSS URL.
        Falls back to curl if aiohttp gets 403 (Cloudflare TLS fingerprinting).
        """
        signals = []
        content = None
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30, connect=10)) as response:
                if response.status == 200:
                    content = await response.text()
                elif response.status == 403:
                    # Cloudflare TLS fingerprinting — try curl
                    content = await self._curl_fallback(url, source_id)
                else:
                    logger.warning(f"TSO RSS ({source_id}): HTTP {response.status}")
        except asyncio.TimeoutError:
            logger.warning(f"TSO RSS ({source_id}): Timeout, trying curl")
            content = await self._curl_fallback(url, source_id)
        except Exception as e:
            logger.warning(f"TSO RSS ({source_id}): {type(e).__name__}: {e}")
            content = await self._curl_fallback(url, source_id)
        
        if not content:
            return signals
        
        try:
            feed = feedparser.parse(content)
            for entry in feed.entries[:max_items]:
                pub_date = None
                if entry.get("published_parsed"):
                    pub_date = datetime(*entry.published_parsed[:6])
                elif entry.get("updated_parsed"):
                    pub_date = datetime(*entry.updated_parsed[:6])
                
                signal = self._create_signal(
                    title=entry.get("title", ""),
                    content=BeautifulSoup(
                        entry.get("summary", entry.get("description", "")),
                        "html.parser"
                    ).get_text()[:2000],
                    url=entry.get("link", ""),
                    published_at=pub_date
                )
                signal["source_name"] = source_name
                signal["source_quality_score"] = quality_score
                signal["source_type"] = source_type
                signals.append(signal)
        except Exception as e:
            logger.warning(f"TSO RSS parse ({source_id}): {e}")
        
        return signals
    
    async def _curl_fallback(self, url: str, source_id: str) -> Optional[str]:
        """Use curl as fallback for Cloudflare-protected feeds."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "curl", "-sL", "--max-time", "25", "--compressed",
                "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "-H", "Accept: application/rss+xml, application/xml, text/xml, */*",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0 and stdout:
                text = stdout.decode("utf-8", errors="replace")
                if "<rss" in text[:500] or "<feed" in text[:500] or "<channel" in text[:1000]:
                    return text
        except Exception as e:
            logger.debug(f"TSO curl fallback ({source_id}): {e}")
        return None
    
    async def scrape(self) -> List[Dict[str, Any]]:
        """Scrape all TSO sources using strategy cascade: direct RSS → Google News RSS."""
        signals = []
        seen_urls = set()  # Dedup within this scraper run
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            # ── 1. TSO News Sources (direct RSS → Google News fallback) ──
            for source_id, config in self.TSO_NEWS_SOURCES.items():
                source_signals = []
                
                # Strategy 1: Direct RSS feed (if configured)
                rss_url = config.get("rss_url")
                if rss_url:
                    source_signals = await self._fetch_rss(
                        session, rss_url, source_id,
                        config["name"], config["quality_score"],
                        source_type="tso"
                    )
                    if source_signals:
                        logger.info(f"TSO News ({source_id}): {len(source_signals)} items (direct RSS)")
                
                # Strategy 2: Google News RSS proxy (if direct RSS failed or not configured)
                if not source_signals and config.get("google_news_query"):
                    query = config["google_news_query"].replace(" ", "+")
                    gn_url = self.GOOGLE_NEWS_RSS.format(query=query)
                    source_signals = await self._fetch_rss(
                        session, gn_url, f"{source_id}_gnews",
                        config["name"], config["quality_score"],
                        source_type="tso_news", max_items=20
                    )
                    if source_signals:
                        logger.info(f"TSO News ({source_id}): {len(source_signals)} items (Google News)")
                
                if not source_signals:
                    logger.warning(f"TSO News ({source_id}): No items from any strategy")
                
                # Dedup by URL within this run
                for sig in source_signals:
                    if sig["url"] not in seen_urls:
                        seen_urls.add(sig["url"])
                        signals.append(sig)
                
                await asyncio.sleep(0.5)  # Rate limit Google News
            
            # ── 2. Market Data & Regulatory RSS Feeds ──
            for source_id, config in self.MARKET_RSS_FEEDS.items():
                source_signals = await self._fetch_rss(
                    session, config["rss_url"], source_id,
                    config["name"], config["quality_score"],
                    source_type="tso_market"
                )
                if source_signals:
                    logger.info(f"TSO Market ({source_id}): {len(source_signals)} items")
                else:
                    logger.warning(f"TSO Market ({source_id}): No items")
                
                for sig in source_signals:
                    if sig["url"] not in seen_urls:
                        seen_urls.add(sig["url"])
                        signals.append(sig)
        
        logger.info(f"TSO Publications: {len(signals)} signals")
        return signals

# =============================================================================
# REGULATORY SCRAPER (BNetzA, ACER)
# =============================================================================

class RegulatoryScraper(BaseScraper):
    """Scrape regulatory bodies for policy updates."""
    
    REGULATORY_SOURCES = {
        "acer": {
            "name": "ACER",
            "rss_url": "https://www.acer.europa.eu/rss.xml",
            "quality_score": 0.95
        },
        "eu_energy": {
            "name": "EU Energy Commission",
            "rss_url": "https://energy.ec.europa.eu/rss_en",
            "quality_score": 0.95
        }
    }
    
    def __init__(self):
        super().__init__("regulatory")
        self.source_type = "regulatory"
        self.quality_score = 0.95
    
    async def scrape(self) -> List[Dict[str, Any]]:
        """Scrape regulatory RSS feeds."""
        signals = []
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            for source_id, config in self.REGULATORY_SOURCES.items():
                try:
                    rss_url = config.get("rss_url")
                    if not rss_url:
                        continue
                    
                    async with session.get(rss_url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.text()
                            feed = feedparser.parse(content)
                            
                            for entry in feed.entries[:20]:
                                signal = self._create_signal(
                                    title=entry.get("title", ""),
                                    content=BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(),
                                    url=entry.get("link", ""),
                                    published_at=datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
                                )
                                signal["source_name"] = config["name"]
                                signal["source_quality_score"] = config["quality_score"]
                                signals.append(signal)
                                
                except Exception as e:
                    logger.error(f"Regulatory scrape error ({source_id}): {e}")
        
        logger.info(f"Regulatory: {len(signals)} signals")
        return signals

# =============================================================================
# ENLIT.WORLD SCRAPER
# =============================================================================

class EnlitWorldScraper(BaseScraper):
    """
    Scrape Enlit.world articles by topic.
    
    Note: Enlit.world blocks direct access (403), but we can:
    1. Use search to find article URLs
    2. Fetch individual articles
    3. Parse article content
    
    Topic hubs:
    - /library/grids/
    - /library/renewable-energy/
    - /library/flexibility/
    - /library/democratisation/
    etc.
    """
    
    # Topic categories from Enlit.world
    TOPIC_HUBS = {
        "grids": "https://www.enlit.world/library/grids",
        "smart_grids": "https://www.enlit.world/library/grids/smart-grids",
        "electricity": "https://www.enlit.world/library/grids/electricity",
        "grid_integration": "https://www.enlit.world/library/renewable-energy/grid-integration",
        "flexibility": "https://www.enlit.world/library/flexibility",
        "energy_storage": "https://www.enlit.world/library/energy-storage",
        "hydrogen": "https://www.enlit.world/library/hydrogen",
        "renewable_energy": "https://www.enlit.world/library/renewable-energy",
        "democratisation": "https://www.enlit.world/library/democratisation",
        "digitalisation": "https://www.enlit.world/library/digitalisation",
        "markets_policy": "https://www.enlit.world/library/markets-policy",
        "finance": "https://www.enlit.world/library/finance-investment"
    }
    
    # Keywords to search for on Enlit.world via Google
    SEARCH_QUERIES = [
        "site:enlit.world TSO flexibility",
        "site:enlit.world grid congestion",
        "site:enlit.world virtual power plant",
        "site:enlit.world HVDC transmission",
        "site:enlit.world energy storage grid",
        "site:enlit.world hydrogen electrolyzer",
        "site:enlit.world smart grid DSO TSO",
        "site:enlit.world renewable integration",
        "site:enlit.world demand response",
        "site:enlit.world V2G vehicle grid"
    ]
    
    def __init__(self):
        super().__init__("enlit_world")
        self.source_type = "industry_publication"
        self.quality_score = 0.90  # High-quality industry content
    
    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Scrape Enlit.world content.
        
        Strategy:
        1. Try direct topic hub URLs
        2. If blocked, use known article URLs from search
        3. Parse available content
        """
        signals = []
        
        async with aiohttp.ClientSession() as session:
            # Try to fetch known good article URLs
            known_articles = await self._get_known_articles()
            
            for article_url in known_articles:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                        "Accept": "text/html"
                    }
                    async with session.get(article_url, headers=headers, timeout=15) as response:
                        if response.status == 200:
                            content = await response.text()
                            signal = self._parse_article(content, article_url)
                            if signal:
                                signals.append(signal)
                        
                except Exception as e:
                    logger.debug(f"Enlit.world article fetch error: {e}")
                
                await asyncio.sleep(0.5)
        
        # Add curated Enlit topics as signals (from their published content)
        signals.extend(self._get_curated_enlit_signals())
        
        logger.info(f"Enlit.world: {len(signals)} signals")
        return signals
    
    async def _get_known_articles(self) -> List[str]:
        """Get known article URLs (could be expanded via search API)."""
        # These are real Enlit.world article URLs from search results
        return [
            "https://www.enlit.world/library/the-flexibility-iceberg-unlocking-the-hidden-value-of-our-grids",
            "https://www.enlit.world/library/turning-flexible-thinking-into-action-for-europes-energy-transformation",
            "https://www.enlit.world/library/flexible-and-dispatchable-the-grid-balancing-technology-shaping-our-renewable-future",
            "https://www.enlit.world/democratisation/monetising-flexibility-as-a-fix-for-grid-congestion/",
            "https://www.enlit.world/customer-services-management/whats-barring-consumer-side-flexibility-in-the-energy-system/",
            "https://www.enlit.world/library/enel-grids-flexibility-lab"
        ]
    
    def _parse_article(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Parse Enlit.world article HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract title
            title_tag = soup.find("h1") or soup.find("title")
            title = title_tag.get_text().strip() if title_tag else "Enlit Article"
            
            # Extract article content
            article = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
            if article:
                # Get text paragraphs
                paragraphs = article.find_all("p")
                content = " ".join([p.get_text().strip() for p in paragraphs[:5]])
            else:
                content = ""
            
            if len(content) < 100:
                return None
            
            return self._create_signal(
                title=title,
                content=content[:2000],
                url=url,
                published_at=None
            )
            
        except Exception as e:
            logger.debug(f"Enlit.world parse error: {e}")
            return None
    
    def _get_curated_enlit_signals(self) -> List[Dict[str, Any]]:
        """
        Return curated signals based on Enlit.world topics and content themes.
        These represent the key topics covered by Enlit.
        """
        curated_topics = [
            {
                "title": "Grid Flexibility: Unlocking Hidden Value",
                "content": "Energy optimisation through shifting consumption, generation and storage. "
                          "Capacity optimisation via orchestrating flexible loads for congestion relief. "
                          "With digital twins, AI and edge intelligence, grids can now be managed as "
                          "dynamic, adaptive systems. Key mechanisms: flexible power tariffs, local "
                          "flexibility markets, flexible connection agreements.",
                "category": "flexibility"
            },
            {
                "title": "Long Duration Energy Storage Critical for Grid Stability",
                "content": "LDES is a critical solution enabling EU objectives of strengthening "
                          "infrastructure security and resilience. Market reforms, investment signals "
                          "and procurement processes must materialise to unlock LDES potential. "
                          "Storage moving beyond lithium-ion as Europe seeks resilience.",
                "category": "energy_storage"
            },
            {
                "title": "TSO-DSO Coordination for Flexibility Markets",
                "content": "Flexibility markets help energy networks monitor flows and create signals "
                          "to change energy supply and demand. GOPACS platform in Netherlands enables "
                          "grid operators to manage congestion through market mechanisms. "
                          "Aggregated assets serve DSOs and TSOs simultaneously.",
                "category": "flexibility"
            },
            {
                "title": "Consumer-Side Flexibility Barriers and Solutions",
                "content": "Elia study 'The Power of Flex' highlights digitalisation barriers. "
                          "Rapid increases in electricity across mobility and heating sectors "
                          "add exponential demand. Flexible consumption should be integrated "
                          "into everyday practices with any owner of flexible assets able to participate.",
                "category": "flexibility"
            },
            {
                "title": "Grid-Forming Inverters for Renewable Future",
                "content": "Flexible dispatchable power generation can stabilise electricity markets "
                          "by reducing negative pricing. By 2030, renewable deployment globally set "
                          "to double while inflexible generation declines. Grid balancing gas engines "
                          "provide flexibility for variable renewable generation.",
                "category": "grid_stability"
            },
            {
                "title": "Digitalisation of Distribution Grid Operations",
                "content": "Smart grids combine digital intelligence with physical infrastructure "
                          "for more flexible, efficient, and reliable energy systems. Use cases in "
                          "demand response, flexibility markets, and grid automation. "
                          "Digital twins enable predictive operations.",
                "category": "digitalisation"
            },
            {
                "title": "Hydrogen Integration in European Energy System",
                "content": "Hydrogen economy developments including electrolyzers, storage, and "
                          "transport infrastructure. Green hydrogen production scaling up. "
                          "Sector coupling between electricity and gas grids through "
                          "power-to-gas applications.",
                "category": "hydrogen"
            },
            {
                "title": "E-Mobility Charging Infrastructure and Grid Impact",
                "content": "EV charging creating new load patterns requiring grid adaptation. "
                          "Vehicle-to-grid technology enabling EVs as distributed storage. "
                          "Smart charging and demand management essential for grid stability. "
                          "Fleet electrification accelerating.",
                "category": "e_mobility"
            }
        ]
        
        signals = []
        for topic in curated_topics:
            signal = self._create_signal(
                title=f"Enlit World: {topic['title']}",
                content=topic['content'],
                url=f"https://www.enlit.world/library/{topic['category']}"
            )
            signals.append(signal)
        
        return signals


# =============================================================================
# INDUSTRY EVENT SCRAPER (Enlit, E-world)
# =============================================================================

class IndustryEventScraper(BaseScraper):
    """Scrape industry event websites for trend signals."""
    
    EVENTS = {
        "enlit": {
            "name": "Enlit Europe",
            "base_url": "https://www.enlit-europe.com",
            "news_path": "/news/",
            "quality_score": 0.90
        },
        "eworld": {
            "name": "E-world Energy & Water",
            "base_url": "https://www.e-world-essen.com",
            "quality_score": 0.90
        },
        "windeurope": {
            "name": "WindEurope",
            "base_url": "https://windeurope.org",
            "news_path": "/newsroom/news/",
            "quality_score": 0.85
        }
    }
    
    # Enlit/Clarion newsletter topics (from provided link)
    ENLIT_TOPICS = [
        "Grid Modernization",
        "Energy Storage",
        "Hydrogen Economy",
        "Offshore Wind",
        "Digital Transformation",
        "Flexibility & Demand Response",
        "EV Infrastructure",
        "Cybersecurity",
        "TSO-DSO Coordination",
        "Market Design",
        "Sector Coupling",
        "AI in Energy"
    ]
    
    def __init__(self):
        super().__init__("industry_events")
        self.source_type = "conference"
        self.quality_score = 0.85
    
    async def scrape(self) -> List[Dict[str, Any]]:
        """Scrape industry event content."""
        signals = []
        
        # Generate topic-based signals for demo
        # In production, would scrape actual event websites
        for topic in self.ENLIT_TOPICS:
            signal = self._create_signal(
                title=f"Enlit Europe 2026: {topic} Track",
                content=f"Key discussions at Enlit Europe covering {topic.lower()} trends, "
                        f"innovations, and market developments in the European energy sector. "
                        f"Topics include grid integration, policy frameworks, and technology adoption.",
                url="https://www.enlit-europe.com/programme"
            )
            signal["source_name"] = "Enlit Europe"
            signals.append(signal)
        
        logger.info(f"Industry Events: {len(signals)} signals")
        return signals

# =============================================================================
# RESEARCH ORGANIZATION SCRAPER (IEA, IRENA, Fraunhofer)
# =============================================================================

class ResearchOrgScraper(BaseScraper):
    """Scrape research organization publications."""
    
    RESEARCH_ORGS = {
        "iea": {
            "name": "International Energy Agency",
            "rss_url": "https://www.iea.org/rss/news.xml",
            "quality_score": 0.95
        },
        "irena": {
            "name": "IRENA",
            "base_url": "https://www.irena.org",
            "quality_score": 0.95
        }
    }
    
    def __init__(self):
        super().__init__("research_org")
        self.source_type = "research_org"
        self.quality_score = 0.95
    
    async def scrape(self) -> List[Dict[str, Any]]:
        """Scrape research organization feeds."""
        signals = []
        
        async with aiohttp.ClientSession() as session:
            # IEA RSS
            try:
                iea_config = self.RESEARCH_ORGS["iea"]
                async with session.get(iea_config["rss_url"], timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:30]:
                            signal = self._create_signal(
                                title=entry.get("title", ""),
                                content=BeautifulSoup(entry.get("summary", ""), "html.parser").get_text(),
                                url=entry.get("link", ""),
                                published_at=datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
                            )
                            signal["source_name"] = iea_config["name"]
                            signal["source_quality_score"] = iea_config["quality_score"]
                            signals.append(signal)
                            
            except Exception as e:
                logger.error(f"IEA scrape error: {e}")
        
        logger.info(f"Research Orgs: {len(signals)} signals")
        return signals

# =============================================================================
# MASTER SCRAPER ORCHESTRATOR
# =============================================================================

class MasterScraper:
    """
    Orchestrates all scrapers for comprehensive data collection.
    """
    
    def __init__(self):
        self.scrapers = {
            "arxiv": ArxivScraper(),
            "rss_news": RSSNewsScraper(),
            "tso": TSOPublicationsScraper(),
            "regulatory": RegulatoryScraper(),
            "events": IndustryEventScraper(),
            "research_org": ResearchOrgScraper(),
            "enlit_world": EnlitWorldScraper()  # NEW: Enlit.world content
        }
    
    async def scrape_all(self, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Scrape all sources in parallel.
        
        Args:
            sources: List of source IDs to scrape (None = all)
        
        Returns:
            Combined list of all signals
        """
        sources_to_scrape = sources or list(self.scrapers.keys())
        
        logger.info(f"🚀 Starting scrape for: {sources_to_scrape}")
        
        # Create tasks
        tasks = []
        for source_id in sources_to_scrape:
            if source_id in self.scrapers:
                tasks.append(self.scrapers[source_id].scrape())
        
        # Run in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_signals = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scraper error: {result}")
            elif isinstance(result, list):
                all_signals.extend(result)
        
        # ── Cross-scraper dedup by URL ──
        # Prevents duplicates when the same feed appears in multiple scrapers
        # (e.g. SMARD was in both RSS_FEEDS and TSO MARKET_RSS_FEEDS)
        all_signals = self._dedup_by_url(all_signals)
        
        logger.info(f"✅ Total signals scraped: {len(all_signals)}")
        return all_signals
    
    @staticmethod
    def _dedup_by_url(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate signals by URL.
        Keeps the first occurrence (higher-priority scrapers run first).
        Reusable: works on any list of signal dicts with a 'url' key.
        """
        seen_urls = set()
        unique = []
        for sig in signals:
            url = sig.get("url", "")
            if not url or url not in seen_urls:
                seen_urls.add(url)
                unique.append(sig)
        removed = len(signals) - len(unique)
        if removed:
            logger.info(f"Dedup: removed {removed} duplicate signals by URL")
        return unique
    
    async def scrape_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Scrape a single source."""
        if source_id not in self.scrapers:
            logger.error(f"Unknown source: {source_id}")
            return []
        
        return await self.scrapers[source_id].scrape()

# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def run_full_scrape() -> List[Dict[str, Any]]:
    """Run a full scrape of all sources."""
    scraper = MasterScraper()
    return await scraper.scrape_all()

# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    async def test():
        scraper = MasterScraper()
        
        # Test individual scrapers
        print("\n" + "=" * 60)
        print("Testing RSS News Scraper")
        print("=" * 60)
        
        rss_scraper = RSSNewsScraper()
        signals = await rss_scraper.scrape(feeds=["utility_dive", "pv_magazine"])
        print(f"Got {len(signals)} signals")
        if signals:
            print(f"Sample: {signals[0]['title'][:60]}...")
        
        print("\n" + "=" * 60)
        print("Testing Industry Events Scraper")
        print("=" * 60)
        
        events_scraper = IndustryEventScraper()
        signals = await events_scraper.scrape()
        print(f"Got {len(signals)} signals")
        for s in signals[:3]:
            print(f"  - {s['title']}")
    
    asyncio.run(test())
