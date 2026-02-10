"""
SONAR.AI Demo Reporter
=======================
Visual logging for demo presentations.
Maps to Amprion's required workflow steps:
  1. Find Signals
  2. Cluster Signals  
  3. Derive Trends
  4. Assess Trends
  5. Prepare Results
  6. Validate Results

Usage:
    from services.demo_reporter import DemoReporter
    reporter = DemoReporter()
    reporter.start_pipeline("Agentic Trend Scan")
    reporter.step_find_signals(...)
    ...
    reporter.end_pipeline()
"""

import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ANSI Color Codes for Terminal Output
# ─────────────────────────────────────────────────────────────────────────────
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def c(text: str, *styles) -> str:
    """Apply color/style to text."""
    return "".join(styles) + str(text) + Colors.RESET


# ─────────────────────────────────────────────────────────────────────────────
# Box Drawing Characters
# ─────────────────────────────────────────────────────────────────────────────
BOX = {
    "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯",
    "h": "─", "v": "│",
    "ltee": "├", "rtee": "┤", "ttee": "┬", "btee": "┴",
    "cross": "┼",
    "h_double": "═", "v_double": "║",
    "tl_double": "╔", "tr_double": "╗", "bl_double": "╚", "br_double": "╝",
}


def box_line(width: int, left: str = "│", right: str = "│", fill: str = " ") -> str:
    return f"{left}{fill * (width - 2)}{right}"


def horizontal_line(width: int = 80, char: str = "─", color=Colors.BRIGHT_BLACK) -> str:
    return c(char * width, color)


# ─────────────────────────────────────────────────────────────────────────────
# Progress Bar
# ─────────────────────────────────────────────────────────────────────────────
def progress_bar(current: int, total: int, width: int = 40, 
                 fill_char: str = "█", empty_char: str = "░",
                 color=Colors.BRIGHT_CYAN) -> str:
    """Generate a progress bar string."""
    if total == 0:
        pct = 0
    else:
        pct = current / total
    filled = int(width * pct)
    bar = fill_char * filled + empty_char * (width - filled)
    return f"{c(bar, color)} {c(f'{pct*100:5.1f}%', Colors.BRIGHT_WHITE)}"


# ─────────────────────────────────────────────────────────────────────────────
# Demo Reporter Class
# ─────────────────────────────────────────────────────────────────────────────
class DemoReporter:
    """
    Visual reporter for SONAR.AI demo presentations.
    
    Provides clear, formatted output for each pipeline step,
    mapping to Amprion's required workflow:
    
    STEP 1: FIND SIGNALS      → Data collection from sources
    STEP 2: CLUSTER SIGNALS   → Semantic classification into categories
    STEP 3: DERIVE TRENDS     → Group signals into trend clusters
    STEP 4: ASSESS TRENDS     → MCDA scoring + SHAP explanation
    STEP 5: PREPARE RESULTS   → Narrative generation + dashboard prep
    STEP 6: VALIDATE RESULTS  → Quality checks + alerts + traceability
    """
    
    def __init__(self, output=None, use_colors: bool = True):
        self.output = output or sys.stdout
        self.use_colors = use_colors
        self.start_time = None
        self.step_times = {}
        self.metrics = {}
        
    def _print(self, text: str = ""):
        """Print to output stream."""
        if not self.use_colors:
            # Strip ANSI codes
            import re
            text = re.sub(r'\033\[[0-9;]*m', '', text)
        print(text, file=self.output)
        
    def _header_box(self, title: str, subtitle: str = "", width: int = 80,
                    bg_color=Colors.BG_BLUE, fg_color=Colors.BRIGHT_WHITE):
        """Print a header box."""
        self._print()
        self._print(c(BOX["tl_double"] + BOX["h_double"] * (width - 2) + BOX["tr_double"], Colors.BRIGHT_BLUE))
        
        # Title line
        padding = width - 4 - len(title)
        self._print(c(BOX["v_double"], Colors.BRIGHT_BLUE) + 
                   c(f"  {title}" + " " * padding, bg_color, fg_color, Colors.BOLD) +
                   c(BOX["v_double"], Colors.BRIGHT_BLUE))
        
        if subtitle:
            padding = width - 4 - len(subtitle)
            self._print(c(BOX["v_double"], Colors.BRIGHT_BLUE) + 
                       c(f"  {subtitle}" + " " * padding, bg_color, fg_color) +
                       c(BOX["v_double"], Colors.BRIGHT_BLUE))
        
        self._print(c(BOX["bl_double"] + BOX["h_double"] * (width - 2) + BOX["br_double"], Colors.BRIGHT_BLUE))
        self._print()

    def _step_header(self, step_num: int, step_name: str, description: str):
        """Print a step header."""
        icons = {
            1: "🔍", 2: "🧩", 3: "📊", 4: "⚖️", 5: "📋", 6: "✅"
        }
        icon = icons.get(step_num, "▶")
        
        self._print()
        self._print(horizontal_line(80))
        self._print(c(f"  {icon}  STEP {step_num}: {step_name}", Colors.BRIGHT_CYAN, Colors.BOLD))
        self._print(c(f"     {description}", Colors.BRIGHT_BLACK))
        self._print(horizontal_line(80))
        self._print()
        
        self.step_times[f"step_{step_num}"] = time.time()

    def _step_complete(self, step_num: int, summary: str):
        """Mark step as complete with timing."""
        start = self.step_times.get(f"step_{step_num}", time.time())
        elapsed = time.time() - start
        
        self._print()
        self._print(c(f"  ✓ Step {step_num} complete", Colors.BRIGHT_GREEN, Colors.BOLD) + 
                   c(f" ({elapsed:.2f}s)", Colors.BRIGHT_BLACK))
        self._print(c(f"    {summary}", Colors.WHITE))
        self._print()

    def _metric_line(self, label: str, value: Any, indent: int = 4):
        """Print a metric line."""
        spaces = " " * indent
        self._print(f"{spaces}{c('•', Colors.BRIGHT_BLACK)} {c(label + ':', Colors.WHITE)} {c(str(value), Colors.BRIGHT_YELLOW)}")

    def _table(self, headers: List[str], rows: List[List[Any]], indent: int = 4):
        """Print a simple table."""
        if not rows:
            return
            
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        spaces = " " * indent
        
        # Header
        header_line = spaces
        for i, h in enumerate(headers):
            header_line += c(h.ljust(widths[i] + 2), Colors.BRIGHT_WHITE, Colors.BOLD)
        self._print(header_line)
        
        # Separator
        sep_line = spaces + c("─" * (sum(widths) + len(widths) * 2), Colors.BRIGHT_BLACK)
        self._print(sep_line)
        
        # Rows
        for row in rows:
            row_line = spaces
            for i, cell in enumerate(row):
                row_line += c(str(cell).ljust(widths[i] + 2), Colors.WHITE)
            self._print(row_line)

    # ═══════════════════════════════════════════════════════════════════════════
    # PIPELINE LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start_pipeline(self, name: str = "SONAR.AI Agentic Scan", goal: str = ""):
        """Start the pipeline with a header."""
        self.start_time = time.time()
        self.metrics = {}
        
        self._header_box(
            f"🚀 {name.upper()}",
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Goal: {goal[:50]}..." if goal else "",
            bg_color=Colors.BG_BLUE
        )
        
        self._print(c("  Pipeline Steps:", Colors.BRIGHT_WHITE, Colors.BOLD))
        steps = [
            ("1", "FIND SIGNALS",           "Collect raw data points from RSS, arXiv, TSO, regulatory sources"),
            ("2", "CLASSIFY SIGNALS",       "Assign each signal to one of 16 TSO taxonomy categories"),
            ("3", "IDENTIFY TRENDS",        "Cluster signals within each category to find distinct patterns"),
            ("4", "ASSESS TRENDS",          "Score each trend using Bayesian MCDA + Amprion weights"),
            ("5", "PREPARE RESULTS",        "Generate actionable briefs with IT/Digital impact"),
            ("6", "VALIDATE RESULTS",       "Quality checks, blind spots, traceability verification"),
        ]
        for num, name, desc in steps:
            self._print(c(f"    [{num}] {name:18}", Colors.CYAN) + c(f" {desc}", Colors.BRIGHT_BLACK))
        self._print()

    def end_pipeline(self, success: bool = True):
        """End the pipeline with summary."""
        elapsed = time.time() - (self.start_time or time.time())
        
        if success:
            self._header_box(
                "✅ PIPELINE COMPLETE",
                f"Total time: {elapsed:.1f}s | Signals: {self.metrics.get('total_signals', '?')} | Trends: {self.metrics.get('total_trends', '?')}",
                bg_color=Colors.BG_GREEN
            )
        else:
            self._header_box(
                "❌ PIPELINE FAILED",
                f"Elapsed: {elapsed:.1f}s",
                bg_color=Colors.BG_RED
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def show_configuration(self, 
                           classifier_backend: str,
                           llm_provider: str = None,
                           ahp_profile: str = "default",
                           narrative_mode: str = "template",
                           scraper_count: int = 7,
                           selfhosted_url: str = None):
        """
        Display the current configuration before starting the pipeline.
        Shows classifier backend, LLM provider (if any), and key settings.
        
        Addresses DoD requirements:
        - Traceability & transparency: Clear visibility into system configuration
        - GDPR compliance: Shows if using self-hosted LLM (no external API)
        - Flexible search: Shows configurable parameters
        """
        self._print()
        self._print(c("  ⚙️  CONFIGURATION", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print(horizontal_line(60, "─", Colors.BRIGHT_BLACK))
        
        # Classifier backend
        if "selfhosted" in classifier_backend.lower() or "ollama" in classifier_backend.lower():
            backend_color = Colors.BRIGHT_GREEN
            backend_status = "✓ SELF-HOSTED (GDPR compliant, no external API)"
        elif "local" in classifier_backend.lower() or "SentenceTransformer" in classifier_backend:
            backend_color = Colors.BRIGHT_CYAN
            backend_status = "✓ LOCAL (SentenceTransformer, no API calls)"
        else:
            backend_color = Colors.BRIGHT_YELLOW
            backend_status = f"⚠ CLOUD API ({classifier_backend})"
        
        self._print(f"    {c('Classifier:', Colors.WHITE)} {c(classifier_backend, backend_color)} {c(backend_status, Colors.BRIGHT_BLACK)}")
        
        # LLM Provider details if self-hosted
        if llm_provider:
            self._print(f"    {c('LLM Provider:', Colors.WHITE)} {c(llm_provider, Colors.BRIGHT_MAGENTA)}")
        if selfhosted_url:
            self._print(f"    {c('LLM Endpoint:', Colors.WHITE)} {c(selfhosted_url, Colors.BRIGHT_BLACK)}")
        
        # Other settings
        self._print(f"    {c('AHP Profile:', Colors.WHITE)} {c(ahp_profile, Colors.BRIGHT_CYAN)}")
        self._print(f"    {c('Narrative Mode:', Colors.WHITE)} {c(narrative_mode, Colors.BRIGHT_CYAN)}")
        self._print(f"    {c('Data Scrapers:', Colors.WHITE)} {c(str(scraper_count), Colors.BRIGHT_CYAN)} active sources")
        self._print(horizontal_line(60, "─", Colors.BRIGHT_BLACK))
        self._print()

    def show_llm_backends(self, backends: List[Dict[str, Any]]):
        """
        Display available LLM backends and their status.
        
        Addresses DoD requirements:
        - Flexible search: Shows configurable LLM options
        - GDPR compliance: Highlights self-hosted options
        """
        self._print()
        self._print(c("  🤖 AVAILABLE LLM BACKENDS:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        for backend in backends:
            name = backend.get("name", "Unknown")
            status = backend.get("status", "unknown")
            url = backend.get("url", "")
            is_active = backend.get("active", False)
            
            if status == "available":
                status_icon = c("✓", Colors.BRIGHT_GREEN)
                status_text = c("available", Colors.BRIGHT_GREEN)
            elif status == "active":
                status_icon = c("★", Colors.BRIGHT_YELLOW)
                status_text = c("ACTIVE", Colors.BRIGHT_YELLOW, Colors.BOLD)
            else:
                status_icon = c("○", Colors.BRIGHT_BLACK)
                status_text = c("not available", Colors.BRIGHT_BLACK)
            
            gdpr = c("[GDPR ✓]", Colors.BRIGHT_GREEN) if backend.get("gdpr_compliant", False) else ""
            
            self._print(f"    {status_icon} {name:20} {status_text:15} {url[:30]:30} {gdpr}")
        
        self._print()

    def show_amprion_strategic_priors(self):
        """
        Display Amprion's pre-defined strategic weights per category.
        
        This is STEP 0 - CONFIGURATION that shows:
        - What each TSO category is
        - Amprion's strategic weight (0-100) for each
        - The tier classification (existential/critical/high/medium/low)
        
        This sets context so that later when we show "[critical]" or "weight 90",
        the viewer understands what it means.
        """
        self._print()
        self._print(c("  📋 AMPRION STRATEGIC PRIORS (Category Weights)", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        self._print(c("  These weights are PRE-DEFINED by Amprion based on TSO strategic priorities.", Colors.BRIGHT_BLACK))
        self._print(c("  They determine how much each category matters REGARDLESS of current signal volume.", Colors.BRIGHT_BLACK))
        self._print()
        
        # Import the priors
        from services.trend_scorer import AMPRION_STRATEGIC_PRIORS
        
        # Group by tier for better display
        tiers = {
            "existential": [],
            "critical": [],
            "high": [],
            "medium-high": [],
            "medium": [],
            "low": [],
        }
        
        for category, data in AMPRION_STRATEGIC_PRIORS.items():
            tier = data.get("tier", "low")
            if tier in tiers:
                tiers[tier].append((category, data))
        
        tier_colors = {
            "existential": Colors.BRIGHT_RED,
            "critical": Colors.BRIGHT_MAGENTA,
            "high": Colors.BRIGHT_YELLOW,
            "medium-high": Colors.BRIGHT_CYAN,
            "medium": Colors.BRIGHT_WHITE,
            "low": Colors.BRIGHT_BLACK,
        }
        
        tier_descriptions = {
            "existential": "Failure is unacceptable — non-negotiable TSO functions",
            "critical": "Core TSO mission — actively planned and resourced",
            "high": "Strategically important — actively tracked",
            "medium-high": "Important but less immediate",
            "medium": "Watch and track",
            "low": "Informational only",
        }
        
        self._print(c("  ┌────────────────────────────────────┬────────┬────────────────────────────────┐", Colors.BRIGHT_BLACK))
        self._print(c("  │ Category/Taxonomy                  │ Weight │ Tier & Rationale               │", Colors.BRIGHT_BLACK))
        self._print(c("  ├────────────────────────────────────┼────────┼────────────────────────────────┤", Colors.BRIGHT_BLACK))
        
        for tier_name in ["existential", "critical", "high", "medium-high", "medium"]:
            if tiers[tier_name]:
                tier_color = tier_colors[tier_name]
                self._print(c(f"  │ {tier_name.upper():36} │        │ {tier_descriptions[tier_name][:30]:30} │", tier_color, Colors.BOLD))
                
                for category, data in sorted(tiers[tier_name], key=lambda x: -x[1]["weight"]):
                    weight = data["weight"]
                    # Category name formatting
                    cat_display = category.replace("_", " ").title()[:34]
                    
                    # Weight bar (scale 0-100, show as mini bar)
                    bar_width = int(weight / 10)
                    bar = "█" * bar_width + "░" * (10 - bar_width)
                    
                    self._print(f"  │   {cat_display:34} │ {c(f'{weight:3}', Colors.BRIGHT_YELLOW)}/100 │ {bar} │")
        
        self._print(c("  └────────────────────────────────────┴────────┴────────────────────────────────┘", Colors.BRIGHT_BLACK))
        self._print()
        self._print(c("  📐 MCDA SCORING FORMULA:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        self._print(c("  Priority Score = (Strategic × 0.43 + Evidence × 0.23 + Growth × 0.15 + Maturity × 0.19) / 10", Colors.WHITE))
        self._print()
        self._print(c("  Where:", Colors.BRIGHT_BLACK))
        self._print(c("    • Strategic = Amprion's pre-defined weight (see table above, 0-100)", Colors.BRIGHT_BLACK))
        self._print(c("    • Evidence  = Signal count + source diversity + quality (computed, 0-100)", Colors.BRIGHT_BLACK))
        self._print(c("    • Growth    = Signal volume change vs previous period (computed, 0-100)", Colors.BRIGHT_BLACK))
        self._print(c("    • Maturity  = Technology readiness level from content (computed, 0-100)", Colors.BRIGHT_BLACK))
        self._print()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: FIND SIGNALS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def step_find_signals(self, 
                          scraper_results: Dict[str, int],
                          total_raw: int,
                          total_after_dedup: int,
                          source_breakdown: Dict[str, int]):
        """
        Report Step 1: Find Signals.
        
        Maps to Amprion requirement:
        "Automatically capture relevant information from publicly available sources."
        """
        self._step_header(1, "FIND SIGNALS", 
                         "Automatically capture relevant information from publicly available sources")
        
        self._print(c("  📡 Data Sources Scraped:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        # Scraper results table
        rows = []
        for scraper, count in sorted(scraper_results.items(), key=lambda x: -x[1]):
            status = c("✓", Colors.BRIGHT_GREEN) if count > 0 else c("○", Colors.BRIGHT_BLACK)
            rows.append([status, scraper, count])
        self._table(["", "Scraper", "Signals"], rows)
        
        self._print()
        self._print(c("  📊 Source Type Breakdown:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        # Source breakdown with visual bars
        max_count = max(source_breakdown.values()) if source_breakdown else 1
        for src_type, count in sorted(source_breakdown.items(), key=lambda x: -x[1]):
            bar_width = int((count / max_count) * 30)
            bar = c("█" * bar_width, Colors.BRIGHT_CYAN) + c("░" * (30 - bar_width), Colors.BRIGHT_BLACK)
            self._print(f"    {src_type:20} {bar} {c(str(count), Colors.BRIGHT_YELLOW)}")
        
        self._print()
        self._metric_line("Raw signals collected", total_raw)
        self._metric_line("After URL deduplication", total_after_dedup)
        self._metric_line("Duplicates removed", total_raw - total_after_dedup)
        
        self.metrics["total_signals"] = total_after_dedup
        self.metrics["raw_signals"] = total_raw
        
        self._step_complete(1, f"Collected {total_after_dedup} unique signals from {len(scraper_results)} scrapers")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: CLUSTER SIGNALS (Classification)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def step_cluster_signals(self,
                             total_classified: int,
                             category_counts: Dict[str, int],
                             off_topic_count: int,
                             classifier_backend: str,
                             avg_confidence: float):
        """
        Report Step 2: Classify Signals into Taxonomy Categories.
        
        Maps to Amprion requirement:
        "Group similar signals to identify patterns and connections."
        
        TERMINOLOGY:
        - Category = A taxonomy bucket (e.g., 'cybersecurity_ot')
        - This step assigns each signal to one of 16 TSO-relevant categories
        """
        self._step_header(2, "CLASSIFY SIGNALS", 
                         "Assign each signal to one of 16 TSO taxonomy categories")
        
        self._print(c(f"  🧠 Classifier: {classifier_backend}", Colors.BRIGHT_MAGENTA))
        self._print()
        
        self._print(c("  📂 Category Distribution:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        # Category distribution with visual bars
        total = sum(category_counts.values())
        max_count = max(category_counts.values()) if category_counts else 1
        
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:16]:
            pct = (count / total) * 100 if total > 0 else 0
            bar_width = int((count / max_count) * 25)
            bar = c("█" * bar_width, Colors.BRIGHT_GREEN) + c("░" * (25 - bar_width), Colors.BRIGHT_BLACK)
            cat_display = cat[:25].ljust(25)
            self._print(f"    {cat_display} {bar} {c(f'{count:4}', Colors.BRIGHT_YELLOW)} ({pct:4.1f}%)")
        
        self._print()
        self._metric_line("Signals classified", total_classified)
        self._metric_line("Off-topic filtered", off_topic_count)
        self._metric_line("Average confidence", f"{avg_confidence:.2%}")
        self._metric_line("Categories with signals", len([c for c, n in category_counts.items() if n > 0]))
        
        self.metrics["classified_signals"] = total_classified
        self.metrics["off_topic"] = off_topic_count
        
        self._step_complete(2, f"Classified {total_classified} signals into {len(category_counts)} categories")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: IDENTIFY TRENDS WITHIN CATEGORIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def step_derive_trends(self,
                           trends_created: int,
                           trend_details: List[Dict[str, Any]],
                           min_signals_threshold: int = 3,
                           categories_with_trends: int = None):
        """
        Report Step 3: Identify distinct trends within categories.
        
        NEW: Uses clustering to find MULTIPLE trends per category.
        
        Example output:
            Category: cybersecurity_ot
            └─ Trend 1: "Ransomware Energy Sector" (62 signals)
            └─ Trend 2: "OT Supply Chain" (48 signals)
        """
        self._step_header(3, "IDENTIFY TRENDS WITHIN CATEGORIES", 
                         f"Cluster signals to find distinct patterns (min {min_signals_threshold} signals)")
        
        # Explain what we're doing
        self._print(c("  📊 TREND IDENTIFICATION:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print(c("      Each category can contain MULTIPLE distinct trends.", Colors.BRIGHT_BLACK))
        self._print(c("      Trends are identified by clustering similar signals within a category.", Colors.BRIGHT_BLACK))
        self._print()
        
        # Group trends by category for display
        trends_by_cat: Dict[str, List[Dict]] = {}
        for t in trend_details:
            cat = t.get("category", "other")
            if cat not in trends_by_cat:
                trends_by_cat[cat] = []
            trends_by_cat[cat].append(t)
        
        # Display category → trends hierarchy
        self._print(c("  🔍 TRENDS IDENTIFIED:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        for cat, cat_trends in sorted(trends_by_cat.items(), 
                                       key=lambda x: -sum(t.get("signal_count", 0) for t in x[1])):
            # Category header
            total_signals = sum(t.get("signal_count", 0) for t in cat_trends)
            cat_display = cat.replace("_", " ").title()
            self._print(f"    {c(cat_display, Colors.BRIGHT_CYAN, Colors.BOLD)} ({total_signals} signals, {len(cat_trends)} trends)")
            
            # List trends under category
            for i, trend in enumerate(sorted(cat_trends, key=lambda x: -x.get("signal_count", 0)), 1):
                trend_name = trend.get("name", "Unknown")[:40]
                signal_count = trend.get("signal_count", 0)
                keywords = trend.get("keywords", [])[:3]
                
                # Tree structure
                prefix = "└─" if i == len(cat_trends) else "├─"
                keywords_str = f" [{', '.join(keywords)}]" if keywords else ""
                
                self._print(f"      {prefix} Trend {i}: {c(trend_name, Colors.WHITE)} ({signal_count} signals){c(keywords_str, Colors.BRIGHT_BLACK)}")
            
            self._print()
        
        # Summary statistics
        total_signals_in_trends = sum(t.get("signal_count", 0) for t in trend_details)
        self._print(c("  📈 Summary:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._metric_line("Total categories", f"{categories_with_trends or len(trends_by_cat)} (from 16 taxonomy)")
        self._metric_line("Total trends identified", f"{trends_created} (across all categories)")
        self._metric_line("Avg trends per category", f"{trends_created / max(len(trends_by_cat), 1):.1f}")
        self._metric_line("Total signals in trends", total_signals_in_trends)
        
        self.metrics["total_trends"] = trends_created
        self.metrics["categories_with_trends"] = categories_with_trends or len(trends_by_cat)
        
        self._step_complete(3, f"Identified {trends_created} distinct trends across {len(trends_by_cat)} categories")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: ASSESS TRENDS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def step_assess_trends(self,
                           scored_trends: List[Dict[str, Any]],
                           ahp_profile: str,
                           mcda_weights: Dict[str, float]):
        """
        Report Step 4: Assess Trends for Amprion Relevance.
        
        Shows FULLY TRANSPARENT scoring:
        - Where each input score comes from
        - The complete calculation with Amprion project linkage bonus
        """
        self._step_header(4, "ASSESS TRENDS", 
                         "Score each trend for Amprion TSO relevance using Bayesian MCDA")
        
        # ──────────────────────────────────────────────────────────────────────
        # SCORING FORMULA WITH PROJECT LINKAGE EXPLANATION
        # ──────────────────────────────────────────────────────────────────────
        self._print(c("  📊 SCORING FORMULA:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        self._print(c("  Base Score = (Strategic×0.43 + Evidence×0.23 + Growth×0.15 + Maturity×0.19) ÷ 10", Colors.WHITE))
        self._print(c("  Final Score = Base Score + Amprion Project Linkage Bonus", Colors.WHITE))
        self._print()
        self._print(c("  📎 AMPRION PROJECT LINKAGE BONUS:", Colors.BRIGHT_CYAN, Colors.BOLD))
        self._print(c("      If trend signals mention Amprion mega-projects, score gets +0.1 to +0.5 bonus:", Colors.BRIGHT_BLACK))
        self._print(c("      • SuedLink, A-Nord, Korridor B (grid corridors) → +0.3 to +0.5", Colors.BRIGHT_BLACK))
        self._print(c("      • BalWin1/2, NeuLink (offshore connections) → +0.3 to +0.5", Colors.BRIGHT_BLACK))
        self._print(c("      • Grid Boosters, STATCOM (stability projects) → +0.2 to +0.4", Colors.BRIGHT_BLACK))
        self._print(c("      • hybridge, GET H2 (hydrogen projects) → +0.2 to +0.3", Colors.BRIGHT_BLACK))
        self._print()
        
        # ──────────────────────────────────────────────────────────────────────
        # WHERE EACH INPUT COMES FROM
        # ──────────────────────────────────────────────────────────────────────
        self._print(c("  📐 INPUT SCORE SOURCES (all 0-100):", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print(c("      Strategic = From Step 0 table (Amprion's pre-defined weight per category)", Colors.BRIGHT_BLACK))
        self._print(c("      Evidence  = Volume(40%) + SourceDiversity(25%) + Quality(20%) + Recency(15%)", Colors.BRIGHT_BLACK))
        self._print(c("      Growth    = Signal count change: (recent_14d - previous_14d) / previous_14d", Colors.BRIGHT_BLACK))
        self._print(c("      Maturity  = TRL from content keywords + source types (research→low, news→high)", Colors.BRIGHT_BLACK))
        self._print()
        
        self._print(c(f"  ⚖️ MCDA Weights (AHP Profile: {ahp_profile}):", Colors.BRIGHT_MAGENTA))
        for factor, weight in mcda_weights.items():
            bar = c("█" * int(weight * 30), Colors.BRIGHT_CYAN) + c("░" * (30 - int(weight * 30)), Colors.BRIGHT_BLACK)
            self._print(f"      {factor:22} {bar} {c(f'{weight:.2f}', Colors.BRIGHT_YELLOW)} ({int(weight*100)}%)")
        self._print()
        
        # ──────────────────────────────────────────────────────────────────────
        # TREND RANKINGS WITH COMPLETE SCORE BREAKDOWN
        # ──────────────────────────────────────────────────────────────────────
        self._print(c("  🏆 TREND RANKINGS (Category → Trend):", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        sorted_trends = sorted(scored_trends, key=lambda x: -x.get("priority_score", 0))
        
        for i, trend in enumerate(sorted_trends[:10], 1):
            priority_score = trend.get("priority_score", 0)
            name = trend.get("name", "Unknown")
            tier = trend.get("amprion_tier", "unknown")
            category = trend.get("category", "")
            signal_count = trend.get("signal_count", 0)
            scores_detail = trend.get("scores", {})
            
            # Tier color
            tier_colors = {
                "existential": Colors.BRIGHT_RED,
                "critical": Colors.BRIGHT_MAGENTA,
                "high": Colors.BRIGHT_YELLOW,
                "medium-high": Colors.BRIGHT_CYAN,
                "medium": Colors.BRIGHT_WHITE,
                "low": Colors.BRIGHT_BLACK,
            }
            tier_color = tier_colors.get(tier, Colors.WHITE)
            
            # Score bar (1-10 scale)
            bar_filled = int(priority_score)
            bar = c("█" * bar_filled, Colors.BRIGHT_GREEN) + c("░" * (10 - bar_filled), Colors.BRIGHT_BLACK)
            
            # Format category name nicely
            cat_display = category.replace("_", " ").title()[:20]
            trend_display = name[:30]
            
            # Print trend header: Category → Trend
            rank_str = c(f"#{i}", Colors.BRIGHT_YELLOW, Colors.BOLD)
            self._print(f"    {rank_str} {bar} {c(f'{priority_score:.1f}/10', Colors.BRIGHT_WHITE, Colors.BOLD)} {c(cat_display, Colors.BRIGHT_MAGENTA)} → {c(trend_display, Colors.BRIGHT_CYAN)} {c(f'[{tier}]', tier_color)}")
            
            # Extract component scores (0-100 scale)
            strategic_100 = scores_detail.get("strategic_relevance_score", 0) or 0
            evidence_100 = scores_detail.get("evidence_strength_score", scores_detail.get("volume_score", 0)) or 0
            growth_100 = scores_detail.get("growth_score", 0) or 0
            maturity_100 = scores_detail.get("maturity_readiness_score", scores_detail.get("quality_score", 0) * 10) or 0
            project_bonus = scores_detail.get("project_bonus", 0) or 0
            
            # Evidence sub-components (if available)
            evidence_detail = scores_detail.get("evidence_detail", {})
            ev_volume = evidence_detail.get("volume", 0)
            ev_diversity = evidence_detail.get("diversity", 0)
            ev_quality = evidence_detail.get("quality", 0)
            ev_recency = evidence_detail.get("recency", 0)
            
            # Get weights
            w_s = mcda_weights.get("strategic_importance", 0.43)
            w_e = mcda_weights.get("evidence_strength", 0.23)
            w_g = mcda_weights.get("growth_momentum", 0.15)
            w_m = mcda_weights.get("maturity_readiness", 0.19)
            
            # Calculate weighted contributions (still in 0-100 scale)
            contrib_s = strategic_100 * w_s
            contrib_e = evidence_100 * w_e
            contrib_g = growth_100 * w_g
            contrib_m = maturity_100 * w_m
            
            # Sum and divide by 10 to get base score (0-10 scale)
            weighted_sum = contrib_s + contrib_e + contrib_g + contrib_m
            base_score = weighted_sum / 10
            final_score = base_score + project_bonus
            
            # Show WHERE each input comes from with detailed breakdown
            self._print(c(f"       Input Scores (0-100 scale) — WHERE THEY COME FROM:", Colors.BRIGHT_BLACK))
            
            # Strategic - from Step 0
            self._print(f"         Strategic: {c(f'{strategic_100:5.1f}', Colors.WHITE)} ← AMPRION_PRIORS['{category}']['weight'] (see Step 0 table)")
            
            # Evidence - computed from sub-components
            self._print(f"         Evidence:  {c(f'{evidence_100:5.1f}', Colors.WHITE)} ← Computed from {signal_count} signals:")
            if ev_volume > 0 or ev_diversity > 0:
                self._print(f"                      Volume({ev_volume:.0f})×0.4 + Diversity({ev_diversity:.0f})×0.25 + Quality({ev_quality:.0f})×0.2 + Recency({ev_recency:.0f})×0.15")
            else:
                self._print(f"                      (Volume×0.4 + SourceDiversity×0.25 + Quality×0.2 + Recency×0.15)")
            
            # Growth - computed from temporal analysis
            self._print(f"         Growth:    {c(f'{growth_100:5.1f}', Colors.WHITE)} ← Signal velocity: (recent_14d - previous_14d) / max(previous_14d, 1)")
            
            # Maturity - computed from TRL analysis
            self._print(f"         Maturity:  {c(f'{maturity_100:5.1f}', Colors.WHITE)} ← TRL estimate: baseline + content_keywords + source_types")
            
            # Show calculation
            self._print(c(f"       Calculation:", Colors.BRIGHT_BLACK))
            self._print(f"         {strategic_100:5.1f} × {w_s:.2f} = {c(f'{contrib_s:5.1f}', Colors.BRIGHT_GREEN)} (strategic)")
            self._print(f"         {evidence_100:5.1f} × {w_e:.2f} = {c(f'{contrib_e:5.1f}', Colors.BRIGHT_GREEN)} (evidence)")
            self._print(f"         {growth_100:5.1f} × {w_g:.2f} = {c(f'{contrib_g:5.1f}', Colors.BRIGHT_GREEN)} (growth)")
            self._print(f"         {maturity_100:5.1f} × {w_m:.2f} = {c(f'{contrib_m:5.1f}', Colors.BRIGHT_GREEN)} (maturity)")
            self._print(c(f"         ─────────────────────────────────", Colors.BRIGHT_BLACK))
            self._print(f"         Weighted Sum = {c(f'{weighted_sum:.1f}', Colors.BRIGHT_YELLOW)}")
            self._print(f"         Base Score = {weighted_sum:.1f} ÷ 10 = {c(f'{base_score:.2f}', Colors.BRIGHT_YELLOW)}")
            
            # Show project bonus if any
            if project_bonus > 0:
                project_type = scores_detail.get("project_type", "")
                linked_projects = scores_detail.get("linked_projects", [])
                if linked_projects:
                    projects_str = ", ".join(linked_projects[:3])
                elif project_type:
                    projects_str = f"Amprion {project_type}"
                else:
                    projects_str = "Amprion mega-project keywords detected"
                self._print(f"         Amprion Project Linkage = {c(f'+{project_bonus:.2f}', Colors.BRIGHT_MAGENTA)} ← {projects_str}")
                self._print(f"         {c('FINAL SCORE', Colors.BRIGHT_WHITE)} = {base_score:.2f} + {project_bonus:.2f} = {c(f'{final_score:.1f}/10', Colors.BRIGHT_GREEN, Colors.BOLD)}")
            else:
                self._print(f"         {c('FINAL SCORE', Colors.BRIGHT_WHITE)} = {c(f'{base_score:.1f}/10', Colors.BRIGHT_GREEN, Colors.BOLD)} (no Amprion project linkage)")
            
            # Check if displayed score matches calculated
            if abs(priority_score - final_score) > 0.2:
                self._print(c(f"         ℹ️  Displayed {priority_score:.1f} vs calculated {final_score:.1f} — difference from rounding/capping", Colors.BRIGHT_BLACK))
            
            # Show why tier might differ from rank
            if tier in ("existential", "critical", "high") and i > 5:
                self._print(c(f"         ⚠️ [{tier}] tier but rank #{i}: evidence ({evidence_100:.0f}) or growth ({growth_100:.0f}) is low", Colors.BRIGHT_YELLOW))
            
            self._print()
        
        # ──────────────────────────────────────────────────────────────────────
        # SUMMARY STATISTICS
        # ──────────────────────────────────────────────────────────────────────
        all_scores = [t.get("priority_score", 0) for t in scored_trends]
        if all_scores:
            self._print(c("  📈 Summary:", Colors.BRIGHT_WHITE, Colors.BOLD))
            self._metric_line("Score range", f"{min(all_scores):.1f} – {max(all_scores):.1f}")
            self._metric_line("Average score", f"{sum(all_scores)/len(all_scores):.2f}")
        
        high_priority = len([s for s in all_scores if s >= 8.0])
        self._metric_line("High priority (≥8.0)", high_priority)
        
        self._step_complete(4, f"Scored {len(scored_trends)} trends, {high_priority} high-priority")
    
    def _mini_bar(self, value: float, max_val: float, width: int = 10) -> str:
        """Create a mini progress bar for score breakdown."""
        filled = int((value / max_val) * width) if max_val > 0 else 0
        filled = min(filled, width)
        bar = c("▓" * filled, Colors.BRIGHT_CYAN) + c("░" * (width - filled), Colors.BRIGHT_BLACK)
        return bar

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: PREPARE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def step_prepare_results(self,
                             narratives_generated: int,
                             briefs_generated: int,
                             deep_dives_generated: int,
                             key_players_extracted: int,
                             narrative_mode: str):
        """
        Report Step 5: Prepare Actionable Results.
        
        Maps to Amprion requirement:
        "Present trends and analyses clearly and understandably so decision-makers 
        can work with them directly."
        
        Generates:
        - Executive briefs with Amprion context
        - IT/Digital impact assessments
        - Concrete action recommendations
        """
        self._step_header(5, "PREPARE RESULTS", 
                         "Generate actionable briefs with Amprion context and IT/Digital impact")
        
        self._print(c(f"  ✍️ Narrative Mode: {narrative_mode}", Colors.BRIGHT_MAGENTA))
        self._print()
        
        self._print(c("  📝 Generated Content:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        items = [
            ("Executive Briefs (250 char)", briefs_generated, "Quick decision context"),
            ("Deep Dive Analyses (2000 char)", deep_dives_generated, "5-section detailed analysis"),
            ("'So What' Summaries", narratives_generated, "Action vs inaction framing"),
            ("Key Players Lists", key_players_extracted, "Organizations & entities"),
        ]
        
        for name, count, desc in items:
            status = c("✓", Colors.BRIGHT_GREEN) if count > 0 else c("○", Colors.BRIGHT_BLACK)
            self._print(f"    {status} {name:35} {c(str(count), Colors.BRIGHT_YELLOW):>5}  {c(desc, Colors.BRIGHT_BLACK)}")
        
        self._print()
        self._metric_line("Total narratives", narratives_generated)
        self._metric_line("Entities extracted", key_players_extracted)
        
        self._step_complete(5, f"Generated {narratives_generated} trend narratives")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6: VALIDATE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def step_validate_results(self,
                              coverage_pct: float,
                              categories_covered: int,
                              total_categories: int,
                              blind_spots: List[str],
                              weak_coverage: List[str],
                              alerts_generated: int,
                              quality_assessment: str,
                              traceability_check: bool):
        """
        Report Step 6: Validate Results.
        
        Maps to Amprion requirement:
        "Continuously compare new signals and derived trends to ensure quality, 
        timeliness, de-duplication, correct assignment, and relevance for Amprion."
        """
        self._step_header(6, "VALIDATE RESULTS", 
                         "Quality checks, alerts, blind spot detection, traceability")
        
        self._print(c("  🔎 Coverage Analysis:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        # Coverage bar
        coverage_bar = progress_bar(categories_covered, total_categories, width=30)
        self._print(f"    Category Coverage: {coverage_bar} ({categories_covered}/{total_categories})")
        self._print()
        
        # Quality assessment
        quality_colors = {
            "HIGH": Colors.BRIGHT_GREEN,
            "MEDIUM": Colors.BRIGHT_YELLOW,
            "LOW": Colors.BRIGHT_RED,
        }
        q_color = quality_colors.get(quality_assessment, Colors.WHITE)
        self._print(f"    Scan Quality: {c(quality_assessment, q_color, Colors.BOLD)}")
        self._print()
        
        # Blind spots
        if blind_spots:
            self._print(c("  ⚠️ BLIND SPOTS (high-tier categories with insufficient data):", Colors.BRIGHT_RED, Colors.BOLD))
            for bs in blind_spots[:5]:
                self._print(f"      {c('!', Colors.BRIGHT_RED)} {bs}")
            self._print()
        
        # Weak coverage
        if weak_coverage:
            self._print(c("  ⚡ WEAK COVERAGE:", Colors.BRIGHT_YELLOW))
            for wc in weak_coverage[:5]:
                self._print(f"      {c('△', Colors.BRIGHT_YELLOW)} {wc}")
            self._print()
        
        # Traceability
        trace_status = c("✓ VERIFIED", Colors.BRIGHT_GREEN) if traceability_check else c("✗ ISSUES", Colors.BRIGHT_RED)
        self._print(f"    Traceability: {trace_status}")
        self._print(c("      Every trend links back to source signals with URLs", Colors.BRIGHT_BLACK))
        self._print()
        
        # Alerts
        self._metric_line("Alerts generated", alerts_generated)
        self._metric_line("Blind spots detected", len(blind_spots))
        self._metric_line("Weak coverage warnings", len(weak_coverage))
        
        self._step_complete(6, f"Validation complete, {alerts_generated} alerts, {coverage_pct:.0f}% coverage")

    # ═══════════════════════════════════════════════════════════════════════════
    # DEVIATION ALERTS (Nice to Have)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def report_trend_deviations(self, deviations: List[Dict[str, Any]]):
        """
        Report significant trend deviations.
        
        Nice to have: "Alerts when the relevance of individual trends rises or falls
        significantly (e.g., a sharp change in signal volume)."
        """
        if not deviations:
            return
            
        self._print()
        self._print(c("  🔔 TREND DEVIATION ALERTS:", Colors.BRIGHT_YELLOW, Colors.BOLD))
        self._print()
        
        for dev in deviations:
            direction = dev.get("direction", "change")
            trend_name = dev.get("trend_name", "Unknown")
            old_value = dev.get("old_value", 0)
            new_value = dev.get("new_value", 0)
            pct_change = dev.get("pct_change", 0)
            
            if direction == "up":
                icon = c("↑", Colors.BRIGHT_GREEN)
                color = Colors.BRIGHT_GREEN
            else:
                icon = c("↓", Colors.BRIGHT_RED)
                color = Colors.BRIGHT_RED
            
            self._print(f"    {icon} {trend_name}: {old_value} → {new_value} ({c(f'{pct_change:+.0f}%', color)})")
        
        self._print()

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY - Actionable for Amprion
    # ═══════════════════════════════════════════════════════════════════════════
    
    def print_final_summary(self, 
                            top_trends: List[Dict[str, Any]],
                            key_insights: List[str],
                            recommended_actions: List[str]):
        """
        Print executive summary with ACTIONABLE insights for Amprion.
        
        DoD requirement: "transparently assessed technology trends that are relevant
        for Amprion as a transmission system operator and have an impact on IT and 
        digitalization"
        """
        self._print()
        self._header_box("📊 EXECUTIVE SUMMARY FOR AMPRION", "", bg_color=Colors.BG_CYAN)
        
        # Terminology clarification
        self._print(c("  📚 TERMINOLOGY:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print(c("     • Category = One of 16 taxonomy buckets (e.g., 'cybersecurity_ot')", Colors.BRIGHT_BLACK))
        self._print(c("     • Trend = Specific pattern identified by clustering signals within a category", Colors.BRIGHT_BLACK))
        self._print(c("     • Signal = Individual data point (article, paper, news item)", Colors.BRIGHT_BLACK))
        self._print()
        
        self._print(c("  🎯 TOP PRIORITY TRENDS FOR AMPRION TSO OPERATIONS:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print()
        
        for i, t in enumerate(top_trends[:5], 1):
            score = t.get("priority_score", 0)
            name = t.get("name", "Unknown")
            category = t.get("category", "")
            signal_count = t.get("signal_count", 0)
            amprion_tier = t.get("amprion_tier", "medium")
            
            # Get detailed scores and explanations
            scores = t.get("scores", {})
            so_what = scores.get("so_what_summary", "")
            top_drivers = scores.get("shapley_explanation", {}).get("top_drivers", [])
            
            # Tier color
            tier_colors = {
                "existential": Colors.BRIGHT_RED,
                "critical": Colors.BRIGHT_MAGENTA,
                "high": Colors.BRIGHT_YELLOW,
            }
            tier_color = tier_colors.get(amprion_tier, Colors.BRIGHT_CYAN)
            
            # Print trend header with category
            cat_display = category.replace("_", " ").title()
            self._print(f"    {c(f'#{i}', Colors.BRIGHT_YELLOW, Colors.BOLD)} {c(name, Colors.BRIGHT_CYAN, Colors.BOLD)}")
            self._print(f"       Category: {cat_display} | Score: {c(f'{score:.1f}/10', Colors.BRIGHT_YELLOW)} | Tier: {c(amprion_tier.upper(), tier_color)} | Signals: {signal_count}")
            
            # What's the specific insight? (the "trend" within this category)
            trend_insight = self._get_trend_insight(category, signal_count, scores)
            self._print(f"       {c('INSIGHT:', Colors.BRIGHT_WHITE)} {trend_insight}")
            
            # Why it matters for Amprion
            amprion_relevance = self._get_amprion_relevance(category, amprion_tier, scores)
            self._print(f"       {c('AMPRION IMPACT:', Colors.BRIGHT_WHITE)} {amprion_relevance}")
            
            # IT/Digital impact
            it_impact = self._get_it_digital_impact(category, scores)
            self._print(f"       {c('IT/DIGITAL:', Colors.BRIGHT_WHITE)} {it_impact}")
            
            # Concrete action
            concrete_action = self._get_concrete_action(category, score, amprion_tier, scores)
            self._print(f"       {c('→ ACTION:', Colors.BRIGHT_GREEN, Colors.BOLD)} {concrete_action}")
            
            # Top evidence (why we're seeing this)
            if top_drivers:
                self._print(f"       {c('Evidence:', Colors.BRIGHT_BLACK)} {top_drivers[0] if top_drivers else 'Multiple signals'}")
            
            self._print()
        
        # Pipeline statistics
        self._print(c("  📈 SCAN STATISTICS:", Colors.BRIGHT_WHITE, Colors.BOLD))
        for insight in key_insights[:4]:
            self._print(f"    • {insight}")
        
        self._print()
        
        # Value statement for Amprion
        self._print(c("  💼 VALUE FOR AMPRION:", Colors.BRIGHT_WHITE, Colors.BOLD))
        self._print(c("    This scan transforms 1000+ public signals into prioritized, actionable", Colors.WHITE))
        self._print(c("    technology TRENDS (not just categories) directly relevant to TSO operations,", Colors.WHITE))
        self._print(c("    grid infrastructure, and IT/digitalization strategy. Each trend is scored", Colors.WHITE))
        self._print(c("    transparently using Bayesian MCDA with explainable weights, ensuring full", Colors.WHITE))
        self._print(c("    traceability from source signals to strategic recommendations.", Colors.WHITE))
        self._print()

    def _get_trend_insight(self, category: str, signal_count: int, scores: Dict) -> str:
        """Generate specific trend insight based on category and signals."""
        growth_rate = scores.get("growth_rate", 0)
        growth_desc = "accelerating" if growth_rate > 100 else "growing" if growth_rate > 0 else "stable"
        
        insights = {
            "cybersecurity_ot": f"OT/ICS security threats {growth_desc} ({signal_count} signals) — ransomware, supply chain attacks targeting energy sector infrastructure",
            "grid_infrastructure": f"Grid modernization investments {growth_desc} — HVDC corridors, GIS substations, dynamic line rating technologies",
            "offshore_systems": f"Offshore wind grid connections {growth_desc} — 66kV/132kV submarine cables, offshore converter platforms, floating substations",
            "energy_storage": f"Grid-scale storage deployments {growth_desc} — battery storage for frequency response, grid boosters, hybrid systems",
            "renewables_integration": f"Variable renewable integration challenges {growth_desc} — forecasting accuracy, curtailment reduction, grid code compliance",
            "hydrogen_p2g": f"Power-to-Gas projects {growth_desc} — electrolyzer capacity additions, hydrogen pipeline planning, sector coupling",
            "e_mobility_v2g": f"EV charging infrastructure {growth_desc} — V2G pilots, smart charging, distribution grid impacts",
            "ai_grid_optimization": f"AI/ML for grid operations {growth_desc} — predictive maintenance, load forecasting, automated dispatch",
            "digital_twin_simulation": f"Digital twin adoption {growth_desc} — real-time grid modeling, scenario simulation, asset management",
            "regulatory_policy": f"Energy regulatory changes {growth_desc} — network codes, market design, climate targets affecting TSOs",
            "energy_trading": f"Energy market evolution {growth_desc} — intraday trading, balancing markets, cross-border capacity",
            "flexibility_vpp": f"Flexibility aggregation {growth_desc} — virtual power plants, demand response, redispatch optimization",
            "biogas_biomethane": f"Biogas grid injection {growth_desc} — biomethane quality standards, gas grid decarbonization",
            "distributed_generation": f"Distributed energy resources {growth_desc} — prosumer integration, microgrids, local energy communities",
            "power_generation": f"Generation fleet changes {growth_desc} — conventional plant retirements, new capacity planning",
            "grid_stability": f"System stability challenges {growth_desc} — inertia reduction, frequency control, black start capability",
        }
        return insights.get(category, f"Technology developments in {category} ({signal_count} signals detected)")

    def _get_amprion_relevance(self, category: str, tier: str, scores: Dict) -> str:
        """Explain why this matters specifically for Amprion."""
        relevance = {
            "cybersecurity_ot": "Direct threat to Amprion SCADA/EMS systems, control centers, and substation automation — operational continuity risk",
            "grid_infrastructure": "Impacts SuedLink, A-Nord, Korridor B projects — €10B+ transmission expansion program",
            "offshore_systems": "Affects BalWin1/2, DolWin6, BorWin6 connections — offshore wind integration for 2030 targets",
            "energy_storage": "Grid Booster projects (Ottenhofen, Audorf) — redispatch cost reduction, congestion management",
            "renewables_integration": "Core TSO mandate — 65% renewable target requires enhanced forecasting and balancing",
            "hydrogen_p2g": "GET H2 Nukleus project — hydrogen backbone planning, sector coupling infrastructure",
            "e_mobility_v2g": "Distribution grid interface — coordination with DSOs on EV load management",
            "ai_grid_optimization": "Control center modernization — predictive analytics for 11,000km transmission network",
            "digital_twin_simulation": "Asset management optimization — 180+ substations, real-time grid state estimation",
            "regulatory_policy": "Compliance requirements — ENTSO-E network codes, BNetzA regulations, EU clean energy package",
            "energy_trading": "Market operations — European balancing platforms, cross-border capacity allocation",
            "flexibility_vpp": "Redispatch 2.0 implementation — flexibility procurement, congestion management",
            "biogas_biomethane": "Gas-electricity sector coupling — interface with gas TSOs (OGE, Thyssengas)",
            "distributed_generation": "TSO-DSO coordination — visibility of distributed resources for system security",
            "power_generation": "Adequacy planning — reserve capacity, must-run units, system services procurement",
            "grid_stability": "System security mandate — frequency stability, voltage control, black start",
        }
        tier_urgency = {"existential": "CRITICAL for Amprion", "critical": "High priority", "high": "Important"}
        prefix = tier_urgency.get(tier, "Relevant")
        return f"{prefix}: {relevance.get(category, 'Affects TSO operations and planning')}"

    def _get_it_digital_impact(self, category: str, scores: Dict) -> str:
        """Explain IT/digitalization impact."""
        impacts = {
            "cybersecurity_ot": "SOC enhancement, OT network segmentation, ICS security monitoring, incident response procedures",
            "grid_infrastructure": "GIS/asset management systems, project management tools, digital construction workflows",
            "offshore_systems": "Remote monitoring platforms, subsea cable management, offshore SCADA integration",
            "energy_storage": "Battery management systems integration, dispatch optimization algorithms, state-of-charge monitoring",
            "renewables_integration": "Forecasting systems upgrade, probabilistic planning tools, curtailment management",
            "hydrogen_p2g": "Sector coupling data platforms, hydrogen tracking systems, new metering infrastructure",
            "e_mobility_v2g": "EV aggregation platforms, DSO data exchange interfaces, flexibility management systems",
            "ai_grid_optimization": "ML model deployment, data lake expansion, MLOps infrastructure, edge computing",
            "digital_twin_simulation": "3D modeling platforms, real-time simulation engines, cloud computing capacity",
            "regulatory_policy": "Compliance reporting systems, regulatory data interfaces, documentation automation",
            "energy_trading": "Trading platform upgrades, market coupling interfaces, settlement systems",
            "flexibility_vpp": "Flexibility platforms, TSO-DSO data exchange, redispatch automation",
            "biogas_biomethane": "Gas quality monitoring, cross-sector data exchange, metering systems",
            "distributed_generation": "DER visibility platforms, TSO-DSO coordination systems, data aggregation",
            "power_generation": "Generation adequacy tools, scheduling systems, ancillary services platforms",
            "grid_stability": "PMU data analytics, frequency monitoring, stability assessment tools",
        }
        return impacts.get(category, "IT system integration and data management requirements")

    def _get_concrete_action(self, category: str, score: float, tier: str, scores: Dict) -> str:
        """Generate concrete, actionable recommendation."""
        if score >= 8.5 or tier == "existential":
            urgency = "IMMEDIATE (Q1)"
        elif score >= 8.0 or tier == "critical":
            urgency = "SHORT-TERM (Q2)"
        elif score >= 7.5:
            urgency = "MEDIUM-TERM (Q3-Q4)"
        else:
            urgency = "MONITOR (ongoing)"
        
        actions = {
            "cybersecurity_ot": f"{urgency}: Commission OT security assessment, update ICS incident response plan, review vendor access controls",
            "grid_infrastructure": f"{urgency}: Review project timelines vs technology roadmap, assess contractor capabilities, update technical specifications",
            "offshore_systems": f"{urgency}: Evaluate offshore digital solutions, assess remote monitoring gaps, review cable route protection",
            "energy_storage": f"{urgency}: Assess Grid Booster expansion opportunities, evaluate battery technology options, update dispatch algorithms",
            "renewables_integration": f"{urgency}: Upgrade forecasting models, review curtailment procedures, assess grid code compliance tools",
            "hydrogen_p2g": f"{urgency}: Align with GET H2 timeline, assess electrolyzer grid connection requirements, review sector coupling interfaces",
            "e_mobility_v2g": f"{urgency}: Coordinate with DSOs on EV impact studies, evaluate V2G pilot participation, assess charging data needs",
            "ai_grid_optimization": f"{urgency}: Pilot ML-based forecasting, assess data infrastructure readiness, evaluate vendor solutions",
            "digital_twin_simulation": f"{urgency}: Define digital twin scope, assess data quality requirements, evaluate platform options",
            "regulatory_policy": f"{urgency}: Review compliance gaps, engage with BNetzA/ENTSO-E, update internal procedures",
            "energy_trading": f"{urgency}: Assess trading system capabilities, review market participation strategy, evaluate automation options",
            "flexibility_vpp": f"{urgency}: Review Redispatch 2.0 readiness, assess flexibility procurement tools, coordinate with DSOs",
            "biogas_biomethane": f"{urgency}: Review gas-electric interfaces, assess metering requirements, coordinate with gas TSOs",
            "distributed_generation": f"{urgency}: Enhance DER visibility, improve TSO-DSO data exchange, assess aggregation platforms",
            "power_generation": f"{urgency}: Update adequacy assessments, review ancillary services procurement, assess reserve requirements",
            "grid_stability": f"{urgency}: Review inertia studies, assess frequency response needs, evaluate stability tools",
        }
        return actions.get(category, f"{urgency}: Investigate trend implications, assess operational impact, define response strategy")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance for easy import
# ─────────────────────────────────────────────────────────────────────────────
_reporter = None

def get_reporter() -> DemoReporter:
    global _reporter
    if _reporter is None:
        _reporter = DemoReporter()
    return _reporter


# ─────────────────────────────────────────────────────────────────────────────
# Test / Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reporter = DemoReporter()
    
    reporter.start_pipeline("Test Pipeline", "Identify emerging TSO technology trends")
    
    reporter.step_find_signals(
        scraper_results={"arxiv": 245, "rss_news": 645, "tso": 127, "regulatory": 10},
        total_raw=1047,
        total_after_dedup=1013,
        source_breakdown={"research": 245, "news": 450, "tso": 150, "regulatory": 100, "cybersecurity": 68}
    )
    
    reporter.step_cluster_signals(
        total_classified=980,
        category_counts={
            "grid_infrastructure": 156, "cybersecurity_ot": 89, "renewables_integration": 120,
            "offshore_systems": 78, "energy_storage": 95, "hydrogen_p2g": 67
        },
        off_topic_count=33,
        classifier_backend="SentenceTransformer(all-MiniLM-L6-v2)",
        avg_confidence=0.72
    )
    
    reporter.step_derive_trends(
        trends_created=16,
        trend_details=[
            {"name": "Grid Infrastructure & Transmission", "signal_count": 156, "category": "grid_infrastructure"},
            {"name": "Renewables Integration", "signal_count": 120, "category": "renewables_integration"},
            {"name": "Energy Storage Systems", "signal_count": 95, "category": "energy_storage"},
        ],
        min_signals_threshold=3
    )
    
    reporter.step_assess_trends(
        scored_trends=[
            {"name": "Grid Infrastructure", "priority_score": 8.2, "amprion_tier": "critical"},
            {"name": "Cybersecurity OT", "priority_score": 7.8, "amprion_tier": "existential"},
            {"name": "Offshore Systems", "priority_score": 7.5, "amprion_tier": "critical"},
        ],
        ahp_profile="default",
        mcda_weights={"strategic_importance": 0.43, "evidence_strength": 0.23, "growth_momentum": 0.15, "maturity_readiness": 0.19}
    )
    
    reporter.step_prepare_results(
        narratives_generated=16,
        briefs_generated=16,
        deep_dives_generated=16,
        key_players_extracted=89,
        narrative_mode="template"
    )
    
    reporter.step_validate_results(
        coverage_pct=87.5,
        categories_covered=14,
        total_categories=16,
        blind_spots=["quantum_computing"],
        weak_coverage=["biogas_biomethane"],
        alerts_generated=3,
        quality_assessment="HIGH",
        traceability_check=True
    )
    
    reporter.end_pipeline(success=True)
