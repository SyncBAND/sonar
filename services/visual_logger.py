"""
SONAR.AI Visual Logger
======================
Rich, step-by-step visual logging for demo presentation.
Shows clear progression through the AGTS workflow phases.

Usage:
    from services.visual_logger import VisualLogger
    logger = VisualLogger()
    
    with logger.phase("FIND SIGNALS"):
        logger.step("Scraping RSS feeds...")
        logger.metric("Signals collected", 1013)
        logger.success("Data collection complete")
"""

import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import logging

# ─── ANSI Color Codes ─────────────────────────────────────────────────────────
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
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


# ─── Phase Icons ──────────────────────────────────────────────────────────────
PHASE_ICONS = {
    "FIND SIGNALS":     ("🔍", Colors.CYAN),
    "CLUSTER SIGNALS":  ("🧩", Colors.MAGENTA),
    "DERIVE TRENDS":    ("📊", Colors.BLUE),
    "ASSESS TRENDS":    ("⚖️", Colors.GREEN),
    "PREPARE RESULTS":  ("📋", Colors.YELLOW),
    "VALIDATE RESULTS": ("✅", Colors.BRIGHT_GREEN),
    "ALERT":            ("🔔", Colors.RED),
}


class VisualLogger:
    """
    Rich visual logger for SONAR.AI agentic workflow.
    Provides clear, demo-friendly output showing each phase and step.
    """
    
    def __init__(self, 
                 enable_colors: bool = True,
                 log_file: Optional[str] = None,
                 verbose: bool = True):
        self.enable_colors = enable_colors
        self.verbose = verbose
        self.log_file = log_file
        self.current_phase = None
        self.phase_start_time = None
        self.step_count = 0
        self.metrics: Dict[str, Any] = {}
        self.events: List[Dict] = []  # Full event log for traceability
        
        # Standard Python logger for file output
        self._logger = logging.getLogger("sonar.visual")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            ))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
    
    def _c(self, text: str, color: str) -> str:
        """Apply color if enabled."""
        if self.enable_colors:
            return f"{color}{text}{Colors.RESET}"
        return text
    
    def _print(self, msg: str, also_log: bool = True):
        """Print to stdout and optionally to log file."""
        print(msg)
        if also_log and self._logger.handlers:
            # Strip ANSI codes for file logging
            import re
            clean = re.sub(r'\033\[[0-9;]*m', '', msg)
            self._logger.info(clean)
    
    def _record_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """Record event for full traceability."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": self.current_phase,
            "type": event_type,
            "message": message,
            "data": data or {}
        }
        self.events.append(event)
    
    # ─── Banner & Headers ─────────────────────────────────────────────────────
    
    def banner(self, title: str = "SONAR.AI", subtitle: str = "Agentic Global Trend Scanner"):
        """Print startup banner."""
        width = 70
        self._print("")
        self._print(self._c("═" * width, Colors.CYAN))
        self._print(self._c(f"  {title}".center(width), Colors.BOLD + Colors.BRIGHT_CYAN))
        self._print(self._c(f"  {subtitle}".center(width), Colors.CYAN))
        self._print(self._c("═" * width, Colors.CYAN))
        self._print(self._c(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.DIM))
        self._print("")
        self._record_event("BANNER", f"{title} - {subtitle}")
    
    def header(self, text: str):
        """Print section header."""
        self._print("")
        self._print(self._c(f"┌{'─' * 68}┐", Colors.WHITE))
        self._print(self._c(f"│  {text.upper():<64}  │", Colors.BOLD + Colors.WHITE))
        self._print(self._c(f"└{'─' * 68}┘", Colors.WHITE))
        self._record_event("HEADER", text)
    
    # ─── Phase Management ─────────────────────────────────────────────────────
    
    @contextmanager
    def phase(self, name: str):
        """
        Context manager for a workflow phase.
        
        Usage:
            with logger.phase("FIND SIGNALS"):
                # ... do work ...
        """
        self.start_phase(name)
        try:
            yield
        finally:
            self.end_phase()
    
    def start_phase(self, name: str):
        """Start a new workflow phase."""
        self.current_phase = name
        self.phase_start_time = time.time()
        self.step_count = 0
        
        icon, color = PHASE_ICONS.get(name, ("▶", Colors.WHITE))
        
        self._print("")
        self._print(self._c("┏" + "━" * 68 + "┓", color))
        self._print(self._c(f"┃  {icon}  {name:<60}  ┃", Colors.BOLD + color))
        self._print(self._c("┗" + "━" * 68 + "┛", color))
        self._record_event("PHASE_START", name)
    
    def end_phase(self, summary: Optional[str] = None):
        """End the current phase."""
        if self.phase_start_time:
            elapsed = time.time() - self.phase_start_time
            icon, color = PHASE_ICONS.get(self.current_phase, ("▶", Colors.WHITE))
            
            self._print(self._c(f"  └─ ✓ {self.current_phase} complete ", Colors.DIM) + 
                       self._c(f"({elapsed:.2f}s, {self.step_count} steps)", Colors.DIM))
            if summary:
                self._print(self._c(f"     {summary}", Colors.DIM))
            
            self._record_event("PHASE_END", self.current_phase, {
                "elapsed_seconds": round(elapsed, 2),
                "steps": self.step_count
            })
        
        self.current_phase = None
        self.phase_start_time = None
    
    # ─── Step Logging ─────────────────────────────────────────────────────────
    
    def step(self, message: str, detail: Optional[str] = None):
        """Log a workflow step."""
        self.step_count += 1
        prefix = self._c(f"  │ [{self.step_count:02d}]", Colors.DIM)
        self._print(f"{prefix} {message}")
        if detail and self.verbose:
            self._print(self._c(f"  │      └─ {detail}", Colors.DIM))
        self._record_event("STEP", message, {"detail": detail})
    
    def substep(self, message: str):
        """Log a sub-step (indented)."""
        self._print(self._c(f"  │      • {message}", Colors.DIM))
        self._record_event("SUBSTEP", message)
    
    def progress(self, current: int, total: int, item: str = "items"):
        """Show progress indicator."""
        pct = (current / total * 100) if total > 0 else 0
        bar_width = 30
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        self._print(f"\r  │      [{bar}] {current}/{total} {item} ({pct:.0f}%)", also_log=False)
        if current >= total:
            print()  # Newline at completion
    
    # ─── Metrics & Results ────────────────────────────────────────────────────
    
    def metric(self, name: str, value: Any, unit: str = "", highlight: bool = False):
        """Log a metric with optional highlighting."""
        self.metrics[name] = value
        color = Colors.BRIGHT_YELLOW if highlight else Colors.YELLOW
        val_str = f"{value:,}" if isinstance(value, int) else str(value)
        self._print(f"  │  {self._c('→', Colors.DIM)} {name}: {self._c(val_str, color)}{' ' + unit if unit else ''}")
        self._record_event("METRIC", name, {"value": value, "unit": unit})
    
    def table(self, headers: List[str], rows: List[List[Any]], title: Optional[str] = None):
        """Print a formatted table."""
        if title:
            self._print(f"  │  {self._c(title, Colors.BOLD)}")
        
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Header
        header_row = " │ ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        self._print(f"  │  {self._c(header_row, Colors.BOLD)}")
        self._print(f"  │  {'─' * (sum(widths) + 3 * (len(headers) - 1))}")
        
        # Data rows
        for row in rows:
            row_str = " │ ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            self._print(f"  │  {row_str}")
    
    def trend_card(self, rank: int, name: str, score: float, signals: int, 
                   tier: str, action: str, growth: float = 0):
        """Print a formatted trend card."""
        # Score color
        if score >= 8.0:
            score_color = Colors.BRIGHT_GREEN
        elif score >= 6.0:
            score_color = Colors.YELLOW
        else:
            score_color = Colors.RED
        
        # Tier color
        tier_colors = {
            "existential": Colors.BRIGHT_RED,
            "critical": Colors.RED,
            "high": Colors.YELLOW,
            "medium-high": Colors.YELLOW,
            "medium": Colors.CYAN,
            "low": Colors.DIM,
        }
        tier_color = tier_colors.get(tier.lower(), Colors.WHITE)
        
        # Growth indicator
        if growth > 50:
            growth_str = self._c(f"↑↑ +{growth:.0f}%", Colors.BRIGHT_GREEN)
        elif growth > 0:
            growth_str = self._c(f"↑ +{growth:.0f}%", Colors.GREEN)
        elif growth < -20:
            growth_str = self._c(f"↓↓ {growth:.0f}%", Colors.RED)
        elif growth < 0:
            growth_str = self._c(f"↓ {growth:.0f}%", Colors.YELLOW)
        else:
            growth_str = self._c("→ 0%", Colors.DIM)
        
        self._print(f"  │  {self._c(f'#{rank}', Colors.BOLD)} {self._c(name[:45], Colors.BRIGHT_WHITE):<45}")
        self._print(f"  │      Score: {self._c(f'{score:.1f}/10', score_color)}  │  "
                   f"Signals: {signals}  │  Tier: {self._c(tier.upper(), tier_color)}  │  "
                   f"Growth: {growth_str}")
        self._print(f"  │      Action: {self._c(action, Colors.CYAN)}")
        self._print(f"  │  {'─' * 60}")
    
    # ─── Status Messages ──────────────────────────────────────────────────────
    
    def success(self, message: str):
        """Log success message."""
        self._print(f"  │  {self._c('✓', Colors.GREEN)} {self._c(message, Colors.GREEN)}")
        self._record_event("SUCCESS", message)
    
    def warning(self, message: str):
        """Log warning message."""
        self._print(f"  │  {self._c('⚠', Colors.YELLOW)} {self._c(message, Colors.YELLOW)}")
        self._record_event("WARNING", message)
    
    def error(self, message: str):
        """Log error message."""
        self._print(f"  │  {self._c('✗', Colors.RED)} {self._c(message, Colors.RED)}")
        self._record_event("ERROR", message)
    
    def info(self, message: str):
        """Log info message."""
        self._print(f"  │  {self._c('ℹ', Colors.BLUE)} {message}")
        self._record_event("INFO", message)
    
    def alert(self, alert_type: str, message: str, severity: str = "medium"):
        """Log an alert with severity indicator."""
        severity_colors = {
            "critical": Colors.BG_RED + Colors.WHITE,
            "high": Colors.BRIGHT_RED,
            "medium": Colors.YELLOW,
            "low": Colors.CYAN,
        }
        color = severity_colors.get(severity.lower(), Colors.YELLOW)
        
        self._print(f"  │  {self._c('🔔', Colors.RED)} {self._c(f'[{severity.upper()}]', color)} "
                   f"{self._c(alert_type, Colors.BOLD)}: {message}")
        self._record_event("ALERT", message, {"type": alert_type, "severity": severity})
    
    # ─── Source Traceability ──────────────────────────────────────────────────
    
    def source_trace(self, signal_id: str, title: str, source: str, 
                     category: str, confidence: float, url: str):
        """Log source traceability info."""
        conf_color = Colors.GREEN if confidence > 0.7 else (Colors.YELLOW if confidence > 0.5 else Colors.RED)
        self._print(f"  │  {self._c('📎', Colors.DIM)} [{signal_id}] {title[:50]}...")
        self._print(f"  │      Source: {source} → Category: {category} "
                   f"(conf: {self._c(f'{confidence:.2f}', conf_color)})")
        self._print(f"  │      URL: {self._c(url[:60], Colors.DIM)}...")
        self._record_event("SOURCE_TRACE", title, {
            "signal_id": signal_id,
            "source": source,
            "category": category,
            "confidence": confidence,
            "url": url
        })
    
    # ─── Validation Results ───────────────────────────────────────────────────
    
    def validation_result(self, check_name: str, passed: bool, details: str = ""):
        """Log a validation check result."""
        status = self._c("✓ PASS", Colors.GREEN) if passed else self._c("✗ FAIL", Colors.RED)
        self._print(f"  │  {status} {check_name}")
        if details:
            self._print(f"  │      └─ {self._c(details, Colors.DIM)}")
        self._record_event("VALIDATION", check_name, {"passed": passed, "details": details})
    
    # ─── Summary & Export ─────────────────────────────────────────────────────
    
    def summary(self, title: str = "SCAN SUMMARY"):
        """Print final summary."""
        self._print("")
        self._print(self._c("╔" + "═" * 68 + "╗", Colors.GREEN))
        self._print(self._c(f"║  {title:<64}  ║", Colors.BOLD + Colors.GREEN))
        self._print(self._c("╠" + "═" * 68 + "╣", Colors.GREEN))
        
        for name, value in self.metrics.items():
            val_str = f"{value:,}" if isinstance(value, int) else str(value)
            self._print(self._c(f"║  {name:<30} │ {val_str:>31}  ║", Colors.GREEN))
        
        self._print(self._c("╚" + "═" * 68 + "╝", Colors.GREEN))
        self._print("")
    
    def get_trace_log(self) -> List[Dict]:
        """Get full event trace for audit/export."""
        return self.events
    
    def export_trace(self, filepath: str):
        """Export trace log to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "generated_at": datetime.utcnow().isoformat(),
                "metrics": self.metrics,
                "events": self.events
            }, f, indent=2)
        self._print(f"  │  {self._c('📁', Colors.DIM)} Trace exported to: {filepath}")


# ─── Global Instance ──────────────────────────────────────────────────────────
_visual_logger: Optional[VisualLogger] = None

def get_visual_logger() -> VisualLogger:
    """Get or create global visual logger instance."""
    global _visual_logger
    if _visual_logger is None:
        _visual_logger = VisualLogger()
    return _visual_logger

def reset_visual_logger():
    """Reset the global logger (for testing)."""
    global _visual_logger
    _visual_logger = None
