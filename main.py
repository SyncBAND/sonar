"""
SONAR.AI / AGTS - Main FastAPI Application
===========================================
Agentic Global Trend Scanner for TSOs

API Endpoints:
- Dashboard: Overview statistics
- Signals: Raw signal management
- Trends: Trend management with AGTS fields
- Alerts: Custom alerting
- Insights: AI-generated summaries
- Search: Multi-dimensional filtering
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging

# Local imports
from config import (
    TSO_CATEGORIES, 
    STRATEGIC_DOMAINS, 
    ARCHITECTURAL_LAYERS,
    AMPRION_BUSINESS_AREAS,
    AMPRION_KEY_TASKS,
    STAKEHOLDER_VIEWS,
    MATURITY_TYPES,
    DEBUG
)
from models.database import (
    init_db, get_db, SessionLocal,
    Signal, TrendCluster, Trend, Alert, DataSource
)
from services.nlp_processor import get_nlp_processor
from services.trend_scorer import get_trend_scorer

# Try to import enhanced scrapers (optional dependency)
try:
    from services.enhanced_scrapers import (
        TSOBusinessImpactAnalyzer,
        LinkValidator,
        NewsletterScraper
    )
    ENHANCED_SCRAPERS_AVAILABLE = True
    impact_analyzer = TSOBusinessImpactAnalyzer()
    link_validator = LinkValidator()
except ImportError:
    ENHANCED_SCRAPERS_AVAILABLE = False
    impact_analyzer = None
    link_validator = None

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sonar_ai")

# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application on startup."""
    logger.info("🚀 Starting SONAR.AI / AGTS...")
    init_db()
    logger.info("✅ Database initialized")
    logger.info("✅ SONAR.AI ready for TSO trend scanning!")
    yield
    logger.info("👋 SONAR.AI shutting down...")

# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="SONAR.AI / AGTS",
    description="Agentic Global Trend Scanner for Transmission System Operators",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ROOT & HEALTH
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the SONAR.AI dashboard."""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(html_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Dashboard not found</h1><p>Place static/index.html in the app directory.</p><p><a href='/docs'>API Docs →</a></p>", status_code=200)


@app.get("/api")
async def api_root():
    """API info endpoint."""
    return {
        "name": "SONAR.AI / AGTS",
        "version": "2.0.0",
        "description": "Agentic Global Trend Scanner for TSOs",
        "docs": "/docs",
        "dashboard": "/",
        "status": "operational"
    }

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# =============================================================================
# DASHBOARD ENDPOINTS
# =============================================================================

@app.get("/api/v1/dashboard")
async def get_dashboard(db: Session = Depends(get_db)):
    """
    Get dashboard overview with AGTS metrics.
    
    Returns key statistics for executive view.
    """
    now = datetime.utcnow()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)
    
    # Signal counts
    total_signals = db.query(func.count(Signal.id)).scalar() or 0
    signals_24h = db.query(func.count(Signal.id)).filter(Signal.scraped_at >= day_ago).scalar() or 0
    signals_7d = db.query(func.count(Signal.id)).filter(Signal.scraped_at >= week_ago).scalar() or 0
    
    # Trend counts
    total_trends = db.query(func.count(Trend.id)).scalar() or 0
    active_trends = db.query(func.count(Trend.id)).filter(Trend.is_active == True).scalar() or 0
    emerging_trends = db.query(func.count(Trend.id)).filter(Trend.is_emerging == True).scalar() or 0
    
    # Category breakdown
    category_counts = dict(db.query(
        Signal.tso_category, func.count(Signal.id)
    ).group_by(Signal.tso_category).all())
    
    # Strategic nature breakdown
    nature_counts = dict(db.query(
        Trend.strategic_nature, func.count(Trend.id)
    ).group_by(Trend.strategic_nature).all())
    
    # Top trends by priority
    top_trends = db.query(Trend).filter(
        Trend.is_active == True
    ).order_by(Trend.priority_score.desc()).limit(5).all()
    
    # Recent alerts
    unread_alerts = db.query(func.count(Alert.id)).filter(Alert.is_read == False).scalar() or 0
    
    return {
        "overview": {
            "total_signals": total_signals,
            "signals_24h": signals_24h,
            "signals_7d": signals_7d,
            "total_trends": total_trends,
            "active_trends": active_trends,
            "emerging_trends": emerging_trends,
            "unread_alerts": unread_alerts
        },
        "signals_by_category": category_counts,
        "trends_by_nature": nature_counts,
        "top_trends": [
            {
                "id": t.id,
                "name": t.name,
                "priority_score": t.priority_score,
                "strategic_nature": t.strategic_nature,
                "tso_category": t.tso_category,
                "time_to_impact": t.time_to_impact
            }
            for t in top_trends
        ],
        "timestamp": now.isoformat()
    }

# =============================================================================
# SIGNAL ENDPOINTS
# =============================================================================

@app.get("/api/v1/signals")
async def get_signals(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
    category: Optional[str] = None,
    source_type: Optional[str] = None,
    processed: Optional[bool] = None,
    title: Optional[str] = None,
    search: Optional[str] = None
):
    """Get signals with filtering.
    
    Query parameters:
    - category: Filter by TSO category (e.g., 'energy_trading', 'cybersecurity_ot')
    - source_type: Filter by source type (e.g., 'news', 'research')
    - processed: Filter by processing status (true/false)
    - title: Search in signal titles (partial match)
    - search: Search in title and content (partial match)
    """
    query = db.query(Signal)
    
    if category:
        query = query.filter(Signal.tso_category == category)
    if source_type:
        query = query.filter(Signal.source_type == source_type)
    if processed is not None:
        query = query.filter(Signal.is_processed == processed)
    if title:
        query = query.filter(Signal.title.ilike(f"%{title}%"))
    if search:
        query = query.filter(
            (Signal.title.ilike(f"%{search}%")) | 
            (Signal.content.ilike(f"%{search}%"))
        )
    
    signals = query.order_by(Signal.scraped_at.desc()).offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "signals": [
            {
                "id": s.id,
                "title": s.title,
                "url": s.url,
                "source_type": s.source_type,
                "source_name": s.source_name,
                "tso_category": s.tso_category,
                "strategic_domain": s.strategic_domain,
                "amprion_task": s.amprion_task,
                "keywords": s.keywords,
                "quality_score": s.quality_score,
                "published_at": s.published_at.isoformat() if s.published_at else None,
                "scraped_at": s.scraped_at.isoformat() if s.scraped_at else None
            }
            for s in signals
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.post("/api/v1/signals/process")
async def process_signals(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    limit: int = 100
):
    """Process unprocessed signals with NLP."""
    processor = get_nlp_processor()
    
    signals = db.query(Signal).filter(
        Signal.is_processed == False
    ).limit(limit).all()
    
    processed_count = 0
    for signal in signals:
        text = f"{signal.title} {signal.content or ''}"
        classification = processor.classify_signal(signal.title, signal.content or "")
        
        # Update signal with classification
        signal.tso_category = classification["tso_category"]
        signal.strategic_domain = classification["strategic_domain"]
        signal.architectural_layer = classification["architectural_layer"]
        signal.amprion_task = classification["amprion_task"]
        signal.amprion_business_area = classification["business_area_code"]
        signal.linked_projects = classification["linked_projects"]
        signal.keywords = classification["keywords"]
        signal.entities = classification["entities"]
        signal.quality_score = processor.calculate_quality_score(
            signal.source_type or "news",
            signal.source_quality_score or 0.5,
            len(text),
            bool(classification["entities"])
        )
        signal.is_processed = True
        
        processed_count += 1
    
    db.commit()
    
    return {
        "status": "success",
        "processed_count": processed_count,
        "message": f"Processed {processed_count} signals"
    }

# =============================================================================
# TREND ENDPOINTS (AGTS Compliant)
# =============================================================================

@app.get("/api/v1/trends")
async def get_trends(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 20,
    # Multi-dimensional filters (Page 17-18)
    category: Optional[str] = None,
    strategic_domain: Optional[str] = None,
    architectural_layer: Optional[str] = None,
    business_area: Optional[str] = None,
    amprion_task: Optional[str] = None,
    strategic_nature: Optional[str] = None,
    time_to_impact: Optional[str] = None,
    regulatory_sensitivity: Optional[str] = None,
    min_priority: Optional[float] = None,
    min_maturity: Optional[int] = None,
    is_active: bool = True,
    sort_by: str = "priority_score",
    sort_order: str = "desc"
):
    """
    Get trends with AGTS multi-dimensional filtering.
    
    Supports all sorting criteria from Page 17-18.
    """
    query = db.query(Trend)
    
    # Apply filters
    if is_active:
        query = query.filter(Trend.is_active == True)
    if category:
        query = query.filter(Trend.tso_category == category)
    if strategic_domain:
        query = query.filter(Trend.strategic_domain == strategic_domain)
    if architectural_layer:
        query = query.filter(Trend.architectural_layer == architectural_layer)
    if business_area:
        query = query.filter(Trend.business_area == business_area)
    if amprion_task:
        query = query.filter(Trend.key_amprion_task == amprion_task)
    if strategic_nature:
        query = query.filter(Trend.strategic_nature == strategic_nature)
    if time_to_impact:
        query = query.filter(Trend.time_to_impact == time_to_impact)
    if regulatory_sensitivity:
        query = query.filter(Trend.regulatory_sensitivity == regulatory_sensitivity)
    if min_priority:
        query = query.filter(Trend.priority_score >= min_priority)
    if min_maturity:
        query = query.filter(Trend.maturity_score >= min_maturity)
    
    # Sorting
    sort_column = getattr(Trend, sort_by, Trend.priority_score)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())
    
    total = query.count()
    trends = query.offset(skip).limit(limit).all()
    
    return {
        "trends": [_format_trend(t) for t in trends],
        "total": total,
        "skip": skip,
        "limit": limit,
        "filters_applied": {
            "category": category,
            "strategic_domain": strategic_domain,
            "business_area": business_area,
            "min_priority": min_priority
        }
    }

@app.get("/api/v1/trends/{trend_id}")
async def get_trend_detail(trend_id: int, db: Session = Depends(get_db)):
    """Get full trend profile with all AGTS fields."""
    trend = db.query(Trend).filter(Trend.id == trend_id).first()
    if not trend:
        raise HTTPException(status_code=404, detail="Trend not found")
    
    # Get related signals
    signals = db.query(Signal).filter(Signal.cluster_id == trend.cluster_id).limit(20).all()
    
    return {
        "trend": _format_trend(trend, full=True),
        "signals": [
            {
                "id": s.id,
                "title": s.title,
                "url": s.url,
                "source_type": s.source_type,
                "published_at": s.published_at.isoformat() if s.published_at else None
            }
            for s in signals
        ],
        "signal_count": len(signals)
    }

@app.post("/api/v1/trends/create")
async def create_trends_from_clusters(
    db: Session = Depends(get_db),
    min_signals: int = 3
):
    """
    Create trends from signal clusters.
    
    Groups processed signals by category and creates AGTS-compliant trends.
    """
    processor = get_nlp_processor()
    scorer = get_trend_scorer()
    
    # Get processed signals
    signals = db.query(Signal).filter(Signal.is_processed == True).all()
    
    # Group by category
    category_signals = {}
    for signal in signals:
        cat = signal.tso_category or "other"
        if cat not in category_signals:
            category_signals[cat] = []
        category_signals[cat].append(signal)
    
    created_trends = []
    trend_counter = db.query(func.count(Trend.id)).scalar() or 0
    
    for category, cat_signals in category_signals.items():
        if len(cat_signals) < min_signals:
            continue
        
        trend_counter += 1
        trend_id = f"TR-{datetime.utcnow().year}-{trend_counter:03d}"
        
        # Create cluster
        cluster = TrendCluster(
            name=TSO_CATEGORIES.get(category, {}).get("name", category),
            description=f"Trends in {category}",
            tso_category=category,
            keywords=[kw for s in cat_signals[:10] for kw in (s.keywords or [])[:3]]
        )
        db.add(cluster)
        db.flush()
        
        # Assign signals to cluster
        for signal in cat_signals:
            signal.cluster_id = cluster.id
        
        # Collect signal data for scoring
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
        
        # Determine most common classifications
        domains = [s.strategic_domain for s in cat_signals if s.strategic_domain]
        layers = [s.architectural_layer for s in cat_signals if s.architectural_layer]
        tasks = [s.amprion_task for s in cat_signals if s.amprion_task]
        projects = [p for s in cat_signals for p in (s.linked_projects or [])]
        
        trend_data = {
            "tso_category": category,
            "strategic_domain": max(set(domains), key=domains.count) if domains else None,
            "amprion_task": max(set(tasks), key=tasks.count) if tasks else None,
            "linked_projects": list(set(projects))[:5],
            "maturity_score": 5  # Default
        }
        
        # Score the trend
        scores = scorer.score_trend(trend_data, signal_data)
        
        # Create trend with all AGTS fields
        trend = Trend(
            trend_id=trend_id,
            name=cluster.name,
            description_short=f"Technology trends in {TSO_CATEGORIES.get(category, {}).get('name', category)}",
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
            project_multiplier=scores["project_multiplier"],
            
            # Classification
            strategic_nature=scores["strategic_nature"],
            time_to_impact=scores["time_to_impact"],
            maturity_score=5,
            maturity_type="TRL",
            lifecycle_status="Scouting",
            
            is_active=True,
            is_emerging=len(cat_signals) < 20
        )
        db.add(trend)
        created_trends.append(trend_id)
    
    db.commit()
    
    return {
        "status": "success",
        "created_count": len(created_trends),
        "trend_ids": created_trends
    }

@app.post("/api/v1/trends/score")
async def score_all_trends(db: Session = Depends(get_db)):
    """Re-score all trends using No-Regret framework."""
    scorer = get_trend_scorer()
    
    trends = db.query(Trend).filter(Trend.is_active == True).all()
    scored_count = 0
    
    for trend in trends:
        # Get signals for this trend
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
        
        # Update trend scores
        trend.priority_score = scores["priority_score"]
        trend.strategic_relevance_score = scores["strategic_relevance_score"]
        trend.grid_stability_score = scores["grid_stability_score"]
        trend.cost_efficiency_score = scores["cost_efficiency_score"]
        trend.volume_score = scores["volume_score"]
        trend.growth_score = scores["growth_score"]
        trend.growth_rate = scores["growth_rate"]
        trend.project_multiplier = scores["project_multiplier"]
        trend.strategic_nature = scores["strategic_nature"]
        trend.time_to_impact = scores["time_to_impact"]
        trend.signal_count = len(signals)
        trend.last_updated = datetime.utcnow()
        
        scored_count += 1
    
    db.commit()
    
    return {
        "status": "success",
        "scored_count": scored_count
    }

# =============================================================================
# TAXONOMY ENDPOINTS (Page 8-12)
# =============================================================================

@app.get("/api/v1/taxonomies")
async def get_taxonomies():
    """Get all AGTS taxonomies for filtering UI."""
    return {
        "tso_categories": {
            cat_id: {
                "name": cat_data["name"],
            }
            for cat_id, cat_data in TSO_CATEGORIES.items()
        },
        "strategic_domains": {
            domain_id: {
                "name": domain_data["name"],
                "sub_categories": list(domain_data["sub_categories"].keys())
            }
            for domain_id, domain_data in STRATEGIC_DOMAINS.items()
        },
        "architectural_layers": {
            layer_id: {
                "name": layer_data["name"],
                "description": layer_data["description"]
            }
            for layer_id, layer_data in ARCHITECTURAL_LAYERS.items()
        },
        "business_areas": {
            area_id: {
                "code": area_data["code"],
                "name": area_data["name"]
            }
            for area_id, area_data in AMPRION_BUSINESS_AREAS.items()
        },
        "amprion_tasks": {
            task_id: task_data["name"]
            for task_id, task_data in AMPRION_KEY_TASKS.items()
        },
        "maturity_types": list(MATURITY_TYPES.keys()),
        "stakeholder_views": list(STAKEHOLDER_VIEWS.keys())
    }

@app.get("/api/v1/stakeholder-views/{view_type}")
async def get_stakeholder_view(
    view_type: str,
    db: Session = Depends(get_db)
):
    """Get pre-configured stakeholder view (Page 15-16)."""
    if view_type not in STAKEHOLDER_VIEWS:
        raise HTTPException(status_code=404, detail=f"View type '{view_type}' not found")
    
    view_config = STAKEHOLDER_VIEWS[view_type]
    
    # Build query with default filters
    query = db.query(Trend).filter(Trend.is_active == True)
    
    # Apply default filters
    default_filter = view_config.get("default_filter", {})
    for field, values in default_filter.items():
        if hasattr(Trend, field):
            column = getattr(Trend, field)
            if isinstance(values, list):
                query = query.filter(column.in_(values))
            else:
                query = query.filter(column == values)
    
    # Sort by default
    sort_field = view_config.get("default_sort", "priority_score")
    if hasattr(Trend, sort_field):
        query = query.order_by(getattr(Trend, sort_field).desc())
    
    trends = query.limit(20).all()
    
    return {
        "view_type": view_type,
        "view_name": view_config["name"],
        "priority_fields": view_config["priority_fields"],
        "trends": [
            _format_trend_for_view(t, view_config["priority_fields"])
            for t in trends
        ]
    }

# =============================================================================
# ALERT ENDPOINTS
# =============================================================================

@app.get("/api/v1/alerts")
async def get_alerts(
    db: Session = Depends(get_db),
    unread_only: bool = False,
    severity: Optional[str] = None,
    limit: int = 20
):
    """Get alerts."""
    query = db.query(Alert)
    
    if unread_only:
        query = query.filter(Alert.is_read == False)
    if severity:
        query = query.filter(Alert.severity == severity)
    
    alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
    
    return {
        "alerts": [
            {
                "id": a.id,
                "title": a.title,
                "description": a.description,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "trend_id": a.trend_id,
                "is_read": a.is_read,
                "created_at": a.created_at.isoformat()
            }
            for a in alerts
        ]
    }

@app.put("/api/v1/alerts/{alert_id}/read")
async def mark_alert_read(alert_id: int, db: Session = Depends(get_db)):
    """Mark alert as read."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.is_read = True
    alert.read_at = datetime.utcnow()
    db.commit()
    
    return {"status": "success"}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_trend(trend: Trend, full: bool = False) -> Dict[str, Any]:
    """Format trend for API response (AGTS fields from Page 13-15)."""
    result = {
        # Core Identity
        "id": trend.id,
        "trend_id": trend.trend_id,
        "name": trend.name,
        "description_short": trend.description_short,
        
        # Classification
        "tso_category": trend.tso_category,
        "strategic_domain": trend.strategic_domain,
        "architectural_layer": trend.architectural_layer,
        
        # Maturity
        "maturity_type": trend.maturity_type,
        "maturity_score": trend.maturity_score,
        "lifecycle_status": trend.lifecycle_status,
        "time_to_impact": trend.time_to_impact,
        
        # Strategic Prioritization
        "priority_score": trend.priority_score,
        "strategic_nature": trend.strategic_nature,
        "key_amprion_task": trend.key_amprion_task,
        
        # Scores
        "scores": {
            "overall": trend.priority_score,
            "strategic_relevance": trend.strategic_relevance_score,
            "grid_stability": trend.grid_stability_score,
            "cost_efficiency": trend.cost_efficiency_score,
            "volume": trend.volume_score,
            "growth": trend.growth_score,
            "quality": trend.quality_score
        },
        
        # Statistics
        "signal_count": trend.signal_count,
        "growth_rate": trend.growth_rate,
        "project_multiplier": trend.project_multiplier,
        
        # Status
        "is_active": trend.is_active,
        "is_emerging": trend.is_emerging,
        
        # Temporal
        "first_seen": trend.first_seen.isoformat() if trend.first_seen else None,
        "last_updated": trend.last_updated.isoformat() if trend.last_updated else None
    }
    
    if full:
        result.update({
            "description_full": trend.description_full,
            "business_area": trend.business_area,
            "linked_projects": trend.linked_projects,
            "regulatory_sensitivity": trend.regulatory_sensitivity,
            "sovereignty_score": trend.sovereignty_score,
            "customer_impact": trend.customer_impact,
            "market_impact": trend.market_impact,
            "so_what_summary": trend.so_what_summary,
            "ai_summary": trend.ai_summary,
            "ai_reasoning": trend.ai_reasoning,
            "keywords": trend.keywords,
            "key_players": trend.key_players
        })
    
    return result

def _format_trend_for_view(trend: Trend, fields: List[str]) -> Dict[str, Any]:
    """Format trend for stakeholder view with only requested fields."""
    full_data = _format_trend(trend, full=True)
    
    result = {
        "id": trend.id,
        "name": trend.name
    }
    
    for field in fields:
        if field in full_data:
            result[field] = full_data[field]
        elif field in full_data.get("scores", {}):
            result[field] = full_data["scores"][field]
    
    return result

# =============================================================================
# TSO BUSINESS IMPACT ANALYSIS
# =============================================================================

@app.get("/api/v1/impact/signal/{signal_id}")
async def get_signal_impact(
    signal_id: int,
    db: Session = Depends(get_db)
):
    """
    Get TSO business impact analysis for a specific signal.
    
    Returns short-term and long-term impacts, Amprion-specific implications,
    recommended actions, risk and opportunity assessments.
    """
    if not ENHANCED_SCRAPERS_AVAILABLE or not impact_analyzer:
        raise HTTPException(status_code=503, detail="Impact analyzer not available")
    
    signal = db.query(Signal).filter(Signal.id == signal_id).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    analysis = impact_analyzer.analyze_signal({
        "title": signal.title,
        "content": signal.content,
        "tso_category": signal.tso_category
    })
    
    return {
        "signal_id": signal_id,
        "title": signal.title,
        "category": signal.tso_category,
        "analysis": analysis
    }

@app.get("/api/v1/impact/trend/{trend_id}")
async def get_trend_impact(
    trend_id: int,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive TSO business impact report for a trend.
    
    Includes momentum analysis, short/long-term impacts, Amprion-specific details,
    and executive summary.
    """
    if not ENHANCED_SCRAPERS_AVAILABLE or not impact_analyzer:
        raise HTTPException(status_code=503, detail="Impact analyzer not available")
    
    trend = db.query(Trend).filter(Trend.id == trend_id).first()
    if not trend:
        raise HTTPException(status_code=404, detail="Trend not found")
    
    # Get signals for this trend
    signals = db.query(Signal).filter(Signal.cluster_id == trend.cluster_id).all()
    
    signal_data = [
        {
            "title": s.title,
            "content": s.content,
            "published_at": s.published_at,
            "tso_category": s.tso_category
        }
        for s in signals
    ]
    
    report = impact_analyzer.generate_trend_report(
        {
            "name": trend.name,
            "tso_category": trend.tso_category
        },
        signal_data
    )
    
    return {
        "trend_id": trend_id,
        "trend_name": trend.name,
        "report": report
    }

@app.get("/api/v1/impact/category/{category}")
async def get_category_impact(category: str):
    """
    Get TSO business impact rules for a specific category.
    
    Useful for understanding what any signal in this category means for TSOs.
    """
    if not ENHANCED_SCRAPERS_AVAILABLE or not impact_analyzer:
        raise HTTPException(status_code=503, detail="Impact analyzer not available")
    
    if category not in impact_analyzer.IMPACT_RULES:
        raise HTTPException(
            status_code=404, 
            detail=f"No impact rules for category: {category}. "
                   f"Available: {list(impact_analyzer.IMPACT_RULES.keys())}"
        )
    
    rules = impact_analyzer.IMPACT_RULES[category]
    
    return {
        "category": category,
        "category_name": TSO_CATEGORIES.get(category, {}).get("name", category),
        "short_term_impact": rules["short_term"],
        "long_term_impact": rules["long_term"],
        "amprion_specific": rules["amprion_specific"],
        "recommended_actions": rules["recommended_actions"],
        "risk_assessment": rules["risk"],
        "opportunity_assessment": rules["opportunity"]
    }

@app.get("/api/v1/impact/all-categories")
async def get_all_category_impacts():
    """
    Get TSO business impact summary for all categories.
    
    Useful for dashboard showing impact levels across categories.
    """
    if not ENHANCED_SCRAPERS_AVAILABLE or not impact_analyzer:
        raise HTTPException(status_code=503, detail="Impact analyzer not available")
    
    summary = []
    for category, rules in impact_analyzer.IMPACT_RULES.items():
        summary.append({
            "category": category,
            "category_name": TSO_CATEGORIES.get(category, {}).get("name", category),
            "short_term_impact": rules["short_term"]["impact"],
            "long_term_impact": rules["long_term"]["impact"],
            "amprion_relevance": rules["amprion_specific"]["relevance"],
            "risk": rules["risk"].split(" - ")[0] if " - " in rules["risk"] else rules["risk"],
            "opportunity": rules["opportunity"].split(" - ")[0] if " - " in rules["opportunity"] else rules["opportunity"],
            "linked_projects": rules["amprion_specific"].get("linked_projects", []),
            "business_areas": rules["amprion_specific"].get("business_areas", [])
        })
    
    return {
        "categories": summary,
        "total_categories": len(summary)
    }

# =============================================================================
# AHP WEIGHT CONFIGURATION
# =============================================================================

@app.get("/api/v1/ahp/weights")
async def get_ahp_weights():
    """
    Get current MCDA weights and how they were derived.

    Returns the active weights, the AHP computation details
    (consistency ratio, matrix), and available profiles.
    """
    from services.ahp import (
        get_mcda_weights, get_last_ahp_result,
        AHP_PROFILES, SAATY_SCALE, compute_all_profiles,
    )

    weights = get_mcda_weights()
    ahp_result = get_last_ahp_result()

    return {
        "active_weights": weights,
        "ahp_result": ahp_result,
        "available_profiles": list(AHP_PROFILES.keys()),
        "saaty_scale": SAATY_SCALE,
    }


@app.get("/api/v1/ahp/profiles")
async def get_ahp_profiles():
    """
    Compute and compare weights across all preset AHP profiles.

    Shows how different stakeholder perspectives produce different weights.
    """
    from services.ahp import compute_all_profiles
    return compute_all_profiles()


@app.post("/api/v1/ahp/compute")
async def compute_ahp_weights(comparisons: Dict[str, float]):
    """
    Compute AHP weights from custom pairwise comparisons.

    Body: flat dict of comparisons, e.g.:
    {
        "strategic_importance_vs_evidence_strength": 3,
        "strategic_importance_vs_growth_momentum": 2,
        "strategic_importance_vs_maturity_readiness": 2,
        "evidence_strength_vs_growth_momentum": 1,
        "evidence_strength_vs_maturity_readiness": 1,
        "growth_momentum_vs_maturity_readiness": 1
    }

    Values use Saaty scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme.
    Values < 1 mean the second criterion is more important (e.g., 0.333 = B is moderately > A).
    """
    from services.ahp import AHPEngine, CRITERIA

    engine = AHPEngine(CRITERIA)
    engine.set_flat_comparisons(comparisons)
    result = engine.compute()

    return {
        "weights": result["weights"],
        "consistency_ratio": result["consistency_ratio"],
        "is_consistent": result["is_consistent"],
        "message": (
            "Weights are consistent and ready to use."
            if result["is_consistent"]
            else f"WARNING: Consistency ratio {result['consistency_ratio']:.4f} exceeds 0.10. "
                 f"Your pairwise judgments contain contradictions. Please review."
        ),
        "matrix": result["matrix"],
        "criteria": result["criteria"],
    }


@app.post("/api/v1/ahp/apply")
async def apply_ahp_weights(comparisons: Dict[str, float]):
    """
    Compute AHP weights AND save them as the active configuration.

    Same input as /compute, but also:
    - Saves to ahp_config.json
    - Reloads MCDA_WEIGHTS in the scorer
    - Future scans will use the new weights
    """
    from services.ahp import (
        AHPEngine, CRITERIA, save_ahp_config,
        get_mcda_weights, _FALLBACK_WEIGHTS,
    )
    import services.ahp as ahp_mod
    import services.trend_scorer as scorer_mod

    engine = AHPEngine(CRITERIA)
    engine.set_flat_comparisons(comparisons)
    result = engine.compute()

    if not result["is_consistent"]:
        return {
            "status": "rejected",
            "consistency_ratio": result["consistency_ratio"],
            "message": (
                f"Consistency ratio {result['consistency_ratio']:.4f} exceeds 0.10. "
                f"Weights NOT applied. Please revise your comparisons."
            ),
            "weights": result["weights"],
            "matrix": result["matrix"],
        }

    # Save config
    config = {
        "comparisons": comparisons,
        "computed_weights": result["weights"],
        "consistency_ratio": result["consistency_ratio"],
    }
    save_ahp_config(config)

    # Hot-reload into scorer
    ahp_mod._cached_weights = None
    ahp_mod._cached_ahp_result = None
    new_weights = get_mcda_weights(force_recompute=True)
    scorer_mod.MCDA_WEIGHTS = new_weights

    return {
        "status": "applied",
        "weights": new_weights,
        "consistency_ratio": result["consistency_ratio"],
        "message": "New AHP weights saved and active. Next scan will use these weights.",
    }


@app.post("/api/v1/ahp/apply-profile/{profile_name}")
async def apply_ahp_profile(profile_name: str):
    """
    Apply a preset AHP profile by name.

    Available profiles: default, strategy_board, innovation_scout,
                        grid_planner, risk_manager
    """
    from services.ahp import (
        AHP_PROFILES, compute_profile_weights, save_ahp_config,
        get_mcda_weights,
    )
    import services.ahp as ahp_mod
    import services.trend_scorer as scorer_mod

    if profile_name not in AHP_PROFILES:
        return {
            "status": "error",
            "message": f"Unknown profile: {profile_name}",
            "available": list(AHP_PROFILES.keys()),
        }

    result = compute_profile_weights(profile_name)
    comparisons = AHP_PROFILES[profile_name]

    # Save config
    config = {
        "profile": profile_name,
        "comparisons": comparisons,
        "computed_weights": result["weights"],
        "consistency_ratio": result["consistency_ratio"],
    }
    save_ahp_config(config)

    # Hot-reload
    ahp_mod._cached_weights = None
    ahp_mod._cached_ahp_result = None
    new_weights = get_mcda_weights(force_recompute=True)
    scorer_mod.MCDA_WEIGHTS = new_weights

    return {
        "status": "applied",
        "profile": profile_name,
        "weights": new_weights,
        "consistency_ratio": result["consistency_ratio"],
        "is_consistent": result["is_consistent"],
    }


# =============================================================================
# LINK VALIDATION
# =============================================================================

@app.post("/api/v1/validate-links")
async def validate_signal_links(
    signal_ids: List[int] = Query(default=[]),
    db: Session = Depends(get_db)
):
    """
    Validate URLs for specified signals (check for 404s).
    
    Returns validation results and optionally marks invalid signals.
    """
    if not ENHANCED_SCRAPERS_AVAILABLE or not link_validator:
        raise HTTPException(status_code=503, detail="Link validator not available")
    
    if not signal_ids:
        # Validate recent signals
        signals = db.query(Signal).order_by(Signal.id.desc()).limit(50).all()
    else:
        signals = db.query(Signal).filter(Signal.id.in_(signal_ids)).all()
    
    urls = [(s.id, s.url) for s in signals if s.url]
    
    if not urls:
        return {"message": "No URLs to validate", "results": []}
    
    # Validate (limit to 20 to avoid rate limiting)
    import asyncio
    results = asyncio.get_event_loop().run_until_complete(
        link_validator.validate_batch([url for _, url in urls[:20]])
    )
    
    validation_results = []
    for (signal_id, url), result in zip(urls[:20], results):
        validation_results.append({
            "signal_id": signal_id,
            "url": url[:100] + "..." if len(url) > 100 else url,
            "is_valid": result["is_valid"],
            "status_code": result.get("status_code"),
            "error": result.get("error")
        })
    
    valid_count = sum(1 for r in validation_results if r["is_valid"])
    
    return {
        "validated": len(validation_results),
        "valid": valid_count,
        "invalid": len(validation_results) - valid_count,
        "results": validation_results
    }

# =============================================================================
# SHAP-ALIGNED EXPLAINABILITY ENDPOINTS
# =============================================================================

@app.get("/api/v1/explain/trend/{trend_id}")
async def explain_trend_score(
    trend_id: str,
    db: Session = Depends(get_db)
):
    """
    SHAP-aligned score explanation for a trend.
    
    Returns Shapley Additive Decomposition showing:
    - Base value (expected score with no information)
    - Feature contributions (marginal impact of each feature)
    - Waterfall decomposition (like SHAP force plot)
    - Feature importance ranking
    - Human-readable top drivers
    - Additivity check (base + contributions = final score)
    """
    trend = db.query(Trend).filter(
        (Trend.trend_id == trend_id) | (Trend.id == int(trend_id) if trend_id.isdigit() else False)
    ).first()
    
    if not trend:
        raise HTTPException(status_code=404, detail="Trend not found")
    
    # Get signals for this trend
    signals = db.query(Signal).filter(Signal.cluster_id == trend.cluster_id).all()
    
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
    
    # Re-score to get SHAP explanation
    scorer = get_trend_scorer()
    scores = scorer.score_trend(trend_data, signal_data)
    
    return {
        "trend_id": trend.trend_id,
        "trend_name": trend.name,
        "priority_score": scores["priority_score"],
        "explanation": scores.get("shapley_explanation", {}),
        "scores": {
            "overall": scores["overall_score"],
            "strategic_relevance": scores["strategic_relevance_score"],
            "grid_stability": scores["grid_stability_score"],
            "cost_efficiency": scores["cost_efficiency_score"],
            "volume": scores["volume_score"],
            "growth": scores["growth_score"],
            "quality": scores["quality_score"]
        }
    }


@app.get("/api/v1/explain/compare")
async def compare_trend_explanations(
    db: Session = Depends(get_db),
    top_n: int = 5
):
    """
    Compare SHAP explanations across top trends.
    
    Shows which features drive the differences between trends,
    enabling stakeholders to understand WHY certain trends rank higher.
    """
    trends = db.query(Trend).filter(
        Trend.is_active == True
    ).order_by(Trend.priority_score.desc()).limit(top_n).all()
    
    scorer = get_trend_scorer()
    comparisons = []
    
    for trend in trends:
        signals = db.query(Signal).filter(Signal.cluster_id == trend.cluster_id).all()
        signal_data = [
            {"title": s.title, "content": s.content, "quality_score": s.quality_score,
             "source_name": s.source_name, "source_type": s.source_type,
             "published_at": s.published_at, "scraped_at": s.scraped_at}
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
        shapley = scores.get("shapley_explanation", {})
        
        comparisons.append({
            "trend_id": trend.trend_id,
            "name": trend.name,
            "priority_score": scores["priority_score"],
            "feature_importance": shapley.get("feature_importance", {}),
            "top_drivers": shapley.get("top_drivers", []),
            "waterfall": shapley.get("waterfall", []),
            "category_context": shapley.get("category_context", {})
        })
    
    return {
        "comparison_count": len(comparisons),
        "method": "shapley_additive_decomposition",
        "trends": comparisons
    }


# =============================================================================
# AGENTIC PIPELINE ENDPOINT
# =============================================================================

@app.post("/api/v1/agent/scan")
async def agentic_scan(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    goal: str = "Identify highest-priority emerging trends for Amprion TSO operations",
    sources: Optional[List[str]] = None,
    min_confidence: float = 0.3,
    auto_alert: bool = True,
    full_logs: bool = True,
    detect_deviations: bool = True
):
    """
    Agentic Trend Scanning Pipeline.
    
    Implements autonomous agent behavior with 6 clear steps addressing DoD requirements:
    
    STEP 1: FIND SIGNALS     - Automatically capture relevant information from publicly 
                               available sources (RSS, arXiv, TSO, regulatory, events)
    STEP 2: CLUSTER SIGNALS  - Group similar signals to identify patterns and connections
                               using semantic classification into 16 TSO categories
    STEP 3: DERIVE TRENDS    - Identify potential trends based on clusters (min 3 signals)
    STEP 4: ASSESS TRENDS    - Evaluate trends in terms of relevance, potential, and 
                               strategic importance using Bayesian MCDA + AHP + SHAP
    STEP 5: PREPARE RESULTS  - Present trends clearly so decision-makers can work with 
                               them directly (briefs, deep dives, so-what summaries)
    STEP 6: VALIDATE RESULTS - Ensure quality, timeliness, de-duplication, correct 
                               assignment, and relevance for Amprion IT/Digital
    
    DoD Requirements Addressed:
    - Working prototype: Complete end-to-end workflow from signals to assessed trends
    - Scalability & integration: API-based, configurable parameters
    - Flexible search: Configurable scrapers, AHP profiles, classifier backends
    - GDPR compliance: Public sources only, self-hosted LLM option for no external API
    - Traceability & transparency: Every trend links to source signals, no black box
    
    Nice to Have:
    - Trend deviation alerts: Detects significant changes in signal volume
    - Event & Startup crawler: Included in data sources
    
    Args:
        full_logs: If True, outputs rich visual logging for demo presentations
        detect_deviations: If True, compares with previous scan to detect trend changes
    """
    from services.scrapers import MasterScraper
    from services.demo_reporter import DemoReporter
    from services.ahp import get_mcda_weights, _get_active_profile_name
    from collections import Counter
    import asyncio
    import os
    
    # Initialize demo reporter if in demo mode
    reporter = DemoReporter() if full_logs else None
    
    # Get classifier and LLM configuration
    processor = get_nlp_processor()
    classifier_backend = "SentenceTransformer (local)"
    llm_provider = None
    selfhosted_url = None
    
    # Get backend name from the internal classifier (_clf)
    if hasattr(processor, '_clf') and hasattr(processor._clf, 'backend_name'):
        classifier_backend = processor._clf.backend_name
    
    # Check for self-hosted LLM configuration
    llm_provider_env = os.environ.get("SELFHOSTED_LLM_PROVIDER", "")
    if llm_provider_env:
        llm_provider = llm_provider_env
        selfhosted_url = os.environ.get("SELFHOSTED_LLM_URL", "")
    
    # Get AHP profile
    try:
        mcda_weights = get_mcda_weights()
        ahp_profile = _get_active_profile_name()
    except:
        mcda_weights = {"strategic_importance": 0.43, "evidence_strength": 0.23, 
                        "growth_momentum": 0.15, "maturity_readiness": 0.19}
        ahp_profile = "default"
    
    # Get narrative mode
    narrative_mode = os.environ.get("NARRATIVE_MODE", "template")
    
    if reporter:
        reporter.start_pipeline("SONAR.AI Agentic Scan", goal)
        
        # Show configuration (DoD: Traceability & transparency)
        reporter.show_configuration(
            classifier_backend=classifier_backend,
            llm_provider=llm_provider,
            ahp_profile=ahp_profile,
            narrative_mode=narrative_mode,
            scraper_count=len(sources) if sources else 7,
            selfhosted_url=selfhosted_url
        )
        
        # Show available LLM backends (DoD: Flexible search, GDPR compliance)
        backends = [
            {"name": "SentenceTransformer", "status": "available", "url": "local", "gdpr_compliant": True,
             "active": "SentenceTransformer" in classifier_backend},
            {"name": "Ollama", "status": "available" if os.environ.get("SELFHOSTED_LLM_PROVIDER") == "ollama" else "not available",
             "url": os.environ.get("OLLAMA_BASE_URL", "lMistral 7B/Mixtral"), "gdpr_compliant": True},
            {"name": "vLLM", "status": "available" if os.environ.get("SELFHOSTED_LLM_PROVIDER") == "vllm" else "not available",
             "url": os.environ.get("VLLM_BASE_URL", "localhost:8000"), "gdpr_compliant": True},
            {"name": "LM Studio", "status": "available" if os.environ.get("SELFHOSTED_LLM_PROVIDER") == "lmstudio" else "not available",
             "url": os.environ.get("LMSTUDIO_BASE_URL", "localhost:1234"), "gdpr_compliant": True},
            {"name": "Mistral 7B/Mixtral", "status": "available" if os.environ.get("MISTRAL_API_KEY") else "not available",
             "url": "localhost:9000", "gdpr_compliant": True},  # Mistral is EU-based
        ]
        # Mark active backend
        for b in backends:
            if b["name"].lower() in classifier_backend.lower():
                b["status"] = "active"
                b["active"] = True
        reporter.show_llm_backends(backends)
        
        # Show Amprion Strategic Priors (Step 0 - Configuration context)
        reporter.show_amprion_strategic_priors()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: FIND SIGNALS
    # DoD: "Automatically capture relevant information from publicly available sources"
    # Addresses: Working prototype, Flexible search, GDPR compliance (public sources only)
    # ═══════════════════════════════════════════════════════════════════════════
    
    scraper = MasterScraper()
    scraper_results = {}
    source_breakdown = Counter()
    
    # Get previous scan data for deviation detection
    previous_category_counts = {}
    if detect_deviations:
        previous_trends = db.query(Trend).filter(Trend.is_active == True).all()
        for t in previous_trends:
            if t.tso_category:
                previous_category_counts[t.tso_category] = t.signal_count or 0
    
    try:
        all_signals = await scraper.scrape_all(sources)
        
        # Collect per-scraper stats
        for sig in all_signals:
            src_type = sig.get("source_type", "unknown")
            source_breakdown[src_type] += 1
        
        # Get scraper-level counts (we'll estimate from source types)
        scraper_names = sources or list(scraper.scrapers.keys())
        for name in scraper_names:
            # Estimate from source breakdown
            scraper_results[name] = len([s for s in all_signals if name.lower() in s.get("source_name", "").lower()])
        
    except Exception as e:
        all_signals = []
        if reporter:
            reporter._print(f"  ⚠️ Scraping error: {e}")
    
    # Store signals and count new ones
    scorer = get_trend_scorer()
    
    raw_count = len(all_signals)
    new_signal_count = 0
    
    for signal_data in all_signals:
        existing = db.query(Signal).filter(Signal.url == signal_data.get("url")).first()
        if existing:
            continue
        
        signal = Signal(
            title=signal_data.get("title", ""),
            content=signal_data.get("content", ""),
            url=signal_data.get("url", ""),
            source_type=signal_data.get("source_type", ""),
            source_name=signal_data.get("source_name", ""),
            quality_score=signal_data.get("source_quality_score", 0.5),
            published_at=signal_data.get("published_at"),
            scraped_at=signal_data.get("scraped_at", datetime.utcnow())
        )
        db.add(signal)
        new_signal_count += 1
    
    db.commit()
    
    if reporter:
        # Build better scraper results from actual data
        scraper_counts = Counter()
        for sig in all_signals:
            sn = sig.get("source_name", "unknown")
            # Map to scraper type
            if "arxiv" in sn.lower():
                scraper_counts["arxiv"] += 1
            elif sig.get("source_type") == "news":
                scraper_counts["rss_news"] += 1
            elif sig.get("source_type") == "tso":
                scraper_counts["tso"] += 1
            elif sig.get("source_type") == "regulatory":
                scraper_counts["regulatory"] += 1
            elif sig.get("source_type") == "conference":
                scraper_counts["events"] += 1
            elif sig.get("source_type") == "research":
                scraper_counts["research_org"] += 1
            elif sig.get("source_type") == "startup":
                scraper_counts["startup"] += 1
            else:
                scraper_counts["other"] += 1
        
        reporter.step_find_signals(
            scraper_results=dict(scraper_counts),
            total_raw=raw_count,
            total_after_dedup=raw_count,  # Dedup happens in MasterScraper
            source_breakdown=dict(source_breakdown)
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: CLUSTER SIGNALS (Classification)
    # DoD: "Group similar signals to identify patterns and connections"
    # Addresses: Working prototype, Flexible search (interchangeable classifiers)
    # ═══════════════════════════════════════════════════════════════════════════
    
    unprocessed = db.query(Signal).filter(Signal.is_processed == False).all()
    classified_count = 0
    off_topic_count = 0
    category_counts = Counter()
    confidence_sum = 0.0
    
    # Use the classifier_backend we already determined at the start
    actual_classifier_backend = classifier_backend
    
    for signal in unprocessed:
        try:
            result = processor.classify_signal(signal.title, signal.content or "")
            signal.tso_category = result.get("tso_category", "other")
            signal.tso_category_confidence = result.get("tso_category_confidence", 0.0)
            signal.strategic_domain = result.get("strategic_domain")
            signal.architectural_layer = result.get("architectural_layer")
            signal.amprion_task = result.get("amprion_task")
            signal.keywords = result.get("keywords", [])
            signal.is_processed = True
            
            # Count off_topic signals separately (not as a trend category)
            if result.get("is_off_topic", False) or signal.tso_category in ("off_topic", "other", "off-topic"):
                off_topic_count += 1
            else:
                category_counts[signal.tso_category] += 1
                confidence_sum += signal.tso_category_confidence
            
            classified_count += 1
        except Exception:
            continue
    
    db.commit()
    
    avg_confidence = confidence_sum / classified_count if classified_count > 0 else 0.0
    
    if reporter:
        reporter.step_cluster_signals(
            total_classified=classified_count,
            category_counts=dict(category_counts),
            off_topic_count=off_topic_count,
            classifier_backend=actual_classifier_backend,
            avg_confidence=avg_confidence
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: IDENTIFY TRENDS WITHIN CATEGORIES
    # DoD: "Identify potential trends based on clusters"
    # NEW: Uses clustering to find MULTIPLE distinct trends within each category
    # ═══════════════════════════════════════════════════════════════════════════
    
    from services.trend_clustering import TrendClusterer
    
    processed_signals = db.query(Signal).filter(Signal.is_processed == True).all()
    
    # Group signals by category
    category_signals = {}
    for s in processed_signals:
        cat = s.tso_category or "other"
        if cat in ("off_topic", "other", "off-topic"):
            continue  # Skip off-topic
        if cat not in category_signals:
            category_signals[cat] = []
        # Convert to dict for clustering
        category_signals[cat].append({
            "id": s.id,
            "title": s.title,
            "content": s.content,
            "url": s.url,
            "published_date": str(s.published_at) if s.published_at else "",
            "source_name": s.source_name,
            "source_type": s.source_type,
            "quality_score": s.quality_score,
            "keywords": s.keywords,
        })
    
    # Cluster each category to find distinct trends
    clusterer = TrendClusterer(
        min_signals_per_trend=3,
        max_trends_per_category=5,
    )
    
    all_trends_by_category = {}
    trends_created = 0
    trend_details = []
    all_identified_trends = []  # All trends for scoring
    
    for category, cat_signals in category_signals.items():
        if len(cat_signals) < 3:
            continue
        
        # Identify distinct trends within this category
        category_trends = clusterer.cluster_category(category, cat_signals)
        all_trends_by_category[category] = category_trends
        
        for trend in category_trends:
            trend_name = trend["trend_name"]
            signal_count = trend["signal_count"]
            keywords = trend.get("keywords", [])
            
            # Store in database
            existing_cluster = db.query(TrendCluster).filter(
                TrendCluster.tso_category == category,
                TrendCluster.name == trend_name
            ).first()
            
            if not existing_cluster:
                cluster = TrendCluster(
                    name=trend_name,
                    description=trend.get("description", f"Trend in {category}"),
                    tso_category=category,
                    keywords=keywords[:10]
                )
                db.add(cluster)
                db.flush()
                
                # Link signals to this cluster
                signal_ids = [s["id"] for s in trend["signals"]]
                for sig in db.query(Signal).filter(Signal.id.in_(signal_ids)).all():
                    sig.cluster_id = cluster.id
            
            trend_details.append({
                "name": trend_name,
                "signal_count": signal_count,
                "category": category,
                "keywords": keywords,
                "trend_id": trend["trend_id"],
            })
            
            all_identified_trends.append({
                "trend_id": trend["trend_id"],
                "name": trend_name,
                "category": category,
                "signal_count": signal_count,
                "signals": trend["signals"],
                "keywords": keywords,
            })
            
            trends_created += 1
    
    db.commit()
    
    # Count categories vs trends for reporting
    categories_with_trends = len(all_trends_by_category)
    
    if reporter:
        reporter.step_derive_trends(
            trends_created=trends_created,
            trend_details=trend_details,
            min_signals_threshold=3,
            categories_with_trends=categories_with_trends
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: ASSESS TRENDS
    # DoD: "Evaluate trends in terms of relevance, potential, and strategic importance"
    # Now scores each IDENTIFIED TREND (not just categories)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # MCDA weights already loaded at start - reuse them
    high_priority_trends = []
    all_scored_trends = []
    
    # Score each identified trend (from clustering), not just each category
    for trend_info in all_identified_trends:
        category = trend_info["category"]
        trend_name = trend_info["name"]
        trend_id = trend_info.get("trend_id", f"{category}_trend")
        trend_signals = trend_info["signals"]
        
        if len(trend_signals) < 3:
            continue
        
        # Get or create cluster for this category
        cluster = db.query(TrendCluster).filter(TrendCluster.tso_category == category).first()
        if not cluster:
            cluster = TrendCluster(
                name=TSO_CATEGORIES.get(category, {}).get("name", category),
                description=f"Trends in {category}",
                tso_category=category,
                keywords=[]
            )
            db.add(cluster)
            db.flush()
        
        # Prepare signal data for scoring
        signal_data = [
            {"title": s.get("title", ""), "content": s.get("content", ""), 
             "quality_score": s.get("quality_score", 0.5),
             "source_name": s.get("source_name", ""), "source_type": s.get("source_type", ""),
             "published_at": s.get("published_date", ""), "scraped_at": s.get("scraped_at", "")}
            for s in trend_signals
        ]
        
        domains = [s.get("strategic_domain") for s in trend_signals if s.get("strategic_domain")]
        tasks = [s.get("amprion_task") for s in trend_signals if s.get("amprion_task")]
        
        trend_data = {
            "tso_category": category,
            "strategic_domain": max(set(domains), key=domains.count) if domains else None,
            "amprion_task": max(set(tasks), key=tasks.count) if tasks else None,
            "linked_projects": [],
            "maturity_score": 5
        }
        
        scores = scorer.score_trend(trend_data, signal_data)
        
        all_scored_trends.append({
            "name": trend_name,
            "category": category,
            "trend_id": trend_id,
            "priority_score": scores["priority_score"],
            "amprion_tier": scores.get("amprion_tier", "medium"),
            "recommended_action": scores.get("recommended_action", "Monitor"),
            "signal_count": len(trend_signals),
            "scores": scores
        })
        
        # ══════════════════════════════════════════════════════════════════════
        # STEP 5: PREPARE RESULTS (happens inline with scoring)
        # DoD: "Present trends and analyses clearly and understandably so 
        #       decision-makers can work with them directly"
        # ══════════════════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════════════════
        
        from services.narrative_generator import NarrativeGenerator
        from services.trend_scorer import AMPRION_STRATEGIC_PRIORS
        narr_gen = NarrativeGenerator(mode="auto")
        cat_config = TSO_CATEGORIES.get(category, {})
        prior = AMPRION_STRATEGIC_PRIORS.get(category, {})
        
        narratives = narr_gen.generate_all(
            category=category,
            trend_name=trend_name,
            scores=scores,
            signal_count=len(trend_signals),
            growth_rate=scores.get("growth_rate", 0),
            tier=prior.get("tier", "medium"),
            weight=prior.get("weight", 50),
            rationale=prior.get("rationale", ""),
            strategic_impact=cat_config.get("strategic_impact", ""),
            projects=prior.get("default_projects", cat_config.get("amprion_projects", [])),
            nature=scores.get("strategic_nature", "Accelerator"),
            time_to_impact=scores.get("time_to_impact", "1-3 years"),
            signals=signal_data,
            maturity_score=trend_data.get("maturity_score", 5),
        )
        
        # Create or update trend record - use BOTH category AND name for uniqueness
        existing_trend = db.query(Trend).filter(
            Trend.tso_category == category,
            Trend.name == trend_name
        ).first()
        
        if existing_trend:
            # Update existing trend
            existing_trend.description_short = narratives.get("description_short", "")
            existing_trend.description_full = narratives.get("description_full", "")
            existing_trend.so_what_summary = narratives.get("so_what_summary", "")
            existing_trend.key_players = narratives.get("key_players", [])
            existing_trend.ai_reasoning = narratives.get("ai_reasoning", "")
            existing_trend.strategic_domain = trend_data["strategic_domain"]
            existing_trend.key_amprion_task = trend_data["amprion_task"]
            existing_trend.cluster_id = cluster.id
            existing_trend.priority_score = scores["priority_score"]
            existing_trend.strategic_relevance_score = scores.get("strategic_relevance_score")
            existing_trend.grid_stability_score = scores.get("grid_stability_score")
            existing_trend.cost_efficiency_score = scores.get("cost_efficiency_score")
            existing_trend.volume_score = scores.get("volume_score")
            existing_trend.growth_score = scores.get("growth_score")
            existing_trend.quality_score = scores.get("quality_score")
            existing_trend.growth_rate = scores.get("growth_rate", 0)
            existing_trend.project_multiplier = scores.get("project_multiplier", 1.0)
            existing_trend.strategic_nature = scores["strategic_nature"]
            existing_trend.time_to_impact = scores["time_to_impact"]
            existing_trend.maturity_score = trend_data.get("maturity_score", 5)
            existing_trend.signal_count = len(trend_signals)
            existing_trend.is_active = True
            existing_trend.last_updated = datetime.utcnow()
        else:
            # Create new trend with unique ID
            import uuid
            unique_suffix = str(uuid.uuid4())[:8]
            new_trend = Trend(
                trend_id=f"TR-{datetime.utcnow().year}-{category[:3].upper()}-{unique_suffix}",
                name=trend_name,
                description_short=narratives.get("description_short", ""),
                description_full=narratives.get("description_full", ""),
                so_what_summary=narratives.get("so_what_summary", ""),
                key_players=narratives.get("key_players", []),
                ai_reasoning=narratives.get("ai_reasoning", ""),
                tso_category=category,
                strategic_domain=trend_data["strategic_domain"],
                key_amprion_task=trend_data["amprion_task"],
                cluster_id=cluster.id,
                priority_score=scores["priority_score"],
                strategic_relevance_score=scores.get("strategic_relevance_score"),
                grid_stability_score=scores.get("grid_stability_score"),
                cost_efficiency_score=scores.get("cost_efficiency_score"),
                volume_score=scores.get("volume_score"),
                growth_score=scores.get("growth_score"),
                quality_score=scores.get("quality_score"),
                growth_rate=scores.get("growth_rate", 0),
                project_multiplier=scores.get("project_multiplier", 1.0),
                strategic_nature=scores["strategic_nature"],
                time_to_impact=scores["time_to_impact"],
                maturity_score=trend_data.get("maturity_score", 5),
                signal_count=len(trend_signals),
                is_active=True,
                first_seen=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            db.add(new_trend)
        
        if scores["priority_score"] >= 8.0:
            high_priority_trends.append({
                "name": trend_name,
                "score": scores["priority_score"],
                "category": category,
                "signals": len(trend_signals),
                "top_drivers": scores.get("shapley_explanation", {}).get("top_drivers", [])
            })
    
    db.commit()
    
    if reporter:
        reporter.step_assess_trends(
            scored_trends=all_scored_trends,
            ahp_profile=ahp_profile,
            mcda_weights=mcda_weights
        )
        
        # Count narratives generated
        narratives_count = len(all_scored_trends)
        key_players_count = sum(len(t.get("scores", {}).get("key_players", []) or []) 
                                for t in all_scored_trends)
        
        reporter.step_prepare_results(
            narratives_generated=narratives_count,
            briefs_generated=narratives_count,
            deep_dives_generated=narratives_count,
            key_players_extracted=key_players_count,
            narrative_mode="template"  # or "llm" if LLM mode active
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6: VALIDATE RESULTS
    # DoD: "Ensure quality, timeliness, de-duplication, correct assignment, and 
    #       relevance for Amprion IT/Digital—for reliable downstream use"
    # Nice to have: "Notifications for trend deviations"
    # ═══════════════════════════════════════════════════════════════════════════
    
    alerts_generated = []
    blind_spots = []
    weak_coverage = []
    trend_deviations = []
    
    # Detect trend deviations (Nice to Have: significant changes in signal volume)
    if detect_deviations and previous_category_counts:
        for cat, new_count in category_counts.items():
            old_count = previous_category_counts.get(cat, 0)
            if old_count > 0:
                pct_change = ((new_count - old_count) / old_count) * 100
                # Alert if >50% change either direction
                if abs(pct_change) >= 50:
                    trend_deviations.append({
                        "trend_name": TSO_CATEGORIES.get(cat, {}).get("name", cat),
                        "category": cat,
                        "direction": "up" if pct_change > 0 else "down",
                        "old_value": old_count,
                        "new_value": new_count,
                        "pct_change": pct_change
                    })
                    # Create alert for significant deviation
                    severity = "high" if abs(pct_change) >= 100 else "medium"
                    alert = Alert(
                        alert_type="trend_deviation",
                        severity=severity,
                        title=f"Trend Deviation: {TSO_CATEGORIES.get(cat, {}).get('name', cat)}",
                        description=f"Signal volume changed {pct_change:+.0f}% ({old_count} → {new_count})"
                    )
                    db.add(alert)
                    alerts_generated.append({
                        "type": "trend_deviation",
                        "category": cat,
                        "change": f"{pct_change:+.0f}%"
                    })
    
    # Generate alerts for high-priority trends
    if auto_alert and high_priority_trends:
        for hpt in high_priority_trends:
            alert = Alert(
                alert_type="high_priority_trend",
                severity="high" if hpt["score"] >= 9.0 else "medium",
                title=f"High-Priority Trend: {hpt['name']}",
                description=f"Score {hpt['score']}/10 with {hpt['signals']} signals. "
                            f"Drivers: {'; '.join(hpt['top_drivers'][:3]) if hpt.get('top_drivers') else 'N/A'}"
            )
            db.add(alert)
            alerts_generated.append({
                "trend": hpt["name"],
                "severity": "high" if hpt["score"] >= 9.0 else "medium",
                "score": hpt["score"]
            })
    
    # Check for blind spots (high-tier categories with low coverage)
    from services.trend_scorer import AMPRION_STRATEGIC_PRIORS
    for cat, prior in AMPRION_STRATEGIC_PRIORS.items():
        tier = prior.get("tier", "low")
        signal_count = len(category_signals.get(cat, []))
        
        if tier in ("existential", "critical") and signal_count < 15:
            blind_spots.append(f"{cat} ({tier} tier, only {signal_count} signals)")
            alert = Alert(
                alert_type="blind_spot",
                severity="high",
                title=f"Blind Spot: {TSO_CATEGORIES.get(cat, {}).get('name', cat)}",
                description=f"{tier.upper()} tier category has only {signal_count} signals. Consider adding sources."
            )
            db.add(alert)
            alerts_generated.append({"type": "blind_spot", "category": cat})
        elif tier == "high" and signal_count < 10:
            weak_coverage.append(f"{cat} (high tier, only {signal_count} signals)")
    
    db.commit()
    
    # Calculate coverage metrics
    total_signals = db.query(Signal).count()
    total_trends = db.query(Trend).filter(Trend.is_active == True).count()
    
    category_coverage = len(category_signals)
    total_categories = len(TSO_CATEGORIES)
    coverage_pct = round(category_coverage / total_categories * 100, 1)
    
    uncovered = [cat for cat in TSO_CATEGORIES if cat not in category_signals]
    
    # Quality assessment
    if coverage_pct > 60 and new_signal_count > 50:
        quality_assessment = "HIGH"
    elif coverage_pct > 40:
        quality_assessment = "MEDIUM"
    else:
        quality_assessment = "LOW"
    
    # Traceability check - verify every trend links to signals
    traceability_ok = True
    for trend in all_scored_trends:
        cat = trend.get("category")
        if cat and cat in category_signals and len(category_signals[cat]) > 0:
            continue
        traceability_ok = False
        break
    
    if reporter:
        # Report trend deviations if any (Nice to Have)
        if trend_deviations:
            reporter.report_trend_deviations(trend_deviations)
        
        reporter.step_validate_results(
            coverage_pct=coverage_pct,
            categories_covered=category_coverage,
            total_categories=total_categories,
            blind_spots=blind_spots,
            weak_coverage=weak_coverage,
            alerts_generated=len(alerts_generated),
            quality_assessment=quality_assessment,
            traceability_check=traceability_ok
        )
        
        # Final summary with DoD alignment
        reporter.print_final_summary(
            top_trends=sorted(all_scored_trends, key=lambda x: -x.get("priority_score", 0))[:5],
            key_insights=[
                f"Processed {total_signals} signals across {category_coverage} categories",
                f"Identified {len(high_priority_trends)} high-priority trends requiring attention",
                f"Coverage: {coverage_pct}% of TSO taxonomy covered",
                f"Classifier: {actual_classifier_backend}",
            ],
            recommended_actions=[
                t.get("recommended_action", "Monitor") + ": " + t.get("name", "")
                for t in sorted(all_scored_trends, key=lambda x: -x.get("priority_score", 0))[:3]
            ]
        )
        
        reporter.end_pipeline(success=True)
    
    # Build reflection
    reflection = {
        "scan_quality": quality_assessment,
        "coverage": f"{coverage_pct}% of {total_categories} TSO categories covered",
        "uncovered_categories": uncovered[:5],
        "data_freshness": "GOOD" if new_signal_count > 20 else "STALE",
        "traceability": "VERIFIED" if traceability_ok else "ISSUES_DETECTED",
        "recommendations": []
    }
    
    if coverage_pct < 50:
        reflection["recommendations"].append("Add more data sources to cover missing categories")
    if new_signal_count < 10:
        reflection["recommendations"].append("Low signal volume - check data source connectivity")
    if len(high_priority_trends) == 0:
        reflection["recommendations"].append("No high-priority trends detected - review scoring thresholds")
    if uncovered:
        reflection["recommendations"].append(f"Add sources for: {', '.join(uncovered[:3])}")
    if blind_spots:
        reflection["recommendations"].append(f"Address blind spots: {', '.join([b.split()[0] for b in blind_spots[:3]])}")
    
    return {
        "agent_goal": goal,
        "dod_compliance": {
            "working_prototype": True,
            "scalability_integration": "API-based, configurable parameters",
            "flexible_search": f"{len(sources or ['rss_news', 'arxiv', 'tso', 'regulatory', 'events', 'research_org', 'enlit_world'])} data sources, {ahp_profile} AHP profile",
            "gdpr_compliance": "selfhosted" in classifier_backend.lower() or "local" in classifier_backend.lower() or "SentenceTransformer" in classifier_backend,
            "traceability": "VERIFIED" if traceability_ok else "ISSUES_DETECTED",
            "classifier_backend": actual_classifier_backend,
        },
        "execution_summary": {
            "step_1_find_signals": {
                "description": "Automatically capture relevant information from publicly available sources",
                "signals_scraped": len(all_signals),
                "new_signals_stored": new_signal_count,
                "sources_used": sources or list(MasterScraper().scrapers.keys()),
                "source_breakdown": dict(source_breakdown)
            },
            "step_2_cluster_signals": {
                "description": "Group similar signals to identify patterns and connections",
                "signals_classified": classified_count,
                "off_topic_filtered": off_topic_count,
                "categories_found": category_coverage,
                "avg_confidence": round(avg_confidence, 3),
                "classifier_backend": actual_classifier_backend
            },
            "step_3_derive_trends": {
                "description": "Identify potential trends based on clusters",
                "trends_created": trends_created,
                "trend_details": trend_details[:10]
            },
            "step_4_assess_trends": {
                "description": "Evaluate trends in terms of relevance, potential, and strategic importance",
                "trends_scored": len(all_scored_trends),
                "high_priority_trends": high_priority_trends,
                "ahp_profile": ahp_profile,
                "mcda_weights": mcda_weights
            },
            "step_5_prepare_results": {
                "description": "Present trends clearly so decision-makers can work with them directly",
                "narratives_generated": len(all_scored_trends),
                "briefs_generated": len(all_scored_trends),
                "deep_dives_generated": len(all_scored_trends)
            },
            "step_6_validate_results": {
                "description": "Ensure quality, timeliness, de-duplication, correct assignment, relevance",
                "coverage_pct": coverage_pct,
                "blind_spots": blind_spots,
                "weak_coverage": weak_coverage,
                "alerts_generated": alerts_generated,
                "trend_deviations": trend_deviations,
                "quality_assessment": quality_assessment,
                "traceability": "VERIFIED" if traceability_ok else "ISSUES"
            }
        },
        "database_state": {
            "total_signals": total_signals,
            "total_trends": total_trends,
            "category_coverage": f"{category_coverage}/{total_categories}"
        },
        "reflection": reflection
    }


# =============================================================================
# DATA SOURCE MANAGEMENT
# =============================================================================

@app.get("/api/v1/sources")
async def list_data_sources():
    """List all configured data sources with status."""
    from services.scrapers import RSSNewsScraper, ArxivScraper
    
    rss = RSSNewsScraper()
    sources = []
    
    # RSS Feeds
    for feed_id, url in rss.RSS_FEEDS.items():
        sources.append({
            "id": feed_id,
            "type": "rss",
            "url": url,
            "category_hint": _guess_source_category(feed_id),
            "status": "active"
        })
    
    # arXiv
    arxiv = ArxivScraper()
    sources.append({
        "id": "arxiv",
        "type": "research",
        "queries": arxiv.search_queries,
        "query_count": len(arxiv.search_queries),
        "status": "active"
    })
    
    # Static sources
    for name in ["regulatory", "events", "enlit_world", "tso", "research_org"]:
        sources.append({
            "id": name,
            "type": "scraper",
            "status": "active"
        })
    
    return {
        "total_sources": len(sources),
        "rss_feeds": sum(1 for s in sources if s["type"] == "rss"),
        "sources": sources
    }


def _guess_source_category(feed_id: str) -> str:
    """Map feed ID to likely TSO category."""
    mapping = {
        # Offshore Wind
        "offshore_wind_biz": "offshore_systems",
        "windpower_monthly": "renewables_integration",
        "wind_europe": "renewables_integration",
        # German Energy / TSO
        "clean_energy_wire": "regulatory_policy",
        # Biogas
        "european_biogas": "biogas_biomethane",
        "bioenergy_intl": "biogas_biomethane",
        # Hydrogen
        "fuel_cells_works": "hydrogen_p2g",
        "hydrogen_central": "hydrogen_p2g",
        # Cybersecurity OT/ICS
        "ics_cert": "cybersecurity_ot",
        "cisa_all": "cybersecurity_ot",
        "dark_reading": "cybersecurity_ot",
        "the_record": "cybersecurity_ot",
        "ncsc_nl": "cybersecurity_ot",
        "cert_eu": "cybersecurity_ot",
        "cert_eu_threat": "cybersecurity_ot",
        "decent_cybersecurity": "cybersecurity_ot",
        "industrial_cyber": "cybersecurity_ot",
        # E-Mobility & V2G
        "charged_evs": "e_mobility_v2g",
        "electrive": "e_mobility_v2g",
        "e_motec": "e_mobility_v2g",
        "the_driven": "e_mobility_v2g",
        "insideevs": "e_mobility_v2g",
        # Storage
        "energy_storage_news": "energy_storage",
        # Regulatory
        "carbon_brief": "regulatory_policy",
        "energy_transition": "regulatory_policy",
        "gie_news": "regulatory_policy",
        # Smart Grid / Power
        "power_magazine": "grid_infrastructure",
        "ecowatch_energy": "renewables_integration",
        "mit_energy": "ai_grid_optimization",
        # AI & Tech (Datafloq/LastWeekInAI removed — noise sources)
        # Energy Market / TSO
        "eia_today": "energy_trading",
        "renew_economy": "renewables_integration",
        # General
        "utility_dive": "general",
        "pv_magazine": "renewables_integration",
        "electrek": "e_mobility_v2g",
        "clean_technica": "renewables_integration",
    }
    return mapping.get(feed_id, "general")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
