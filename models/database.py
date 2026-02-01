"""
SONAR.AI / AGTS Database Models
================================
Based on AGTS Framework Pages 13-15 Trend List Fields
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, 
    ForeignKey, JSON, Enum, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import enum

from config import DATABASE_URL

Base = declarative_base()

# =============================================================================
# ENUMS
# =============================================================================

class LifecycleStatus(str, enum.Enum):
    SCOUTING = "Scouting"
    PILOT = "Pilot"
    IMPLEMENTATION = "Implementation"
    STANDARD = "Standard"

class MaturityType(str, enum.Enum):
    TRL = "TRL"  # Technology Readiness Level
    MRL = "MRL"  # Market Readiness Level
    SRL = "SRL"  # Societal Readiness Level
    RRL = "RRL"  # Regulatory Readiness Level

class TimeToImpact(str, enum.Enum):
    IMMEDIATE = "<1 year"
    SHORT_TERM = "1-3 years"
    MEDIUM_TERM = "3-5 years"
    LONG_TERM = "5+ years"

class StrategicNature(str, enum.Enum):
    ACCELERATOR = "Accelerator"
    DISRUPTOR = "Disruptor"
    TRANSFORMATIONAL = "Transformational"

class RegulatorySensitivity(str, enum.Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class SourceType(str, enum.Enum):
    RESEARCH = "research"
    PATENT = "patent"
    NEWS = "news"
    REGULATORY = "regulatory"
    TSO = "tso"
    STARTUP = "startup"
    CONFERENCE = "conference"
    RESEARCH_ORG = "research_org"

# =============================================================================
# SIGNAL MODEL (Raw Data from Sources)
# =============================================================================

class Signal(Base):
    """Raw signals captured from data sources (Scout Agent output)."""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Core Identity
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    url = Column(String(1000), nullable=True)
    
    # Source Information
    source_type = Column(String(50), nullable=True)  # research, patent, news, etc.
    source_name = Column(String(200), nullable=True)
    source_quality_score = Column(Float, default=0.5)
    
    # Temporal
    published_at = Column(DateTime, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    # NLP Processing Results
    entities = Column(JSON, nullable=True)  # Named entities
    keywords = Column(JSON, nullable=True)  # Extracted keywords
    
    # Classification (from config.py TSO_CATEGORIES)
    tso_category = Column(String(100), nullable=True)
    strategic_domain = Column(String(100), nullable=True)
    architectural_layer = Column(String(100), nullable=True)
    
    # Amprion-Specific Classification
    amprion_task = Column(String(100), nullable=True)  # Grid Expansion, System Security, etc.
    amprion_business_area = Column(String(50), nullable=True)  # SO, AM, ITD, etc.
    linked_projects = Column(JSON, nullable=True)  # Related Amprion projects
    
    # Embedding & Clustering
    embedding_id = Column(String(100), nullable=True)
    cluster_id = Column(Integer, ForeignKey("trend_clusters.id"), nullable=True)
    
    # Quality & Processing
    language = Column(String(10), default="en")
    quality_score = Column(Float, default=0.5)
    is_processed = Column(Boolean, default=False)
    is_duplicate = Column(Boolean, default=False)
    
    # Relationships
    cluster = relationship("TrendCluster", back_populates="signals")

# =============================================================================
# TREND CLUSTER MODEL (Architect Agent output)
# =============================================================================

class TrendCluster(Base):
    """Clusters of related signals grouped semantically."""
    __tablename__ = "trend_clusters"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Identity
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Classification
    tso_category = Column(String(100), nullable=True)
    strategic_domain = Column(String(100), nullable=True)
    
    # Keywords & Representatives
    keywords = Column(JSON, nullable=True)  # Top cluster keywords
    representative_signals = Column(JSON, nullable=True)  # Top signal IDs
    
    # Temporal
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    signals = relationship("Signal", back_populates="cluster")
    trends = relationship("Trend", back_populates="cluster")

# =============================================================================
# TREND MODEL (Full AGTS Trend Profile - Pages 13-15)
# =============================================================================

class Trend(Base):
    """
    Complete Trend Profile as per AGTS specification.
    Implements all fields from pages 13-15 of the framework.
    """
    __tablename__ = "trends"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # =========================================================================
    # SECTION 1: CORE IDENTITY & CLASSIFICATION (Page 13)
    # =========================================================================
    
    # Trend ID & Name
    trend_id = Column(String(50), unique=True, index=True)  # e.g., "TR-2026-001"
    name = Column(String(300), nullable=False)  # Descriptive, non-technical title
    
    # Executive Summary (The "Brief") - 250 chars
    description_short = Column(String(500), nullable=True)
    
    # Contextual Analysis (The "Deep Dive") - up to 2000 chars
    description_full = Column(Text, nullable=True)
    
    # Lifecycle Status
    lifecycle_status = Column(String(50), default="Scouting")  # Scouting/Pilot/Implementation/Standard
    
    # Maturity Framework
    maturity_type = Column(String(10), default="TRL")  # TRL/MRL/SRL/RRL
    maturity_score = Column(Integer, default=1)  # 1-9 scale
    
    # Time-to-Impact
    time_to_impact = Column(String(50), default="1-3 years")  # <1 year, 1-3 years, 3-5 years, 5+ years
    
    # =========================================================================
    # SECTION 2: STRATEGIC PRIORITIZATION (Page 14)
    # =========================================================================
    
    # Priority Score (1.0 - 10.0)
    priority_score = Column(Float, default=5.0)
    
    # Strategic Nature
    strategic_nature = Column(String(50), default="Accelerator")  # Accelerator/Disruptor/Transformational
    
    # Key Amprion Task (Fact Book)
    key_amprion_task = Column(String(100), nullable=True)  # Grid Expansion, System Security, etc.
    
    # Amprion Field of Action
    field_of_action = Column(String(100), nullable=True)  # Robust Planning, Grid Operation Evolution, etc.
    
    # =========================================================================
    # SECTION 3: IMPACT & BUSINESS CONTEXT (Page 14)
    # =========================================================================
    
    # Impacted Business Area
    business_area = Column(String(50), nullable=True)  # SO, AM, ITD, GP, M, CS
    business_area_name = Column(String(200), nullable=True)  # Full name
    
    # Linked Internal Projects (JSON array)
    linked_projects = Column(JSON, nullable=True)  # ["Korridor B", "Rhein-Main Link"]
    
    # Regulatory Sensitivity
    regulatory_sensitivity = Column(String(20), default="Medium")  # High/Medium/Low
    
    # Sovereignty Score (0-100)
    sovereignty_score = Column(Float, default=50.0)
    
    # =========================================================================
    # SECTION 4: TECHNICAL & MARKET INTELLIGENCE (Page 14)
    # =========================================================================
    
    # Architectural Layer
    architectural_layer = Column(String(100), nullable=True)  # Perception/Connectivity/Data/Service
    
    # Customer/Market Delta
    customer_impact = Column(Text, nullable=True)  # Does it change industrial demand?
    market_impact = Column(Text, nullable=True)  # Does it alter price formation?
    
    # "So What?" Summary (2-sentence executive summary)
    so_what_summary = Column(Text, nullable=True)
    
    # =========================================================================
    # SECTION 5: SCORING COMPONENTS (Page 22)
    # =========================================================================
    
    # No-Regret Framework Scores (0-100)
    strategic_relevance_score = Column(Float, default=50.0)  # 40% weight
    grid_stability_score = Column(Float, default=50.0)       # 30% weight
    cost_efficiency_score = Column(Float, default=50.0)      # 30% weight
    
    # Project Multiplier
    project_multiplier = Column(Float, default=1.0)
    
    # Component Scores (for transparency)
    volume_score = Column(Float, default=0.0)  # Number of signals
    growth_score = Column(Float, default=0.0)  # Week-over-week growth
    quality_score = Column(Float, default=0.0)  # Source quality average
    
    # =========================================================================
    # SECTION 6: ADDITIONAL METADATA
    # =========================================================================
    
    # TSO Classification
    tso_category = Column(String(100), nullable=True)
    strategic_domain = Column(String(100), nullable=True)
    
    # Signal Statistics
    signal_count = Column(Integer, default=0)
    signal_count_7d = Column(Integer, default=0)  # Last 7 days
    signal_count_30d = Column(Integer, default=0)  # Last 30 days
    growth_rate = Column(Float, default=0.0)  # Percentage
    
    # AI-Generated Content
    ai_summary = Column(Text, nullable=True)
    ai_reasoning = Column(Text, nullable=True)  # Agentic Reasoning Log
    strategic_implications = Column(Text, nullable=True)
    
    # Keywords & Entities
    keywords = Column(JSON, nullable=True)
    key_players = Column(JSON, nullable=True)  # Companies, organizations
    
    # Status
    is_active = Column(Boolean, default=True)
    is_emerging = Column(Boolean, default=True)
    is_validated = Column(Boolean, default=False)  # Human validated
    
    # Temporal
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Cluster Relationship
    cluster_id = Column(Integer, ForeignKey("trend_clusters.id"), nullable=True)
    cluster = relationship("TrendCluster", back_populates="trends")

# =============================================================================
# ALERT MODEL
# =============================================================================

class Alert(Base):
    """Alerts for disruptive events and high-priority trends."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Alert Info
    title = Column(String(300), nullable=False)
    description = Column(Text, nullable=True)
    alert_type = Column(String(50), default="trend")  # trend, signal, threshold
    severity = Column(String(20), default="medium")  # critical, high, medium, low
    
    # Related Entities
    trend_id = Column(Integer, ForeignKey("trends.id"), nullable=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)
    
    # Target Audience
    target_business_area = Column(String(50), nullable=True)
    target_stakeholder = Column(String(100), nullable=True)
    
    # Status
    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)
    
    # Temporal
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime, nullable=True)

# =============================================================================
# DATA SOURCE MODEL
# =============================================================================

class DataSource(Base):
    """Track data source status and statistics."""
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    
    source_id = Column(String(100), unique=True, index=True)
    name = Column(String(200), nullable=False)
    source_type = Column(String(50), nullable=True)
    url = Column(String(500), nullable=True)
    
    # Quality & Status
    quality_score = Column(Float, default=0.5)
    is_active = Column(Boolean, default=True)
    
    # Statistics
    total_signals = Column(Integer, default=0)
    last_scraped_at = Column(DateTime, nullable=True)
    last_signal_at = Column(DateTime, nullable=True)
    
    # Configuration
    scrape_interval_hours = Column(Integer, default=24)
    config = Column(JSON, nullable=True)

# =============================================================================
# PROCESSING JOB MODEL
# =============================================================================

class ProcessingJob(Base):
    """Track background processing jobs."""
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    job_type = Column(String(50), nullable=False)  # scrape, cluster, score, etc.
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    
    # Progress
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    
    # Results
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Temporal
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

# =============================================================================
# USER WORKSPACE MODEL (For Stakeholder Views)
# =============================================================================

class UserWorkspace(Base):
    """Save user workspace settings and filters."""
    __tablename__ = "user_workspaces"
    
    id = Column(Integer, primary_key=True, index=True)
    
    user_id = Column(String(100), index=True)
    workspace_name = Column(String(200), nullable=False)
    stakeholder_type = Column(String(50), nullable=True)  # executive_board, it_architecture, etc.
    
    # Saved Filters
    filters = Column(JSON, nullable=True)
    sort_by = Column(String(100), nullable=True)
    visible_columns = Column(JSON, nullable=True)
    
    # Status
    is_default = Column(Boolean, default=False)
    
    # Temporal
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# =============================================================================
# CUSTOM TAG MODEL
# =============================================================================

class CustomTag(Base):
    """User-defined tags for signals and trends."""
    __tablename__ = "custom_tags"
    
    id = Column(Integer, primary_key=True, index=True)
    
    name = Column(String(100), nullable=False)  # e.g., "#KorridorB", "#RheinMainLink"
    description = Column(String(500), nullable=True)
    color = Column(String(20), default="#3B82F6")
    
    # Associated Entities
    signal_ids = Column(JSON, nullable=True)
    trend_ids = Column(JSON, nullable=True)
    
    # Creator
    created_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

# Create engine and session
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully")

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize on import
if __name__ == "__main__":
    init_db()
