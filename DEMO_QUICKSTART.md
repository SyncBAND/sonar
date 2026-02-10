# SONAR.AI Demo Quickstart

## Running the Demo

### 1. Start the Server
```bash
cd /home/claude/sonar_full
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Trigger Agentic Scan with Visual Demo Mode
```bash
# Full demo with visual logging
curl -X POST "http://localhost:8000/api/v1/agent/scan?full_logs=true&detect_deviations=true"

# With custom goal
curl -X POST "http://localhost:8000/api/v1/agent/scan?full_logs=true&goal=Find%20cybersecurity%20threats%20for%20TSO%20infrastructure"
```

## DoD Requirements Mapping

| Requirement | Implementation | Visible in Demo |
|------------|----------------|-----------------|
| **Working prototype** | Complete 6-step workflow | ✅ Steps 1-6 visually logged |
| **Scalability & integration** | FastAPI endpoints, configurable | ✅ Configuration display |
| **Flexible search** | 7 scrapers, 5 AHP profiles, interchangeable classifiers | ✅ Config + LLM backends |
| **GDPR compliance** | Public sources, self-hosted LLM option | ✅ [GDPR ✓] badge |
| **Traceability** | Every trend links to signals | ✅ "VERIFIED" status |
| **Trend deviation alerts** | Detects ±50% signal volume changes | ✅ 🔔 DEVIATION ALERTS |
| **Event & Startup crawler** | Included in scrapers | ✅ Source breakdown |

## Using Self-Hosted LLM (No API Keys Needed)

### Option 1: Ollama (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:8b

# Set environment
export SELFHOSTED_LLM_PROVIDER=ollama
export SELFHOSTED_LLM_MODEL=llama3.2:8b
export CLASSIFIER_PROVIDER=selfhosted
```

### Option 2: LM Studio
```bash
# Download from https://lmstudio.ai
# Start server on port 1234

export SELFHOSTED_LLM_PROVIDER=lmstudio
export LMSTUDIO_BASE_URL=http://localhost:1234
export CLASSIFIER_PROVIDER=selfhosted
```

### Option 3: vLLM
```bash
pip install vllm
vllm serve meta-llama/Llama-3.2-8B-Instruct --port 8000

export SELFHOSTED_LLM_PROVIDER=vllm
export VLLM_BASE_URL=http://localhost:8000
export CLASSIFIER_PROVIDER=selfhosted
```

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/agent/scan` | POST | **Main demo endpoint** - runs full pipeline |
| `/api/v1/dashboard` | GET | Summary stats + top trends |
| `/api/v1/trends` | GET | All trends with filters |
| `/api/v1/trends/{id}` | GET | Single trend with signals |
| `/api/v1/explain/trend/{id}` | GET | SHAP waterfall explanation |
| `/api/v1/explain/compare` | GET | Side-by-side trend comparison |
| `/api/v1/alerts` | GET | Deviation alerts |
| `/api/v1/ahp/profiles` | GET | Available AHP profiles |

## Demo Output Sections

The visual demo shows:
1. **Configuration** - Classifier, AHP profile, narrative mode, scrapers
2. **LLM Backends** - Available backends with GDPR status
3. **Step 1: Find Signals** - Sources scraped, breakdown, deduplication
4. **Step 2: Cluster Signals** - Category distribution, classifier used
5. **Step 3: Derive Trends** - Trends identified, signal counts
6. **Step 4: Assess Trends** - Rankings, MCDA weights, tiers
7. **Step 5: Prepare Results** - Narratives generated
8. **Trend Deviations** - Volume changes since last scan
9. **Step 6: Validate Results** - Coverage, blind spots, traceability
10. **Executive Summary** - Top trends, insights, actions
