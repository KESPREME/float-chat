# FloatChat - AI-Powered ARGO Ocean Data Interface

An AI-powered conversational system for ARGO float data discovery and visualization that enables users to query, explore, and visualize oceanographic information using natural language.

## Features

- **Natural Language Queries**: Ask questions like "Show me salinity profiles near the equator in March 2023"
- **ARGO Data Processing**: Ingest and process NetCDF files from ARGO floats
- **Interactive Visualizations**: Geospatial maps, depth-time plots, profile comparisons
- **RAG-Powered AI**: Retrieval-Augmented Generation with Google Gemini
- **Real-time Chat Interface**: Conversational data exploration

## Usage Examples

### Natural Language Queries

```python
# Example queries you can ask:
"Show me temperature profiles from floats in the Arabian Sea"
"What's the average salinity at 500m depth in the last 6 months?"
"Display oxygen data from float 2901623 between January and March 2023"
"Compare temperature trends in the Bay of Bengal vs Arabian Sea"
```

### API Endpoints

- `POST /query` - Process natural language queries
- `GET /floats` - List available ARGO floats
- `POST /upload` - Upload new NetCDF files
- `GET /summary` - Get data summary statistics

## Architecture

```
Data Sources (NetCDF) → Data Pipeline → Databases (PostgreSQL + Vector Store)
                                              ↓
Chat Interface ← RAG System (Gemini + MCP) ← Database Layer
     ↓                    ↓
Dashboard ← Backend API ←┘
```

## Tech Stack

- **Backend**: FastAPI, PostgreSQL, ChromaDB
- **Frontend**: Streamlit with Plotly/Folium visualizations
- **AI/ML**: Google Gemini 1.5 Pro, RAG pipeline, Vector embeddings
- **Data Processing**: NetCDF4, Pandas, Parquet

## Prerequisites

- Python 3.11+ (Python 3.13 recommended)
- PostgreSQL (optional, for full system)
- Google API Key for Gemini

## Quick Start

### Option 1: Demo Version (Recommended for first try)

1. **Install minimal dependencies:**
```bash
pip install streamlit plotly pandas numpy folium
```

2. **Run the demo:**
```bash
streamlit run demo_app.py
```

3. **Access the demo at:** http://localhost:8501

### Option 2: Full System Setup

1. **Clone and navigate to project:**
```bash
git clone <repository-url>
cd float-chat
```

2. **Install dependencies:**
```bash
# For Windows users with Python 3.13, install core packages first:
pip install streamlit fastapi uvicorn pandas numpy plotly folium sqlalchemy psycopg2-binary python-dotenv pydantic httpx requests

# Then install AI packages:
pip install google-generativeai langchain langchain-google-genai chromadb sentence-transformers
```

3. **Set up environment variables:**
```bash
cp .env.example .env
```

Edit `.env` file with your configuration:
```env
# Google Gemini Configuration
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-1.5-pro

# Database (optional for demo)
DATABASE_URL=postgresql://username:password@localhost:5432/floatchat

# Application Settings
APP_HOST=localhost
APP_PORT=8000
STREAMLIT_PORT=8501
DEBUG=True
```

4. **Get Google API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

5. **Initialize system (optional - for full database setup):**
```bash
python scripts/init_db.py
python scripts/sample_data_loader.py  # Load synthetic test data
```

6. **Run the application:**

**Option A: Full system with API backend**
```bash
# Terminal 1: Start API server
python run_api.py

# Terminal 2: Start Streamlit frontend
python run_streamlit.py
```

**Option B: Streamlit-only version**
```bash
streamlit run app/main.py
```

## Access Points

- **Streamlit Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs (if running full system)
- **API Health Check**: http://localhost:8000/health

## Data Sources

- [Argo Global Data Repository](ftp://ftp.ifremer.fr/ifremer/argo)
- [Indian Argo Project](https://incois.gov.in/OON/index.jsp)

## Project Structure

```
float-chat/
├── app/                    # Main application
│   ├── main.py            # Streamlit main app
│   ├── chat/              # Chat interface components
│   ├── dashboard/         # Visualization components
│   └── api/               # Backend API
├── data/                  # Data processing pipeline
│   ├── ingestion/         # NetCDF ingestion
│   ├── processing/        # Data transformation
│   └── storage/           # Database operations
├── rag/                   # RAG system components
│   ├── llm/               # LLM integration
│   ├── retrieval/         # Vector search
│   └── mcp/               # Model Context Protocol
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## License

MIT License
