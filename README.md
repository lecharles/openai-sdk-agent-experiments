# OpenAI SDK Agent Experiments

This repository contains various experiments with the OpenAI Assistants API and other AI agent implementations.

## Agents

### RAG Agent
Located in `rag_agent/`
- Vector search-based research paper retrieval system
- Uses FAISS for efficient similarity search
- Built with sentence-transformers (multilingual-e5-large-instruct)
- Components:
  - `vector_search_agent.py`: Handles vector similarity search
  - `document_indexer.py`: Processes and indexes documents
  - `test_rag.py`: Interactive search interface

### PDF Extractor Agent
Located in `agent_pdf_extractor_vibe.py`
- Extracts and processes information from PDF documents
- Focuses on research paper analysis

### Multi-Agent System
Located in `multi_agent_system.py`
- Demonstrates agent collaboration
- Multiple specialized agents working together

### Structured Data Extraction
Located in `structured_data_extract.py`
- Extracts structured data from text
- Pattern recognition and data formatting

### Example Agent
Located in `example_agent.py`
- Basic agent implementation
- Demonstrates core concepts

### Agent Tools
Located in `agent_tools.py` and `agent_as_tools.py`
- Utility functions for agents
- Tool implementations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For the RAG agent:
```bash
cd rag_agent
python test_rag.py
```

## Documentation

- `read-about-other-agents.md`: Research on different agent implementations
- `papers_output.md`: Example outputs from paper analysis 