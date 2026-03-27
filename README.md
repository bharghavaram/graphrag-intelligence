# GraphRAG Intelligence

> Microsoft-inspired Knowledge Graph + Vector Hybrid RAG for complex multi-hop reasoning

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-orange)](https://neo4j.com)
[![FAISS](https://img.shields.io/badge/FAISS-1.8-red)](https://faiss.ai)

## Overview

GraphRAG Intelligence implements Microsoft's GraphRAG research (2024) — combining **knowledge graph entity relationships** with **vector similarity search** for unprecedented context understanding and multi-hop reasoning capability.

Traditional RAG systems retrieve semantically similar chunks independently. GraphRAG maps **how entities relate to each other**, enabling the system to answer complex questions that require connecting multiple concepts across an entire corpus.

## Architecture

```
Documents → Entity Extraction (GPT-4o) → Neo4j Knowledge Graph
     ↓                                          ↓
Text Chunks → Embeddings → FAISS Index    Community Summaries
     ↓                          ↓               ↓
   Query → Vector Search + Graph Traversal → Hybrid Answer
```

## Key Features

- **Neo4j Knowledge Graph** – stores entities (PERSON, ORG, CONCEPT, TECHNOLOGY) with typed relationships
- **FAISS Vector Index** – semantic similarity search over document chunks
- **Community Detection** – groups related entities for global context awareness
- **3 Query Modes** – `hybrid` (graph + vector), `vector_only`, `graph_only`
- **Automated Entity Extraction** – GPT-4o extracts entities and relationships from every chunk
- **NetworkX Fallback** – works without Neo4j using in-memory graph

## Quick Start

```bash
git clone https://github.com/bharghavaram/graphrag-intelligence
cd graphrag-intelligence
pip install -r requirements.txt
cp .env.example .env    # Add your API keys
uvicorn main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/graphrag/ingest` | Upload documents (PDF/TXT) for indexing |
| POST | `/api/v1/graphrag/query` | Query with hybrid/vector/graph mode |
| GET | `/api/v1/graphrag/stats` | Graph statistics |
| GET | `/api/v1/graphrag/health` | Health check |

### Example: Ingest Documents

```bash
curl -X POST "http://localhost:8000/api/v1/graphrag/ingest" \
  -F "files=@research_paper.pdf"
```

### Example: Hybrid Query

```bash
curl -X POST "http://localhost:8000/api/v1/graphrag/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does GraphRAG improve over naive RAG?", "mode": "hybrid"}'
```

## Docker

```bash
# Start Neo4j + GraphRAG
docker-compose up --build
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `LLM_MODEL` | LLM for reasoning | `gpt-4o` |
| `EMBED_MODEL` | Embedding model | `text-embedding-3-small` |

## Why GraphRAG Stands Out

- Standard RAG: "What is machine learning?" → retrieves 5 similar chunks
- GraphRAG: "How does Transformer architecture relate to BERT's training methodology and its impact on NLP benchmarks?" → traverses entity relationships across the entire corpus
