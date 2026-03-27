"""GraphRAG Intelligence – FastAPI Application Entry Point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.graph import router as graph_router
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")

app = FastAPI(
    title="GraphRAG Intelligence",
    description="Microsoft-inspired Knowledge Graph + Vector Hybrid RAG for complex multi-hop reasoning. Combines Neo4j entity graphs with FAISS for unparalleled context understanding.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "GraphRAG Intelligence",
        "version": "1.0.0",
        "description": "Knowledge Graph + Vector Hybrid RAG",
        "docs": "/docs",
        "features": [
            "Neo4j entity graph storage",
            "FAISS vector similarity search",
            "Automated entity & relationship extraction",
            "Community-aware global query mode",
            "Multi-hop reasoning across entity relationships",
            "Hybrid, vector-only, and graph-only query modes",
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
