"""GraphRAG Intelligence – API routes."""
import shutil, tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from app.services.graphrag_service import GraphRAGService, get_graphrag_service

router = APIRouter(prefix="/graphrag", tags=["GraphRAG"])

class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"  # hybrid | vector_only | graph_only

@router.post("/query")
async def query(req: QueryRequest, svc: GraphRAGService = Depends(get_graphrag_service)):
    if req.mode not in ("hybrid", "vector_only", "graph_only"):
        raise HTTPException(400, "mode must be hybrid, vector_only, or graph_only")
    return svc.query(req.question, req.mode)

@router.post("/ingest")
async def ingest(files: List[UploadFile] = File(...), svc: GraphRAGService = Depends(get_graphrag_service)):
    tmp = tempfile.mkdtemp()
    try:
        paths = []
        for f in files:
            dest = Path(tmp) / f.filename
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            paths.append(str(dest))
        result = svc.ingest_documents(paths)
        result["files"] = [f.filename for f in files]
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

@router.get("/stats")
async def stats(svc: GraphRAGService = Depends(get_graphrag_service)):
    return svc.get_graph_stats()

@router.get("/health")
async def health():
    return {"status": "ok", "service": "GraphRAG Intelligence – Knowledge Graph + Vector Hybrid RAG"}
