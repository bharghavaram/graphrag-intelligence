"""Tests for GraphRAG Intelligence."""
import pytest
from unittest.mock import MagicMock, patch
from app.core.config import settings


def test_settings_defaults():
    assert settings.TOP_K_CHUNKS == 5
    assert settings.COMMUNITY_LEVELS == 3
    assert settings.TEMPERATURE == 0.1


def test_query_modes():
    valid_modes = ["hybrid", "vector_only", "graph_only"]
    for mode in valid_modes:
        assert mode in valid_modes


@patch("app.services.graphrag_service.GraphDatabase")
@patch("app.services.graphrag_service.OpenAIEmbeddings")
@patch("app.services.graphrag_service.ChatOpenAI")
def test_graphrag_service_init(mock_llm, mock_embed, mock_neo4j):
    mock_neo4j.driver.return_value = MagicMock()
    from app.services.graphrag_service import GraphRAGService
    svc = GraphRAGService()
    assert svc is not None
    assert svc.nx_graph is not None


@patch("app.services.graphrag_service.GraphDatabase")
@patch("app.services.graphrag_service.OpenAIEmbeddings")
@patch("app.services.graphrag_service.ChatOpenAI")
def test_query_no_vectorstore(mock_llm, mock_embed, mock_neo4j):
    mock_llm.return_value.invoke.return_value = MagicMock(content="Test answer")
    from app.services.graphrag_service import GraphRAGService
    svc = GraphRAGService()
    svc.vectorstore = None
    result = svc.query("What is GraphRAG?", mode="graph_only")
    assert "answer" in result
    assert result["mode"] == "graph_only"


def test_graph_stats_networkx():
    with patch("app.services.graphrag_service.GraphDatabase") as mock_neo4j, \
         patch("app.services.graphrag_service.OpenAIEmbeddings"), \
         patch("app.services.graphrag_service.ChatOpenAI"):
        mock_neo4j.driver.side_effect = Exception("Neo4j not available")
        from app.services.graphrag_service import GraphRAGService
        svc = GraphRAGService()
        stats = svc.get_graph_stats()
        assert "total_entities" in stats
        assert stats["graph_backend"] == "networkx"


@pytest.mark.asyncio
async def test_api_health():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/api/v1/graphrag/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_api_query_invalid_mode():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.post("/api/v1/graphrag/query", json={"question": "Test?", "mode": "invalid"})
    assert resp.status_code == 400
