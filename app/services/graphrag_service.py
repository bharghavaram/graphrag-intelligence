"""
GraphRAG Intelligence – Microsoft-inspired Knowledge Graph + Vector Hybrid RAG.
Combines Neo4j entity graphs with FAISS for multi-hop reasoning.
"""
import logging
import hashlib
from pathlib import Path
from typing import Optional, List
import networkx as nx
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from app.core.config import settings

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """Extract all named entities and their relationships from this text.
Return ONLY valid JSON:
{{
  "entities": [
    {{"name": "...", "type": "PERSON|ORG|CONCEPT|TECHNOLOGY|METRIC|EVENT", "description": "..."}}
  ],
  "relationships": [
    {{"source": "entity_name", "target": "entity_name", "relation": "VERB_PHRASE", "weight": 0.0-1.0}}
  ]
}}

Text:
{text}"""

GLOBAL_QUERY_PROMPT = """You are an expert analyst with access to a knowledge graph and document corpus.

COMMUNITY SUMMARIES (graph-derived high-level context):
{community_context}

RELEVANT DOCUMENT CHUNKS (vector search results):
{chunk_context}

ENTITY RELATIONSHIPS (graph traversal):
{graph_context}

Question: {question}

Provide a comprehensive, well-structured answer using all three knowledge sources.
Cite evidence from both the graph entities and document chunks.
Use multi-hop reasoning when the question requires connecting multiple concepts."""


class GraphRAGService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBED_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self.vectorstore: Optional[FAISS] = None
        self.nx_graph = nx.DiGraph()
        self.neo4j_driver = None
        self._init_neo4j()
        self._load_faiss()

    def _init_neo4j(self):
        try:
            self.neo4j_driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j connected at %s", settings.NEO4J_URI)
        except Exception as exc:
            logger.warning("Neo4j unavailable (%s) – using in-memory NetworkX graph", exc)
            self.neo4j_driver = None

    def _load_faiss(self):
        index_path = Path(settings.FAISS_INDEX_PATH)
        if index_path.exists():
            self.vectorstore = FAISS.load_local(
                str(index_path), self.embeddings, allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded")

    def _store_entity_neo4j(self, name: str, etype: str, description: str):
        if not self.neo4j_driver:
            self.nx_graph.add_node(name, type=etype, description=description)
            return
        with self.neo4j_driver.session() as session:
            session.run(
                "MERGE (e:Entity {name: $name}) SET e.type = $type, e.description = $desc",
                name=name, type=etype, desc=description
            )

    def _store_relationship_neo4j(self, source: str, target: str, relation: str, weight: float):
        if not self.neo4j_driver:
            self.nx_graph.add_edge(source, target, relation=relation, weight=weight)
            return
        with self.neo4j_driver.session() as session:
            session.run(
                """MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                   MERGE (a)-[r:RELATES {type: $rel}]->(b)
                   SET r.weight = $weight""",
                src=source, tgt=target, rel=relation, weight=weight
            )

    def _extract_entities(self, text: str) -> dict:
        import json
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:3000])
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            return json.loads(response.content)
        except Exception:
            return {"entities": [], "relationships": []}

    def ingest_documents(self, file_paths: List[str]) -> dict:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        all_chunks = []
        total_entities = 0
        total_relationships = 0

        for fp in file_paths:
            try:
                loader = PyPDFLoader(fp) if fp.endswith(".pdf") else TextLoader(fp)
                docs = loader.load()
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)

                # Entity extraction for each chunk
                for chunk in chunks:
                    extracted = self._extract_entities(chunk.page_content)
                    for ent in extracted.get("entities", []):
                        self._store_entity_neo4j(ent["name"], ent.get("type", "CONCEPT"), ent.get("description", ""))
                        total_entities += 1
                    for rel in extracted.get("relationships", []):
                        self._store_relationship_neo4j(
                            rel["source"], rel["target"], rel["relation"], rel.get("weight", 0.5)
                        )
                        total_relationships += 1
            except Exception as exc:
                logger.error("Error processing %s: %s", fp, exc)

        # Build / update FAISS index
        if all_chunks:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            else:
                self.vectorstore.add_documents(all_chunks)
            index_path = Path(settings.FAISS_INDEX_PATH)
            index_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(index_path))

        return {
            "chunks_indexed": len(all_chunks),
            "entities_extracted": total_entities,
            "relationships_extracted": total_relationships,
        }

    def _get_community_summaries(self) -> str:
        if self.neo4j_driver:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.description AS desc LIMIT 20"
                )
                records = [f"[{r['type']}] {r['name']}: {r['desc']}" for r in result]
            return "\n".join(records) if records else "No community data available."
        else:
            nodes = list(self.nx_graph.nodes(data=True))[:20]
            return "\n".join(
                [f"[{d.get('type','?')}] {n}: {d.get('description','')}" for n, d in nodes]
            ) or "No community data."

    def _get_graph_context(self, query: str) -> str:
        keywords = query.lower().split()
        relevant_edges = []
        if self.neo4j_driver:
            with self.neo4j_driver.session() as session:
                for kw in keywords[:3]:
                    result = session.run(
                        """MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                           WHERE toLower(a.name) CONTAINS $kw OR toLower(b.name) CONTAINS $kw
                           RETURN a.name, r.type, b.name LIMIT 10""",
                        kw=kw
                    )
                    for rec in result:
                        relevant_edges.append(f"{rec['a.name']} --[{rec['r.type']}]--> {rec['b.name']}")
        else:
            for u, v, data in self.nx_graph.edges(data=True):
                if any(kw in u.lower() or kw in v.lower() for kw in keywords[:3]):
                    relevant_edges.append(f"{u} --[{data.get('relation','?')}]--> {v}")
        return "\n".join(relevant_edges[:15]) or "No graph relationships found."

    def query(self, question: str, mode: str = "hybrid") -> dict:
        chunk_context = "No vector index available."
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(question, k=settings.TOP_K_CHUNKS)
            chunk_context = "\n\n---\n\n".join([d.page_content for d in docs])

        community_context = self._get_community_summaries()
        graph_context = self._get_graph_context(question)

        if mode == "vector_only":
            prompt_text = f"Answer based on these documents:\n{chunk_context}\n\nQuestion: {question}"
        elif mode == "graph_only":
            prompt_text = f"Answer based on the knowledge graph:\nEntities/Communities:\n{community_context}\nRelationships:\n{graph_context}\n\nQuestion: {question}"
        else:
            prompt_text = GLOBAL_QUERY_PROMPT.format(
                community_context=community_context,
                chunk_context=chunk_context,
                graph_context=graph_context,
                question=question,
            )

        response = self.llm.invoke([HumanMessage(content=prompt_text)])
        return {
            "answer": response.content,
            "mode": mode,
            "graph_entities_used": len(community_context.split("\n")),
            "chunks_used": settings.TOP_K_CHUNKS if self.vectorstore else 0,
            "graph_relationships": graph_context.count("-->"),
        }

    def get_graph_stats(self) -> dict:
        if self.neo4j_driver:
            with self.neo4j_driver.session() as session:
                nodes = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
                rels = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS count").single()["count"]
        else:
            nodes = self.nx_graph.number_of_nodes()
            rels = self.nx_graph.number_of_edges()
        return {
            "total_entities": nodes,
            "total_relationships": rels,
            "vector_chunks": self.vectorstore.index.ntotal if self.vectorstore else 0,
            "graph_backend": "neo4j" if self.neo4j_driver else "networkx",
        }


_service: Optional[GraphRAGService] = None
def get_graphrag_service() -> GraphRAGService:
    global _service
    if _service is None:
        _service = GraphRAGService()
    return _service
