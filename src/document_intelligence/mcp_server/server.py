"""Serveur MCP (Model Context Protocol) : recherche documentaire et agent de données.

Outils adossés aux fonctions de `document_intelligence.tools` :
- `search_documents(query, k)` : chunks pertinents, sans appel LLM ;
- `ask_question(question, k)` : réponse du backend RAG configuré, avec ses sources ;
- `list_documents()` : documents indexés ;
- `describe_tables()`, `run_sql(sql)` : base de l'agent de données, sans appel
  LLM (exécution en lecture seule derrière les garde-fous SQL) ;
- `ask_data(question)` : agent text-to-SQL (réponse citant la requête exécutée).

Construit avec le SDK MCP officiel (`mcp.server.fastmcp`). Les outils sont
asynchrones et délèguent le travail bloquant (recherche, appel LLM) à un
thread, pour ne pas bloquer la boucle d'événements du serveur.
"""

from anyio import to_thread
from mcp.server.fastmcp import FastMCP

from document_intelligence import tools
from document_intelligence.data_agent.schema import DatabaseSchema
from document_intelligence.data_agent.sql_guard import SafeSQLExecutor, SQLResult
from document_intelligence.orchestration.backends import RAGBackend
from document_intelligence.orchestration.data_agent import DataAgent
from document_intelligence.vectorization.store import VectorStore

SERVER_NAME = "document-intelligence"
INSTRUCTIONS = (
    "Recherche et questions-réponses citées sur les documents d'entreprise indexés. "
    "Utilise search_documents pour retrouver des passages, ask_question pour obtenir "
    "une réponse rédigée avec ses sources, list_documents pour voir le corpus. "
    "Pour les données chiffrées des taxis de New York (janvier 2025) : ask_data en "
    "langage naturel, ou describe_tables puis run_sql (SELECT en lecture seule)."
)


def build_server(
    store: VectorStore,
    backend: RAGBackend | None,
    host: str = "127.0.0.1",
    port: int = 8001,
    sql_executor: SafeSQLExecutor | None = None,
    data_agent: DataAgent | None = None,
) -> FastMCP:
    """Construit le serveur MCP sur un vector store et un backend RAG donnés.

    Args:
        backend: backend de réponse (`RAG_BACKEND`) ; `None` = `ask_question`
            renvoie une erreur explicite, les deux autres outils restent disponibles.
        host, port: adresse d'écoute en transport `streamable-http` (ignorés en stdio).
        sql_executor: exécuteur SQL protégé (défaut : base taxi versionnée).
        data_agent: agent text-to-SQL ; `None` = `ask_data` renvoie une erreur
            explicite, `describe_tables` et `run_sql` restent disponibles.
    """
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS, host=host, port=port)
    executor = sql_executor or SafeSQLExecutor()

    @server.tool()
    async def search_documents(query: str, k: int = 5) -> tools.SearchResult:
        """Retourne les k passages (chunks) des documents indexés les plus pertinents
        pour la requête, avec leur score et leur document source. Aucun appel LLM."""
        return await to_thread.run_sync(tools.search_documents, store, query, k)

    @server.tool()
    async def ask_question(question: str, k: int = 5) -> tools.CitedAnswer:
        """Répond à la question à partir des documents indexés (pipeline RAG :
        recherche des k meilleurs passages puis synthèse citée). Renvoie la réponse,
        les sources citées et sufficient_context=false si les documents ne
        permettent pas de répondre."""
        return await to_thread.run_sync(tools.ask_question, backend, question, k)

    @server.tool()
    async def list_documents() -> tools.DocumentList:
        """Liste les documents indexés (identifiant, chemin source, nombre de chunks)."""
        return await to_thread.run_sync(tools.list_documents, store)

    @server.tool()
    async def describe_tables() -> DatabaseSchema:
        """Décrit les tables de la base de données (trajets des taxis jaunes de New
        York, janvier 2025, agrégés) : colonnes, types et signification."""
        return tools.describe_tables()

    @server.tool()
    async def run_sql(sql: str) -> SQLResult:
        """Exécute une requête SQL SELECT (SQLite) en lecture seule sur la base des
        taxis. Écritures, tables système et requêtes multiples sont refusées ;
        200 lignes au plus, 2 s au plus. Aucun appel LLM."""
        return await to_thread.run_sync(tools.run_sql, executor, sql)

    @server.tool()
    async def ask_data(question: str) -> tools.DataQueryAnswer:
        """Répond en langage naturel à une question chiffrée sur les taxis de New
        York (janvier 2025) : l'agent écrit et exécute le SQL, puis répond en
        citant la requête exécutée, ou refuse si la base ne permet pas de répondre."""
        return await to_thread.run_sync(tools.ask_data, data_agent, question)

    return server
