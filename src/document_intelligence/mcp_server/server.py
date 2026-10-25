"""Serveur MCP (Model Context Protocol) exposant la recherche documentaire.

Trois outils, adossés aux fonctions de `document_intelligence.tools` :
- `search_documents(query, k)` : chunks pertinents, sans appel LLM ;
- `ask_question(question, k)` : réponse du pipeline RAG avec ses sources ;
- `list_documents()` : documents indexés.

Construit avec le SDK MCP officiel (`mcp.server.fastmcp`). Les outils sont
asynchrones et délèguent le travail bloquant (recherche, appel LLM) à un
thread, pour ne pas bloquer la boucle d'événements du serveur.
"""

from anyio import to_thread
from mcp.server.fastmcp import FastMCP

from document_intelligence import tools
from document_intelligence.orchestration.llm import LLMClient
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.store import VectorStore

SERVER_NAME = "document-intelligence"
INSTRUCTIONS = (
    "Recherche et questions-réponses citées sur les documents d'entreprise indexés. "
    "Utilise search_documents pour retrouver des passages, ask_question pour obtenir "
    "une réponse rédigée avec ses sources, list_documents pour voir le corpus."
)


def build_server(
    store: VectorStore,
    llm: LLMClient | None,
    verifier: AnswerVerifier | None = None,
    max_regenerations: int = 1,
    host: str = "127.0.0.1",
    port: int = 8001,
) -> FastMCP:
    """Construit le serveur MCP sur un vector store et un client LLM donnés.

    Args:
        llm: client de synthèse ; `None` = `ask_question` renvoie une erreur
            explicite, les deux autres outils restent disponibles.
        host, port: adresse d'écoute en transport `streamable-http` (ignorés en stdio).
    """
    server = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS, host=host, port=port)

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
        return await to_thread.run_sync(
            lambda: tools.ask_question(
                store,
                llm,
                question,
                k=k,
                verifier=verifier,
                max_regenerations=max_regenerations,
            )
        )

    @server.tool()
    async def list_documents() -> tools.DocumentList:
        """Liste les documents indexés (identifiant, chemin source, nombre de chunks)."""
        return await to_thread.run_sync(tools.list_documents, store)

    return server
