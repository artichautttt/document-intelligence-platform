"""Lance le serveur MCP : `python -m document_intelligence.mcp_server`.

Même configuration que l'API (`.env` / variables d'environnement) : vector
store (`VECTOR_STORE_BACKEND`, `CHROMA_PERSIST_DIRECTORY`...), fournisseur LLM
(`LLM_PROVIDER` et sa clé), vérification (`ANSWER_VERIFICATION`).

Transport `stdio` par défaut (client qui lance le serveur, ex. Claude Desktop,
Claude Code) ; `streamable-http` pour un serveur réseau. En stdio, stdout est
réservé au protocole : les logs partent sur stderr.
"""

import argparse
import sys

from document_intelligence.core.config import settings
from document_intelligence.core.logging import configure_logging, get_logger
from document_intelligence.mcp_server.server import build_server
from document_intelligence.services import (
    build_backend,
    build_llm,
    build_store,
    build_verifier,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serveur MCP Document Intelligence.")
    parser.add_argument(
        "--transport", choices=["stdio", "streamable-http"], default="stdio"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)

    configure_logging(stream=sys.stderr)
    store = build_store()
    llm = build_llm()
    backend = build_backend(store, llm, build_verifier(None, llm))
    server = build_server(store, backend, host=args.host, port=args.port)
    get_logger(__name__).info(
        "mcp.startup",
        transport=args.transport,
        vector_store_backend=settings.vector_store_backend,
        retrieval_mode=settings.retrieval_mode,
        rag_backend=backend.name if backend else None,
    )
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
