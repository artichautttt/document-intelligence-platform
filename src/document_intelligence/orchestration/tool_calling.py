"""Outils communs aux boucles de tool calling natif (API Messages d'Anthropic)."""

from typing import Any


def block_to_param(block: Any) -> dict[str, Any] | None:
    """Bloc de réponse → bloc de message renvoyé au tour suivant (`None` = omis).

    Un bloc texte vide est omis : l'API le renvoie parfois à côté d'appels
    d'outils mais le refuse en entrée (400 « text content blocks must be
    non-empty »). Les autres types de blocs sont renvoyés tels quels.
    """
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if block.type == "text":
        text = getattr(block, "text", "")
        return {"type": "text", "text": text} if text.strip() else None
    dump = getattr(block, "model_dump", None)
    return dict(dump(exclude_none=True)) if callable(dump) else None
