"""Interface abstraite commune à tout fournisseur de modèle de langage.

Cette abstraction isole les agents du fournisseur LLM concret (Anthropic,
OpenAI, modèle local...), à l'image de `VectorStore` pour le vector store.
En plus de la complétion libre, elle expose la génération de **sorties
structurées** validées par un schéma Pydantic : les fournisseurs qui
supportent le function calling la surchargent, les autres héritent d'un
repli par prompt JSON.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from document_intelligence.orchestration.exceptions import LLMGenerationError

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_structured_output(text: str, schema: type[T]) -> T:
    """Extrait puis valide un objet JSON depuis la sortie brute d'un modèle.

    Tolère les balises de code Markdown et le texte autour de l'objet.

    Raises:
        LLMGenerationError: si aucun JSON valide conforme au schéma n'est trouvé.
    """
    cleaned = _CODE_FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise LLMGenerationError("La sortie du modèle ne contient aucun objet JSON.")
    try:
        return schema.model_validate_json(cleaned[start : end + 1])
    except ValidationError as exc:
        raise LLMGenerationError(
            f"La sortie du modèle ne respecte pas le schéma {schema.__name__} : {exc}"
        ) from exc


class LLMClient(ABC):
    """Contrat que doit respecter tout backend de génération de texte."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Retourne la complétion du modèle pour le prompt fourni.

        Raises:
            LLMGenerationError: si le fournisseur échoue à produire une réponse.
        """

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        """Retourne une réponse validée par `schema`.

        Implémentation par défaut : demande un objet JSON dans le prompt puis
        le parse. Les clients supportant le function calling la surchargent
        pour obtenir un JSON garanti conforme au schéma par le fournisseur.

        Raises:
            LLMGenerationError: si la génération échoue ou si la sortie est invalide.
        """
        instruction = (
            "Réponds uniquement avec un objet JSON conforme à ce schéma JSON :\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
        parts = [system, prompt, instruction] if system else [prompt, instruction]
        return parse_structured_output(self.complete("\n\n".join(parts)), schema)
