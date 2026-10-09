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
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from document_intelligence.orchestration.exceptions import (
    LLMOutputError,
)

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
        raise LLMOutputError("La sortie du modèle ne contient aucun objet JSON.")
    try:
        return schema.model_validate_json(cleaned[start : end + 1])
    except ValidationError as exc:
        raise LLMOutputError(
            f"La sortie du modèle ne respecte pas le schéma {schema.__name__} : {exc}"
        ) from exc


def _decode_stringified_fields(
    schema: type[T], payload: dict[str, Any]
) -> dict[str, Any]:
    """Décode les champs non textuels que le modèle a sérialisés en chaîne JSON.

    Le function calling renvoie parfois une liste sous forme de chaîne, voire
    l'objet entier sérialisé dans le champ (observé sur `claims` avec le vrai
    modèle). Seuls les champs dont le schéma n'est pas `str` sont décodés, pour
    ne jamais altérer un champ texte qui commencerait par « [ » ou « { ».
    """
    fixed: dict[str, Any] = {}
    for name, value in payload.items():
        field = schema.model_fields.get(name)
        fixed[name] = value
        if not (
            isinstance(value, str)
            and field is not None
            and field.annotation is not str
            and value.lstrip()[:1] in ("[", "{")
        ):
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            continue
        # La chaîne peut contenir l'objet entier (`'{"claims": [...]}'`) plutôt
        # que la seule valeur du champ.
        if isinstance(decoded, dict) and set(decoded) == {name}:
            decoded = decoded[name]
        fixed[name] = decoded
    return fixed


def validate_tool_input(schema: type[T], payload: Any) -> T:
    """Valide les arguments d'un appel d'outil, avec une réparation ciblée.

    Raises:
        LLMOutputError: si la charge utile ne respecte pas `schema`, même après réparation.
    """
    try:
        return schema.model_validate(payload)
    except ValidationError as first_error:
        if isinstance(payload, dict):
            try:
                return schema.model_validate(
                    _decode_stringified_fields(schema, payload)
                )
            except ValidationError:
                pass
        raise LLMOutputError(
            f"La sortie du modèle ne respecte pas le schéma {schema.__name__} : {first_error}"
        ) from first_error


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
        return parse_structured_output(
            self.complete(json_instruction_prompt(prompt, schema, system)), schema
        )


def json_instruction_prompt(
    prompt: str, schema: type[BaseModel], system: str | None = None
) -> str:
    """Prompt du repli « JSON par instruction » (modèles sans function calling).

    Partagé avec la construction du jeu de fine-tuning, pour que le modèle soit
    entraîné exactement sur le texte qu'il recevra à l'inférence.
    """
    instruction = (
        "Réponds uniquement avec un objet JSON conforme à ce schéma JSON :\n"
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
    )
    parts = [system, prompt, instruction] if system else [prompt, instruction]
    return "\n\n".join(parts)
