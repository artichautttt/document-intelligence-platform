"""Interface abstraite commune à tout fournisseur de modèle de langage.

Cette abstraction isole les agents du fournisseur LLM concret (Anthropic,
OpenAI, modèle local...), à l'image de `VectorStore` pour le vector store.
Aucune implémentation concrète n'est fournie au Sprint 3 : le choix du
fournisseur et la gestion des clés d'API sont hors périmètre de ce sprint.
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Contrat que doit respecter tout backend de génération de texte."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Retourne la complétion du modèle pour le prompt fourni.

        Raises:
            LLMGenerationError: si le fournisseur échoue à produire une réponse.
        """
