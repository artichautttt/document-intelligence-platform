"""Exceptions explicites du module d'orchestration."""


class OrchestrationError(Exception):
    """Erreur de base pour toute anomalie survenant pendant l'exécution du pipeline d'agents."""


class EmptyQueryError(OrchestrationError):
    """Levée quand une requête utilisateur vide est soumise au pipeline."""


class NoRelevantChunkError(OrchestrationError):
    """Levée quand la recherche vectorielle ne retourne aucun chunk pour la requête."""


class LLMGenerationError(OrchestrationError):
    """Levée quand un `LLMClient` échoue à produire une complétion."""


class LLMOutputError(LLMGenerationError):
    """Levée quand le modèle répond, mais avec une sortie inexploitable (JSON/schéma invalide).

    Distincte des erreurs d'API (authentification, crédit, réseau), qui doivent
    interrompre un traitement, alors qu'une sortie invalide peut être isolée.
    """
