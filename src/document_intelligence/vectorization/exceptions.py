"""Exceptions explicites du module de vectorisation."""


class VectorizationError(Exception):
    """Erreur de base pour toute anomalie survenant pendant la vectorisation ou la recherche."""


class EmptyChunkListError(VectorizationError):
    """Levée quand `add_chunks` est appelé avec une liste de chunks vide."""


class VectorStoreConnectionError(VectorizationError):
    """Levée quand la connexion ou l'initialisation du vector store échoue."""
