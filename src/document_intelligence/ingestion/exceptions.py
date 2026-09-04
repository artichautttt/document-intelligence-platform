"""Exceptions explicites du module d'ingestion.

Le pipeline ne doit jamais échouer silencieusement sur un fichier corrompu
ou dans un format non supporté : ces exceptions typées permettent à l'appelant
de distinguer précisément la cause et de décider comment réagir (retry, alerte,
mise en quarantaine du fichier, etc.).
"""


class IngestionError(Exception):
    """Erreur de base pour toute anomalie survenant pendant l'ingestion d'un document."""


class UnsupportedFormatError(IngestionError):
    """Levée quand l'extension/format du fichier n'est pas géré par un parser connu."""


class CorruptFileError(IngestionError):
    """Levée quand le fichier existe et a un format supporté, mais est illisible ou malformé."""


class EmptyDocumentError(IngestionError):
    """Levée quand le document est parsé avec succès mais ne contient aucun contenu exploitable."""
