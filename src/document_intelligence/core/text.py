"""Utilitaires de normalisation de texte partagés (recherche lexicale, évaluation)."""

import re
import unicodedata


def normalize(text: str) -> str:
    """Normalise pour la comparaison : sans accents, minuscules, espaces compactés."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped).casefold().strip()
