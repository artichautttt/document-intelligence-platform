"""Cache exact-match des réponses LLM, avec TTL, en mémoire ou dans un fichier local.

La clé est une empreinte du modèle, du mode d'appel et du prompt **normalisé**
(Unicode NFKC, casse, espaces) : deux requêtes qui ne diffèrent que par la
casse ou les espaces partagent la même réponse. Le prompt contient les
extraits récupérés, donc une réponse n'est réutilisée que pour un contexte
identique — une ré-ingestion qui change les extraits n'est jamais servie
depuis une entrée périmée.
"""

import hashlib
import json
import os
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from document_intelligence.core.logging import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Forme canonique d'un texte pour la comparaison exact-match."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def cache_key(*parts: str | None) -> str:
    """Empreinte SHA-256 des éléments normalisés d'un appel."""
    normalized = [None if part is None else normalize_text(part) for part in parts]
    payload = json.dumps(normalized, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """Réponse mise en cache et tokens consommés par l'appel qui l'a produite."""

    value: Any
    input_tokens: int
    output_tokens: int
    expires_at: float


class LLMResponseCache:
    """Cache LRU borné avec expiration ; sûr en accès concurrent.

    Args:
        ttl_seconds: durée de validité d'une entrée.
        max_entries: au-delà, l'entrée la moins récemment utilisée est évincée.
        path: fichier JSON de persistance ; `None` = mémoire seule.
        clock: horloge injectable (tests).
    """

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int = 1024,
        path: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("ttl_seconds et max_entries doivent être positifs.")
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._path = path
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        if path is not None:
            self._load(path)

    def get(self, key: str) -> CacheEntry | None:
        """Entrée valide pour `key`, ou `None` (absente ou expirée)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self._clock():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry

    def set(self, key: str, value: Any, input_tokens: int, output_tokens: int) -> None:
        """Enregistre `value` (JSON-sérialisable) pour la durée du TTL."""
        entry = CacheEntry(
            value=value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            expires_at=self._clock() + self._ttl,
        )
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            if self._path is not None:
                self._save(self._path)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _load(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            now = self._clock()
            for key, fields in raw.items():
                entry = CacheEntry(**fields)
                if entry.expires_at > now:
                    self._entries[key] = entry
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError):
            # Un cache illisible ne doit jamais empêcher de répondre.
            logger.warning("llm_cache.load_failed", path=str(path), exc_info=True)
            self._entries.clear()

    def _save(self, path: Path) -> None:
        payload = {key: asdict(entry) for key, entry in self._entries.items()}
        tmp = path.with_name(f"{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            logger.warning("llm_cache.save_failed", path=str(path), exc_info=True)
