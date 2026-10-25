"""Vérificateur de sortie déterministe (lexical, sans appel LLM).

Chaque phrase d'une réponse doit être soutenue par les extraits qu'elle cite :

- tous ses nombres figurent dans ces extraits (en chiffres ou, de 0 à 100, en
  toutes lettres : « 7 ans » est soutenu par « sept ans ») ;
- au moins `min_overlap` de ses mots porteurs de sens (4 lettres ou plus, hors
  mots vides) figurent dans ces extraits ;
- elle ne contient ni image Markdown, ni URL absente des extraits, ni passage
  du prompt système.

Une phrase sans mot porteur de sens (« Source : », puce vide) est neutre. Si
toutes les phrases utiles sont soutenues, la réponse passe ; si une partie
seulement l'est, elle est dégradée (seules ces phrases sont gardées) ; sinon
elle devient un refus.

Limite assumée : c'est un contrôle de surface. Une paraphrase fidèle peut
être refusée à tort, et une phrase fausse qui réemploie les mots des extraits
peut passer.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from document_intelligence.guardrails.rules import normalize
from document_intelligence.orchestration.prompts import SYNTHESIS_SYSTEM_PROMPT

Action = Literal["pass", "degrade", "refuse"]

DEFAULT_MIN_OVERLAP = 0.5
_LEAK_WINDOW = 6

_STOPWORDS = frozenset(
    [
        "avec",
        "dans",
        "pour",
        "par",
        "sur",
        "sous",
        "sans",
        "entre",
        "chez",
        "vers",
        "depuis",
        "pendant",
        "avant",
        "apres",
        "selon",
        "contre",
        "afin",
        "ainsi",
        "alors",
        "aussi",
        "autre",
        "autres",
        "aucun",
        "aucune",
        "cela",
        "celle",
        "celui",
        "ceux",
        "cette",
        "ces",
        "ceci",
        "comme",
        "dont",
        "elle",
        "elles",
        "etre",
        "etait",
        "sont",
        "sera",
        "seront",
        "avoir",
        "avait",
        "ont",
        "leur",
        "leurs",
        "lequel",
        "laquelle",
        "lesquels",
        "mais",
        "meme",
        "moins",
        "plus",
        "tres",
        "tout",
        "toute",
        "tous",
        "toutes",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "quand",
        "quoi",
        "qui",
        "que",
        "est",
        "une",
        "des",
        "les",
        "nous",
        "vous",
        "ils",
        "fait",
        "faire",
        "doit",
        "peut",
        "donc",
        "encore",
        "deja",
        "bien",
        "ici",
        "cest",
        "dun",
        "dune",
        "extrait",
        "extraits",
        "document",
        "documents",
        "reponse",
        "question",
        "indique",
        "indiquent",
        "precise",
        "mentionne",
        "mentionnent",
        "prevoit",
        "prevu",
        "prevue",
        "prevus",
        "selon",
        "montant",
        "concerne",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "into",
        "about",
        "which",
        "their",
        "there",
        "have",
        "been",
        "were",
        "will",
        "would",
        "should",
        "could",
        "also",
        "than",
        "then",
        "they",
        "them",
        "what",
        "when",
        "where",
    ]
)
_NUMBER_WORDS = {
    0: "zero", 1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six",
    7: "sept", 8: "huit", 9: "neuf", 10: "dix", 11: "onze", 12: "douze",
    13: "treize", 14: "quatorze", 15: "quinze", 16: "seize", 20: "vingt",
    30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante",
    70: "soixante-dix", 80: "quatre-vingts", 90: "quatre-vingt-dix", 100: "cent",
}  # fmt: skip

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
# Milliers séparés par une espace (« 184 000 »), décimale par point ou virgule.
_NUMBER = re.compile(r"\d+(?: \d{3})*(?:[.,]\d+)?")
_WORD = re.compile(r"[a-z][a-z0-9'-]{3,}")
_URL = re.compile(r"https?://[^\s)\]>\"']+")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_SOURCE_LINE = re.compile(r"^\W*(?:sources?|references?)\W*(?::|$)")


@dataclass(frozen=True)
class SentenceCheck:
    sentence: str
    supported: bool
    reason: str


@dataclass
class OutputVerdict:
    action: Action
    kept: list[str] = field(default_factory=list)
    removed: list[SentenceCheck] = field(default_factory=list)

    @property
    def answer(self) -> str:
        return " ".join(self.kept)


def _canonical_number(raw: str) -> str:
    """« 184 000 » → « 184000 » ; « 48,2 » → « 48.2 »."""
    return raw.replace(" ", "").replace(",", ".")


def _numbers(normalized: str) -> set[str]:
    return {_canonical_number(m.group(0)) for m in _NUMBER.finditer(normalized)}


def _words(normalized: str) -> set[str]:
    return {w.strip("'-") for w in _WORD.findall(normalized)} - _STOPWORDS


def _stem(word: str) -> str:
    """Radical grossier (6 premières lettres) : « redevances » ~ « redevance »."""
    return word[:6]


def _leak_windows() -> set[str]:
    words = normalize(SYNTHESIS_SYSTEM_PROMPT).split()
    return {
        " ".join(words[i : i + _LEAK_WINDOW])
        for i in range(len(words) - _LEAK_WINDOW + 1)
    }


_PROMPT_WINDOWS = _leak_windows()


def leaks_system_prompt(text: str, context: str = "") -> bool:
    """Vrai si `text` reprend 6 mots consécutifs du prompt système.

    Un passage également présent dans les extraits (`context`, normalisé) n'est
    pas une fuite : l'exemple du prompt reprend une clause du corpus, qu'une
    réponse fidèle cite légitimement.
    """
    words = normalize(text).split()
    flat_context = " ".join(context.split())
    windows = (
        " ".join(words[i : i + _LEAK_WINDOW])
        for i in range(len(words) - _LEAK_WINDOW + 1)
    )
    return any(w in _PROMPT_WINDOWS and w not in flat_context for w in windows)


def split_sentences(answer: str) -> list[str]:
    cleaned = re.sub(r"[*_`#]+", "", answer)
    return [
        s.strip(" -•\t") for s in _SENTENCE_SPLIT.split(cleaned) if s.strip(" -•\t")
    ]


def check_sentence(
    sentence: str, context: str, min_overlap: float = DEFAULT_MIN_OVERLAP
) -> SentenceCheck | None:
    """Vérifie une phrase contre le texte normalisé des extraits (`None` = neutre)."""
    if _MD_IMAGE.search(sentence):
        return SentenceCheck(sentence, False, "image Markdown")
    for url in _URL.findall(sentence):
        if normalize(url) not in context:
            return SentenceCheck(sentence, False, f"URL absente des extraits : {url}")
    if leaks_system_prompt(sentence, context):
        return SentenceCheck(sentence, False, "reprise du prompt système")

    normalized = normalize(sentence)
    if _SOURCE_LINE.match(normalized):
        return None
    context_numbers = _numbers(context)
    for number in _numbers(normalized):
        word = _NUMBER_WORDS.get(int(float(number)))
        if number not in context_numbers and not (
            word and re.search(rf"\b{word}\b", context)
        ):
            return SentenceCheck(
                sentence, False, f"nombre absent des extraits : {number}"
            )
    words = _words(normalized)
    if not words:
        return None
    context_stems = {_stem(w) for w in _WORD.findall(context)}
    covered = sum(1 for w in words if _stem(w) in context_stems)
    ratio = covered / len(words)
    if ratio < min_overlap:
        return SentenceCheck(
            sentence, False, f"recouvrement lexical {ratio:.2f} < {min_overlap}"
        )
    return SentenceCheck(sentence, True, f"recouvrement lexical {ratio:.2f}")


def verify_output(
    answer: str, cited_texts: list[str], min_overlap: float = DEFAULT_MIN_OVERLAP
) -> OutputVerdict:
    """Verdict sur une réponse non abstentionniste et les textes qu'elle cite."""
    context = normalize("\n".join(cited_texts))
    if not context.strip():
        return OutputVerdict(
            action="refuse",
            removed=[SentenceCheck(answer, False, "aucun extrait cité")],
        )
    kept: list[str] = []
    removed: list[SentenceCheck] = []
    useful = 0
    for sentence in split_sentences(answer):
        check = check_sentence(sentence, context, min_overlap)
        if check is None:
            kept.append(sentence)
            continue
        useful += 1
        if check.supported:
            kept.append(sentence)
        else:
            removed.append(check)
    if not removed:
        return OutputVerdict(action="pass", kept=kept)
    if useful == len(removed):
        return OutputVerdict(action="refuse", removed=removed)
    return OutputVerdict(action="degrade", kept=kept, removed=removed)
