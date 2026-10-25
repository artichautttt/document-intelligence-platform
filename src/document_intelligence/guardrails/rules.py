"""Règles de détection de prompt injection, nommées et testables une par une.

Chaque règle est une liste d'expressions régulières appliquées à un texte
normalisé (minuscules, sans accents, espaces réduits) et à sa version
désobfusquée (leetspeak, lettres séparées par des points ou des tirets). Les
règles visent des *intentions* (remplacer les consignes, usurper un rôle
système, faire fuiter le prompt, exfiltrer par lien ou image) et non des
chaînes précises : aucune ne connaît les témoins du jeu d'attaques.

Cibles : `question` (texte saisi par l'utilisateur) et `document` (chunk
ingéré, injection indirecte). Certaines règles ne valent que pour une cible :
une URL avec paramètres est banale dans un document, suspecte dans une question.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Target = Literal["question", "document"]
BOTH: tuple[Target, ...] = ("question", "document")


@dataclass(frozen=True)
class InjectionRule:
    """Une règle : nom stable, description, motifs, cibles concernées."""

    name: str
    description: str
    patterns: tuple[str, ...]
    targets: tuple[Target, ...] = BOTH

    def matches(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in self.patterns)


@dataclass(frozen=True)
class RuleMatch:
    rule: str
    description: str


_IGNORE_VERBS = (
    r"ignor\w*|disregard\w*|forget|oubli\w*|olvid\w*|"
    r"ne tiens? (?:pas|plus) compte|fais abstraction|pass\w* outre|override|bypass"
)
_INSTRUCTION_NOUNS = (
    r"instruction\w*|consigne\w*|regle\w*|directive\w*|rules?|prompt|"
    r"ce qu.on t.a (?:dit|demande)|what you were told|instrucciones|indicaciones"
)

RULES: tuple[InjectionRule, ...] = (
    InjectionRule(
        name="instruction_override",
        description="Demande d'ignorer ou de remplacer les consignes en vigueur.",
        patterns=(
            rf"\b(?:{_IGNORE_VERBS})\b.{{0,60}}\b(?:{_INSTRUCTION_NOUNS})",
            r"\b(?:previous|prior|above|earlier|precedent\w*|anterieur\w*|"
            r"anteriores)\s+(?:instructions?|consignes?|regles?|rules|instrucciones)",
        ),
    ),
    InjectionRule(
        name="new_instructions",
        description="Annonce de nouvelles consignes ou d'une nouvelle mission.",
        patterns=(
            r"\bnouvel\w* (?:consigne|instruction|mission|regle|directive)s?\b",
            r"\bnew (?:instructions?|task|mission|rules?)\b",
            r"\b(?:from now on|a partir de maintenant|desormais|dorenavant|"
            r"de ahora en adelante)\b",
        ),
    ),
    InjectionRule(
        name="fake_system_role",
        description="Usurpation d'un rôle système ou d'un mode privilégié.",
        patterns=(
            r"(?:^|[\n#>\[(]|\s{2})\s*#*\s*(?:system|systeme|developer|admin)\s*(?:#+|:)",
            r"<\|?\s*(?:im_start|im_end|system|endoftext)\s*\|?>",
            r"\[/?(?:inst|sys)\]",
            r"\bmode (?:administrateur|admin|developpeur|debug|sans restriction)\b",
            r"\b(?:developer|admin|god|jailbreak|dan) mode\b",
        ),
    ),
    InjectionRule(
        name="delimiter_escape",
        description="Balises du format de prompt (fermeture d'extrait, fausse consigne).",
        patterns=(
            r"</?\s*(?:extraits?|question|instructions?|system|consignes?)\b[^>]*>",
        ),
    ),
    InjectionRule(
        name="prompt_leak",
        description="Demande de révéler le prompt ou les consignes système.",
        patterns=(
            r"\b(?:revele\w*|recopi\w*|affich\w*|repet\w*|donne\w*|montre\w*|"
            r"ecri\w*|reveal|print|show|repeat|output|display|tell me)\b.{0,50}"
            r"\b(?:system prompt|prompt systeme|ton prompt|your prompt|"
            r"(?:tes|vos|leurs|your|their) (?:instructions|consignes)|"
            r"(?:instructions|consignes) (?:systeme|initiales|cachees)|"
            r"system instructions)",
            r"\bwhat (?:instructions|rules) were you given\b",
        ),
    ),
    InjectionRule(
        name="markdown_image",
        description="Image Markdown (canal d'exfiltration par chargement d'URL).",
        patterns=(r"!\[[^\]]*\]\(\s*https?://",),
    ),
    InjectionRule(
        name="exfiltration_link",
        description="Demande d'ajouter un lien ou une image à la réponse.",
        patterns=(
            r"\b(?:ajout\w*|inclu\w*|insere\w*|termine\w*|append|add|include|insert)\b"
            r".{0,40}\b(?:lien|link|url|image|pixel)\b",
        ),
    ),
    InjectionRule(
        name="url_with_parameters",
        description="URL porteuse de paramètres dans une question.",
        patterns=(r"https?://\S+\?\S*=",),
        targets=("question",),
    ),
    InjectionRule(
        name="addressed_to_ai",
        description="Passage qui s'adresse au modèle qui le lit (injection indirecte).",
        patterns=(
            r"\b(?:note|remarque|message|instruction|consigne)s?\s+"
            r"(?:to|for|pour|a|destine\w* (?:a|au|aux))\s+(?:l.|les?\s+)?"
            r"(?:ai|ia|llm|assistant|chatbot|modele|language model|systemes? d.ia|"
            r"ai systems?)",
            r"\b(?:les |aux |the |any )?(?:assistants?|modeles? de langage|"
            r"language models?|llms?|ai systems?|chatbots?)\s+"
            r"(?:qui (?:lisent|lit|traitent|traite)|reading|processing|that reads?)\b",
            r"^\s*(?:assistant|ai|ia)\s*:",
            r"\b(?:assistant|ai|ia)\s*:\s*(?:quelle que soit|whatever|regardless)",
            r"\bquelle que soit la question\b|\bregardless of the question\b",
        ),
    ),
    InjectionRule(
        name="roleplay_jailbreak",
        description="Jeu de rôle visant à lever les règles.",
        patterns=(
            r"\b(?:imagine|fais comme si|suppose|pretend|act as|roleplay|tu es desormais|"
            r"you are now)\b.{0,80}\b(?:sans (?:aucune )?(?:regle|restriction|limite|filtre)s?|"
            r"no (?:rules|restrictions|limits|filters)|without (?:any )?(?:rules|restrictions))",
        ),
    ),
    InjectionRule(
        name="forced_output",
        description="Impose une réponse fixe sans rapport avec les documents.",
        patterns=(
            r"\b(?:reponds?|repondez|ecris|ecrivez|commence\w*|reply|respond|answer|"
            r"output|start|responde)\b.{0,40}\b(?:uniquement|seulement|only|solo|"
            r"rien d.autre|nothing else|par le mot|with the word)\b",
            r"\bla (?:seule |bonne )?reponse (?:attendue|correcte) est\b",
            r"\bthe (?:only |correct )?(?:expected )?answer is\b",
        ),
    ),
)

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"}
)
_SPACED_LETTERS = re.compile(r"\b(?:\w[.\-_*·]){3,}\w\b")


def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces (y compris insécables) réduits."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[ \t  ]+", " ", stripped.lower())


def deobfuscate(normalized: str) -> str:
    """Version « décodée » : leetspeak et lettres séparées (`i.g.n.o.r.e`)."""
    joined = _SPACED_LETTERS.sub(
        lambda m: re.sub(r"[.\-_*·]", "", m.group(0)), normalized
    )
    return joined.translate(_LEET)


def detect(text: str, target: Target) -> list[RuleMatch]:
    """Règles déclenchées par `text` pour la cible donnée (liste vide = sain)."""
    normalized = normalize(text)
    variants = (normalized, deobfuscate(normalized))
    return [
        RuleMatch(rule=rule.name, description=rule.description)
        for rule in RULES
        if target in rule.targets and any(rule.matches(v) for v in variants)
    ]
