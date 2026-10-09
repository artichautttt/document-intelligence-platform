"""Calibration de l'agent de vérification par injection d'erreurs contrôlées.

Compter les réponses signalées par le vérificateur ne dit pas s'il détecte
vraiment une hallucination : sur un pipeline qui hallucine peu, « 0 signalement »
est compatible avec un vérificateur qui ne signale jamais rien. On lui soumet
donc, pour chaque réponse réelle contenant un nombre, une copie dont le premier
nombre est altéré ; un vérificateur utile doit la rejeter.

Limite : seules des substitutions numériques sont testées — l'erreur la plus
facile à détecter. Cela ne mesure ni les extrapolations subtiles ni les
affirmations ajoutées sans chiffre.
"""

import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def corrupt_answer(answer: str) -> str | None:
    """Copie de `answer` dont le premier nombre est altéré ; `None` sans nombre.

    Chaque chiffre du premier nombre est décalé de 3 (mod 10) : la valeur change
    toujours, le format (séparateurs, nombre de chiffres) est conservé.
    """
    match = _NUMBER.search(answer)
    if match is None:
        return None
    altered = "".join(
        str((int(char) + 3) % 10) if char.isdigit() else char for char in match.group()
    )
    return f"{answer[: match.start()]}{altered}{answer[match.end() :]}"
