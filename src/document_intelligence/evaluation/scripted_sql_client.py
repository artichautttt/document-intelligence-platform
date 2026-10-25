"""Client Anthropic simulé et déterministe pour la suite de non-régression SQL.

Il imite `anthropic.Anthropic().messages.create` sans aucun appel réseau : pour
une question du jeu d'évaluation, il appelle `run_sql` avec la **requête de
référence**, puis `submit_answer` avec la première valeur du résultat ; pour une
question sans réponse, il refuse directement.

Il vérifie donc la mécanique (boucle d'outils, garde-fous SQL, exécution,
format de la réponse, calcul des métriques), **pas la qualité d'un vrai
modèle** : par construction, il connaît la bonne requête.
"""

import json
from types import SimpleNamespace
from typing import Any

from document_intelligence.evaluation.sql_eval_cases import SQLEvalCase


class ScriptedSQLClient:
    """Doublure de `anthropic.Anthropic` jouant les requêtes de référence."""

    def __init__(self, cases: list[SQLEvalCase]) -> None:
        self._cases = {case.question: case for case in cases}
        self.calls = 0
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        messages: list[dict[str, Any]] = kwargs["messages"]
        question = str(messages[0]["content"])
        case = self._cases.get(question)
        if case is None or not case.answerable or case.reference_sql is None:
            return _response(
                _tool(
                    "submit_answer",
                    answer="La base ne permet pas de répondre à cette question.",
                    value=None,
                    sql=None,
                    refused=True,
                )
            )
        if len(messages) == 1:
            return _response(_tool("run_sql", sql=case.reference_sql))
        result = messages[-1]["content"][0]
        if result.get("is_error"):
            return _response(
                _tool(
                    "submit_answer",
                    answer=f"Échec de la requête : {result['content']}",
                    value=None,
                    sql=None,
                    refused=True,
                )
            )
        rows = json.loads(result["content"])["rows"]
        value = rows[0][0] if rows and rows[0] else None
        return _response(
            _tool(
                "submit_answer",
                answer=f"Résultat : {value}.",
                value=value,
                sql=case.reference_sql,
                refused=False,
            )
        )


def _tool(name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use", id=f"scripted-{name}", name=name, input=arguments
    )


def _response(*blocks: SimpleNamespace) -> SimpleNamespace:
    # Aucun jeton facturé : le coût d'un run simulé est nul.
    return SimpleNamespace(
        content=list(blocks),
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
    )
