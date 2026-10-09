"""Agent de vérification : boucle de régénération du graphe et adaptateur CrewAI.

Le graphe est testé avec un vérificateur factice (sans CrewAI ni réseau). Les
tests de `CrewAIVerifier` exécutent le vrai CrewAI avec un `LLMClient` factice ;
ils sont ignorés si le groupe de dépendances `agents` n'est pas installé.
"""

import re
import uuid

import pytest

from document_intelligence.chunking.models import ChunkProvenance
from document_intelligence.evaluation.models import EvalExample
from document_intelligence.evaluation.runner import evaluate
from document_intelligence.evaluation.verification import corrupt_answer
from document_intelligence.observability.prometheus import (
    RAG_ANSWER_VERIFICATION_TOTAL,
)
from document_intelligence.orchestration.exceptions import LLMOutputError
from document_intelligence.orchestration.graph import answer_query
from document_intelligence.orchestration.llm import LLMClient, T
from document_intelligence.orchestration.models import VerificationVerdict
from document_intelligence.orchestration.verification import AnswerVerifier
from document_intelligence.vectorization.models import QueryResult
from document_intelligence.vectorization.store import VectorStore


def _result(text: str) -> QueryResult:
    return QueryResult(
        chunk_id=str(uuid.uuid4()),
        text=text,
        score=1.0,
        provenance=ChunkProvenance(
            document_id="doc-1", source_path="/tmp/doc.pdf", element_ids=["e1"]
        ),
    )


class StaticStore(VectorStore):
    def __init__(self, *texts: str) -> None:
        self._results = [_result(t) for t in texts]

    def add_chunks(self, chunks: list) -> None:  # type: ignore[type-arg]
        pass

    def query(self, text: str, k: int = 5) -> list[QueryResult]:
        return self._results[:k]


class ScriptedLLM(LLMClient):
    """Rédacteur factice : renvoie les réponses prévues, dans l'ordre."""

    def __init__(self, *answers: str, sufficient: bool = True) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.sufficient = sufficient

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        self.prompts.append(prompt)
        ids = re.findall(r'<extrait id="([^"]+)"', prompt)
        return schema.model_validate(
            {
                "answer": self.answers.pop(0),
                "cited_chunk_ids": ids[:1],
                "sufficient_context": self.sufficient,
            }
        )


class NeutralJudge(LLMClient):
    """LLM-juge factice : aucune affirmation relevée, pertinence maximale."""

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        return schema.model_validate({"claims": [], "score": 5, "reasoning": ""})


class ScriptedVerifier(AnswerVerifier):
    """Vérificateur factice : rejette les réponses contenant un mot interdit."""

    def __init__(self, forbidden: str = "15") -> None:
        self.forbidden = forbidden
        self.seen: list[str] = []

    def verify(
        self, query: str, answer: str, sources: list[QueryResult]
    ) -> VerificationVerdict:
        self.seen.append(answer)
        if self.forbidden in answer:
            return VerificationVerdict(
                grounded=False,
                unsupported_claims=[f"{self.forbidden} M€"],
                feedback="Le chiffre n'est pas dans les extraits.",
            )
        return VerificationVerdict(grounded=True)


class TestVerificationLoop:
    def test_grounded_answer_is_verified_without_regeneration(self) -> None:
        llm = ScriptedLLM("Le CA est de 48,2 M€.")
        verifier = ScriptedVerifier()

        answer = answer_query(
            "CA ?", store=StaticStore("CA 48,2 M€"), llm=llm, verifier=verifier
        )

        assert answer.verified is True
        assert answer.regenerations == 0
        assert len(llm.prompts) == 1

    def test_flagged_answer_is_regenerated_with_feedback(self) -> None:
        llm = ScriptedLLM("Le CA est de 15 M€.", "Le CA est de 48,2 M€.")
        verifier = ScriptedVerifier()
        flagged = RAG_ANSWER_VERIFICATION_TOTAL.labels("flagged")
        before = flagged._value.get()

        answer = answer_query(
            "CA ?", store=StaticStore("CA 48,2 M€"), llm=llm, verifier=verifier
        )

        assert answer.answer == "Le CA est de 48,2 M€."
        assert answer.verified is True
        assert answer.regenerations == 1
        assert answer.rejected_answers == ["Le CA est de 15 M€."]
        assert [v.grounded for v in answer.verifications] == [False, True]
        assert "<retour_verification>" not in llm.prompts[0]
        assert "<retour_verification>" in llm.prompts[1]
        assert "15 M€" in llm.prompts[1]
        assert flagged._value.get() - before == 1

    def test_regeneration_stops_at_the_limit(self) -> None:
        llm = ScriptedLLM("15 a", "15 b", "15 c")
        verifier = ScriptedVerifier()

        answer = answer_query(
            "CA ?",
            store=StaticStore("CA 48,2 M€"),
            llm=llm,
            verifier=verifier,
            max_regenerations=1,
        )

        assert answer.answer == "15 b"
        assert answer.verified is False
        assert answer.regenerations == 1
        assert len(llm.prompts) == 2

    def test_zero_regenerations_only_flags(self) -> None:
        llm = ScriptedLLM("15 a")

        answer = answer_query(
            "CA ?",
            store=StaticStore("x"),
            llm=llm,
            verifier=ScriptedVerifier(),
            max_regenerations=0,
        )

        assert answer.verified is False
        assert len(llm.prompts) == 1

    def test_abstention_is_not_verified(self) -> None:
        verifier = ScriptedVerifier()

        answer = answer_query(
            "CA ?",
            store=StaticStore("x"),
            llm=ScriptedLLM("Pas d'information.", sufficient=False),
            verifier=verifier,
        )

        assert answer.verified is None
        assert verifier.seen == []

    def test_invalid_verdict_leaves_answer_unverified(self) -> None:
        class BrokenVerifier(AnswerVerifier):
            def verify(
                self, query: str, answer: str, sources: list[QueryResult]
            ) -> VerificationVerdict:
                raise LLMOutputError("verdict illisible")

        answer = answer_query(
            "CA ?",
            store=StaticStore("x"),
            llm=ScriptedLLM("réponse"),
            verifier=BrokenVerifier(),
        )

        assert answer.answer == "réponse"
        assert answer.verified is None

    def test_without_verifier_the_graph_is_unchanged(self) -> None:
        answer = answer_query("CA ?", store=StaticStore("x"), llm=ScriptedLLM("15"))

        assert answer.verified is None
        assert answer.verifications == []


class TestCalibration:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("Le CA est de 48,2 M€.", "Le CA est de 71,5 M€."),
            ("Hausse de 12 % en 2025.", "Hausse de 45 % en 2025."),
            ("Préavis de 90 jours.", "Préavis de 23 jours."),
        ],
    )
    def test_first_number_is_altered(self, answer: str, expected: str) -> None:
        assert corrupt_answer(answer) == expected

    def test_answer_without_number_is_skipped(self) -> None:
        assert corrupt_answer("Le contrat est régi par le droit français.") is None

    def test_evaluate_counts_flags_and_calibration(self) -> None:
        examples = [
            EvalExample(question="CA ?", relevant_substrings=["48,2"]),
            EvalExample(question="Droit ?", relevant_substrings=["48,2"]),
        ]
        llm = ScriptedLLM("Le CA est de 15 M€.", "Le CA est de 48,2 M€.", "Droit FR")
        # Rejette « 15 » ; la copie faussée de « 15 M€ » (« 48 M€ ») est acceptée.
        verifier = ScriptedVerifier()

        report = evaluate(
            examples,
            StaticStore("CA 48,2 M€"),
            k=1,
            llm=llm,
            judge=NeutralJudge(),
            verifier=verifier,
            calibrate_verifier=True,
        )

        agg = report.aggregate
        assert agg["verification_checked"] == 2
        assert agg["verification_flagged"] == 1
        assert agg["verification_regenerations"] == 1
        assert agg["verification_final_flagged"] == 0
        # « Droit FR » ne contient aucun nombre : une seule copie faussée.
        assert agg["calibration_cases"] == 1
        outcome = report.results[0].verification
        assert outcome is not None
        assert outcome.calibration_corrupted_answer == "Le CA est de 48 M€."


class RecordingLLM(LLMClient):
    """`LLMClient` factice qui enregistre ce que CrewAI lui transmet."""

    model = "fake-model"

    def __init__(self, verdict: VerificationVerdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str | None, str]] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(("complete", None, prompt))
        return self.verdict.model_dump_json()

    def complete_structured(
        self, prompt: str, schema: type[T], system: str | None = None
    ) -> T:
        self.calls.append((schema.__name__, system, prompt))
        return schema.model_validate(self.verdict.model_dump())


class TestCrewAIVerifier:
    @pytest.fixture(autouse=True)
    def _crewai(self) -> None:
        pytest.importorskip("crewai")

    def test_crew_delegates_to_the_project_llm_client(self) -> None:
        from document_intelligence.orchestration.crew_verifier import CrewAIVerifier

        expected = VerificationVerdict(
            grounded=False, unsupported_claims=["15 M€"], feedback="Corriger le CA."
        )
        llm = RecordingLLM(expected)
        source = _result("Le CA est de 48,2 M€ {accolades}.")

        verdict = CrewAIVerifier(llm).verify(
            "Quel CA ?", "Le CA est de 15 M€.", [source]
        )

        assert verdict == expected
        assert len(llm.calls) == 1
        schema_name, system, prompt = llm.calls[0]
        # Sortie structurée par function calling, pas par parsing de texte.
        assert schema_name == "VerificationVerdict"
        assert system is not None and "Vérificateur de faits" in system
        assert "<reponse>Le CA est de 15 M€.</reponse>" in prompt
        assert f'<extrait id="{source.chunk_id}">' in prompt
        assert "{accolades}" in prompt

    def test_verifier_plugs_into_the_graph(self) -> None:
        from document_intelligence.orchestration.crew_verifier import CrewAIVerifier

        llm = RecordingLLM(VerificationVerdict(grounded=True))

        answer = answer_query(
            "CA ?",
            store=StaticStore("CA 48,2 M€"),
            llm=ScriptedLLM("48,2 M€"),
            verifier=CrewAIVerifier(llm),
        )

        assert answer.verified is True
