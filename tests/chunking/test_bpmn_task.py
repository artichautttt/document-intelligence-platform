"""Tests du chunking BPMN (une tâche = un chunk) et du choix de stratégie par défaut."""

import uuid
from pathlib import Path

from document_intelligence.chunking.service import chunk_document
from document_intelligence.chunking.strategies.bpmn_task import BpmnTaskChunker
from document_intelligence.chunking.strategies.semantic_section import (
    SemanticSectionChunker,
)
from document_intelligence.ingestion.models import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)
from document_intelligence.ingestion.parsers.bpmn_parser import BpmnParser

EXAMPLES = Path("examples/bpmn")


def _document(elements: list[DocumentElement], source_format: str) -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-1",
        source_path=f"/tmp/fake.{source_format}",
        source_format=source_format,
        elements=elements,
    )


def _el(
    element_type: ElementType, text: str, element_id: str | None = None
) -> DocumentElement:
    return DocumentElement(
        element_id=element_id or str(uuid.uuid4()), element_type=element_type, text=text
    )


class TestBpmnTaskChunker:
    def test_one_chunk_per_task_cited_as_process_then_task(self) -> None:
        document = BpmnParser().parse(EXAMPLES / "exemple_incident_informatique.bpmn")
        chunks = BpmnTaskChunker().chunk(document)

        tasks = [el for el in document.elements if el.element_type is ElementType.TASK]
        assert len(chunks) == len(tasks) == 8
        assert [c.provenance.element_ids for c in chunks[:3]] == [
            ["Process_Utilisateur", "Task_SignalerIncident"],
            ["Process_Utilisateur", "Task_RecevoirResolution"],
            ["Process_Support", "Task_QualifierTicket"],
        ]
        assert all(c.provenance.document_id == document.document_id for c in chunks)

    def test_chunk_starts_with_its_process_then_its_task(self) -> None:
        document = BpmnParser().parse(EXAMPLES / "exemple_demande_conges.bpmn")
        first = BpmnTaskChunker().chunk(document)[0]
        assert first.text.splitlines()[:2] == [
            "Processus : Demande de congés",
            "Tâche : Saisir la demande de congés",
        ]

    def test_each_chunk_holds_actor_inputs_and_outputs(self) -> None:
        for path in sorted(EXAMPLES.glob("*.bpmn")):
            for chunk in BpmnTaskChunker().chunk(BpmnParser().parse(path)):
                lines = chunk.text.splitlines()
                assert {"Acteur", "Entrées", "Sorties"} <= {
                    line.split(" : ")[0] for line in lines
                }

    def test_task_without_process_is_cited_alone(self) -> None:
        document = _document([_el(ElementType.TASK, "Tâche : A", "T1")], "bpmn")
        chunks = BpmnTaskChunker().chunk(document)
        assert chunks[0].provenance.element_ids == ["T1"]
        assert chunks[0].text == "Tâche : A"

    def test_process_element_alone_gives_no_chunk(self) -> None:
        document = _document([_el(ElementType.PROCESS, "Processus : P", "P")], "bpmn")
        assert BpmnTaskChunker().chunk(document) == []


class TestDefaultStrategy:
    def test_bpmn_document_uses_the_task_chunker(self) -> None:
        document = BpmnParser().parse(EXAMPLES / "exemple_note_de_frais.bpmn")
        chunks = chunk_document(document)
        assert len(chunks) == 6
        assert all(len(c.provenance.element_ids) == 2 for c in chunks)

    def test_other_formats_keep_the_section_chunker(self) -> None:
        elements = [
            _el(ElementType.TITLE, "Section A", "e1"),
            _el(ElementType.NARRATIVE_TEXT, "Contenu de la section A.", "e2"),
            _el(ElementType.TITLE, "Section B", "e3"),
            _el(ElementType.NARRATIVE_TEXT, "Contenu de la section B.", "e4"),
        ]
        document = _document(elements, "pdf")
        default = chunk_document(document)
        expected = SemanticSectionChunker().chunk(document)
        assert [(c.text, c.provenance.element_ids) for c in default] == [
            (c.text, c.provenance.element_ids) for c in expected
        ]

    def test_explicit_strategy_wins_over_the_format(self) -> None:
        document = BpmnParser().parse(EXAMPLES / "exemple_demande_conges.bpmn")
        chunks = chunk_document(document, strategy=SemanticSectionChunker())
        assert len(chunks) != 5
