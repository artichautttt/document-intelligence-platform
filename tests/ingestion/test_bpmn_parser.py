"""Tests du parser BPMN 2.0 sur les exemples modélisés à la main et sur des cas limites."""

from pathlib import Path

import pytest

from document_intelligence.ingestion.exceptions import (
    CorruptFileError,
    EmptyDocumentError,
)
from document_intelligence.ingestion.models import ElementType, ParsedDocument
from document_intelligence.ingestion.parsers.bpmn_parser import (
    BpmnParser,
    bpmn_document,
    parse_bpmn,
)
from document_intelligence.ingestion.service import ingest

EXAMPLES = Path("examples/bpmn")
NS = 'xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"'


def _parse(name: str) -> ParsedDocument:
    return BpmnParser().parse(EXAMPLES / name)


def _task_text(document: ParsedDocument, task_id: str) -> str:
    return next(el.text for el in document.elements if el.element_id == task_id)


def _xml(body: str) -> bytes:
    return (
        f'<?xml version="1.0"?><definitions {NS} id="D">{body}</definitions>'.encode()
    )


class TestStructure:
    def test_one_process_element_then_its_tasks_in_flow_order(self) -> None:
        document = _parse("exemple_demande_conges.bpmn")
        assert document.source_format == "bpmn"
        assert [(el.element_type, el.element_id) for el in document.elements] == [
            (ElementType.PROCESS, "Process_Conges"),
            (ElementType.TASK, "Task_SaisirDemande"),
            (ElementType.TASK, "Task_ExaminerDemande"),
            (ElementType.TASK, "Task_MettreAJourSolde"),
            (ElementType.TASK, "Task_NotifierRefus"),
            (ElementType.TASK, "Task_ConfirmerConges"),
        ]

    def test_counts_on_the_three_examples(self) -> None:
        counts = {}
        for path in sorted(EXAMPLES.glob("*.bpmn")):
            document = BpmnParser().parse(path)
            counts[path.name] = (
                sum(el.element_type is ElementType.PROCESS for el in document.elements),
                sum(el.element_type is ElementType.TASK for el in document.elements),
            )
        assert counts == {
            "exemple_demande_conges.bpmn": (1, 5),
            "exemple_incident_informatique.bpmn": (2, 8),
            "exemple_note_de_frais.bpmn": (1, 6),
        }

    def test_process_element_lists_actors_and_tasks(self) -> None:
        process = _parse("exemple_demande_conges.bpmn").elements[0].text
        assert process.splitlines()[0] == "Processus : Demande de congés"
        assert "Organisation : Société Exemple" in process
        assert "Acteurs : Salarié, Manager, Service RH" in process
        assert (
            "Tâches : Saisir la demande de congés ; Examiner la demande de congés"
            in process
        )


class TestTaskDescription:
    def test_actor_is_lane_and_pool(self) -> None:
        text = _task_text(_parse("exemple_demande_conges.bpmn"), "Task_ExaminerDemande")
        assert "Acteur : Manager (Société Exemple)" in text

    def test_pool_without_lane_is_the_actor(self) -> None:
        text = _task_text(
            _parse("exemple_incident_informatique.bpmn"), "Task_SignalerIncident"
        )
        assert "Acteur : Utilisateur" in text

    def test_decision_lists_each_condition_and_target(self) -> None:
        text = _task_text(_parse("exemple_demande_conges.bpmn"), "Task_ExaminerDemande")
        assert (
            "Décision : « Demande approuvée ? » : si « Oui », « Mettre à jour le solde de congés » ;"
            " si « Non », « Notifier le refus motivé »"
        ) in text
        assert "Sorties : selon la décision ci-dessous" in text

    def test_nested_decision_is_detailed(self) -> None:
        text = _task_text(
            _parse("exemple_note_de_frais.bpmn"), "Task_ControlerJustificatifs"
        )
        assert (
            "si « Oui », (décision « Montant total supérieur à 500 euros ? » : "
            "si « Oui », « Approuver la dépense » ; si « Non », « Rembourser par virement »)"
        ) in text

    def test_inputs_follow_gateways_with_the_branch_taken(self) -> None:
        text = _task_text(_parse("exemple_demande_conges.bpmn"), "Task_NotifierRefus")
        assert (
            "Entrées : « Examiner la demande de congés » (décision « Demande approuvée ? » : « Non »)"
        ) in text
        assert "Sorties : fin du processus « Demande refusée »" in text

    def test_join_gateway_lists_every_predecessor(self) -> None:
        text = _task_text(
            _parse("exemple_incident_informatique.bpmn"), "Task_InformerUtilisateur"
        )
        assert (
            "Entrées : « Appliquer la solution connue » ; « Corriger l'incident »"
            in text
        )

    def test_parallel_split_is_an_output(self) -> None:
        text = _task_text(_parse("exemple_incident_informatique.bpmn"), "Task_Corriger")
        assert (
            "Sorties : en parallèle : « Informer l'utilisateur de la résolution »,"
            " « Documenter la solution dans la base de connaissances »"
        ) in text

    def test_message_flows_in_and_out(self) -> None:
        document = _parse("exemple_incident_informatique.bpmn")
        sent = _task_text(document, "Task_InformerUtilisateur")
        received = _task_text(document, "Task_RecevoirResolution")
        assert (
            "message « Avis de résolution » envoyé à « Recevoir l'avis de résolution » (Utilisateur)"
        ) in sent
        assert (
            "message « Avis de résolution » reçu de "
            "« Informer l'utilisateur de la résolution » (Support informatique)"
        ) in received

    def test_message_on_start_event_reaches_the_first_task(self) -> None:
        text = _task_text(
            _parse("exemple_incident_informatique.bpmn"), "Task_QualifierTicket"
        )
        assert (
            "début du processus « Ticket reçu », sur message « Ticket d'incident » reçu de "
            "« Signaler l'incident sur le portail » (Utilisateur)"
        ) in text

    def test_data_inputs_and_outputs(self) -> None:
        document = _parse("exemple_note_de_frais.bpmn")
        assert "donnée « Justificatifs de dépense (factures, tickets) »" in _task_text(
            document, "Task_ControlerJustificatifs"
        )
        assert "donnée « Ordre de virement »" in _task_text(document, "Task_Rembourser")

    def test_documentation_becomes_the_description(self) -> None:
        text = _task_text(_parse("exemple_demande_conges.bpmn"), "Task_SaisirDemande")
        assert "Description : Le salarié indique les dates" in text


class TestEdgeCases:
    def test_unnamed_task_falls_back_to_its_id(self) -> None:
        raw = _xml(
            '<process id="P"><startEvent id="S"/><task id="T1"/>'
            '<sequenceFlow id="f" sourceRef="S" targetRef="T1"/></process>'
        )
        document = bpmn_document(parse_bpmn(raw), "x.bpmn")
        assert document.elements[1].text.startswith("Tâche : T1")
        assert "Acteur : non précisé" in document.elements[1].text
        assert "Sorties : aucune" in document.elements[1].text

    def test_task_not_reached_from_a_start_event_is_kept(self) -> None:
        raw = _xml('<process id="P"><task id="T1" name="Orpheline"/></process>')
        document = bpmn_document(parse_bpmn(raw), "x.bpmn")
        assert [el.element_id for el in document.elements] == ["P", "T1"]

    def test_process_without_task_raises_empty_document(self) -> None:
        raw = _xml('<process id="P"><startEvent id="S"/></process>')
        with pytest.raises(EmptyDocumentError):
            bpmn_document(parse_bpmn(raw), "x.bpmn")

    def test_rejects_a_dtd(self) -> None:
        raw = b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY e "x">]><definitions/>'
        with pytest.raises(CorruptFileError):
            parse_bpmn(raw)

    def test_rejects_invalid_xml(self) -> None:
        with pytest.raises(CorruptFileError):
            parse_bpmn(b"<definitions><process>")

    def test_rejects_xml_that_is_not_bpmn(self) -> None:
        with pytest.raises(CorruptFileError):
            parse_bpmn(
                b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>'
            )

    def test_unreadable_file_is_a_corrupt_file(self, tmp_path: Path) -> None:
        with pytest.raises(CorruptFileError):
            BpmnParser().parse(tmp_path / "absent.bpmn")


class TestIngestService:
    def test_supports_bpmn_extension_only(self) -> None:
        parser = BpmnParser()
        assert parser.supports(Path("process.BPMN"))
        assert not parser.supports(Path("process.xml"))

    def test_ingest_selects_the_bpmn_parser_by_default(self) -> None:
        document = ingest(EXAMPLES / "exemple_note_de_frais.bpmn")
        assert document.source_format == "bpmn"
        assert document.elements[0].element_id == "Process_Frais"
