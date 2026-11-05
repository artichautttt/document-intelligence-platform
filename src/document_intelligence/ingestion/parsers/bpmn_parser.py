"""Parser BPMN 2.0 (XML) : processus, lanes, tâches, passerelles, événements et flux.

Un modèle de processus n'a pas de texte suivi : sa connaissance est dans la
structure (qui fait quoi, après quoi, selon quelle décision). Le parser
reconstruit donc, pour chaque tâche, une description textuelle à partir du
graphe :

- **acteur** : la lane qui contient la tâche, et le pool (participant) du
  processus ;
- **entrées** : tâches ou événements qui la précèdent (à travers les
  passerelles, avec la branche empruntée), messages reçus, données lues ;
- **sorties** : tâches ou fin qui la suivent, branches parallèles, messages
  envoyés, données produites ;
- **décision** : passerelle de décision qui suit la tâche, avec chaque
  condition et la tâche à laquelle elle mène.

Le `ParsedDocument` contient, par processus, un élément `PROCESS` suivi d'un
élément `TASK` par tâche, dans l'ordre du flux. Les `element_id` sont les
identifiants BPMN du fichier : une citation renvoie au processus puis à la
tâche exacte (voir `BpmnTaskChunker`).

Sécurité : un fichier avec DTD (`<!DOCTYPE`, `<!ENTITY`) est refusé, ce qui
écarte les attaques par expansion d'entités ; aucun XML externe n'est résolu.

Hors périmètre : le contenu des sous-processus (traités comme une tâche), les
événements attachés (boundary) et le diagramme (BPMNDI) ne sont pas décrits.
"""

import re
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from document_intelligence.ingestion.exceptions import (
    CorruptFileError,
    EmptyDocumentError,
)
from document_intelligence.ingestion.models import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)
from document_intelligence.ingestion.parsers.base import DocumentParser

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
_Q = f"{{{BPMN_NS}}}"
_DTD = re.compile(rb"<!(DOCTYPE|ENTITY)", re.IGNORECASE)
_MAX_DEPTH = 10  # passerelles enchaînées suivies au plus

ACTIVITY_KINDS: dict[str, str] = {
    "task": "tâche",
    "userTask": "tâche utilisateur",
    "manualTask": "tâche manuelle",
    "serviceTask": "tâche automatique (service)",
    "scriptTask": "tâche automatique (script)",
    "businessRuleTask": "règle métier",
    "sendTask": "envoi de message",
    "receiveTask": "réception de message",
    "callActivity": "appel de sous-processus",
    "subProcess": "sous-processus",
}
GATEWAY_KINDS: dict[str, str] = {
    "exclusiveGateway": "exclusive",
    "inclusiveGateway": "inclusive",
    "parallelGateway": "parallèle",
    "eventBasedGateway": "sur événement",
    "complexGateway": "complexe",
}
EVENT_TAGS = frozenset(
    {
        "startEvent",
        "endEvent",
        "intermediateCatchEvent",
        "intermediateThrowEvent",
        "boundaryEvent",
    }
)


@dataclass
class FlowNode:
    """Nœud du flux : activité, passerelle ou événement."""

    id: str
    tag: str
    name: str
    process_id: str
    documentation: str = ""

    @property
    def label(self) -> str:
        return self.name or self.id

    @property
    def is_activity(self) -> bool:
        return self.tag in ACTIVITY_KINDS

    @property
    def is_gateway(self) -> bool:
        return self.tag in GATEWAY_KINDS


@dataclass
class Flow:
    """Flux de séquence (dans un processus) ou de message (entre pools)."""

    id: str
    source: str
    target: str
    name: str = ""
    condition: str = ""

    @property
    def branch_label(self) -> str:
        return self.name or self.condition


@dataclass
class BpmnProcess:
    id: str
    name: str
    participant: str = ""
    documentation: str = ""
    lanes: dict[str, str] = field(default_factory=dict)  # id de nœud -> lane
    node_ids: list[str] = field(default_factory=list)  # ordre du fichier


@dataclass
class BpmnModel:
    """Graphe d'un fichier BPMN : processus, nœuds, flux et données."""

    processes: list[BpmnProcess]
    nodes: dict[str, FlowNode]
    sequence_flows: list[Flow]
    message_flows: list[Flow]
    participants: dict[str, str]  # id de participant -> nom
    data_inputs: dict[str, list[str]]  # id d'activité -> noms des données lues
    data_outputs: dict[str, list[str]]  # id d'activité -> noms des données produites

    def __post_init__(self) -> None:
        self._out: dict[str, list[Flow]] = {}
        self._in: dict[str, list[Flow]] = {}
        for flow in self.sequence_flows:
            self._out.setdefault(flow.source, []).append(flow)
            self._in.setdefault(flow.target, []).append(flow)

    def outgoing(self, node_id: str) -> list[Flow]:
        return self._out.get(node_id, [])

    def incoming(self, node_id: str) -> list[Flow]:
        return self._in.get(node_id, [])

    def ordered_activities(self, process: BpmnProcess) -> list[FlowNode]:
        """Activités du processus dans l'ordre du flux (parcours en largeur
        depuis les événements de début), puis celles non atteintes, dans
        l'ordre du fichier."""
        nodes = [self.nodes[i] for i in process.node_ids]
        starts = [n.id for n in nodes if n.tag == "startEvent"]
        seen: set[str] = set(starts)
        queue = deque(starts)
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for flow in self.outgoing(current):
                if flow.target not in seen:
                    seen.add(flow.target)
                    queue.append(flow.target)
        order += [n.id for n in nodes if n.id not in seen]
        return [
            self.nodes[i]
            for i in order
            if i in self.nodes and self.nodes[i].is_activity
        ]


def _text(element: ET.Element | None) -> str:
    return " ".join((element.text or "").split()) if element is not None else ""


def _local(tag: str) -> str:
    return tag[len(_Q) :] if tag.startswith(_Q) else ""


def parse_bpmn(raw: bytes, source: str = "<bpmn>") -> BpmnModel:
    """Lit un fichier BPMN 2.0 et retourne son graphe.

    Raises:
        CorruptFileError: XML invalide, DTD présente ou racine non BPMN 2.0.
    """
    if _DTD.search(raw):
        raise CorruptFileError(f"BPMN '{source}' refusé : DTD ou entité déclarée.")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise CorruptFileError(f"XML invalide dans '{source}' : {exc}") from exc
    if root.tag != f"{_Q}definitions":
        raise CorruptFileError(
            f"'{source}' n'est pas un fichier BPMN 2.0 (definitions)."
        )

    participants: dict[str, str] = {}
    process_participant: dict[str, str] = {}
    for participant in root.iter(f"{_Q}participant"):
        name = participant.get("name", "")
        participants[participant.get("id", "")] = name
        if participant.get("processRef"):
            process_participant[participant.get("processRef", "")] = name

    data_names: dict[str, str] = {}
    data_objects = {
        d.get("id", ""): d.get("name", "") for d in root.iter(f"{_Q}dataObject")
    }
    for ref in root.iter(f"{_Q}dataObjectReference"):
        data_names[ref.get("id", "")] = ref.get("name") or data_objects.get(
            ref.get("dataObjectRef", ""), ""
        )
    for ref in root.iter(f"{_Q}dataStoreReference"):
        data_names[ref.get("id", "")] = ref.get("name", "")

    processes: list[BpmnProcess] = []
    nodes: dict[str, FlowNode] = {}
    sequence_flows: list[Flow] = []
    raw_inputs: dict[str, list[str]] = {}
    raw_outputs: dict[str, list[str]] = {}

    for proc in root.findall(f"{_Q}process"):
        pid = proc.get("id", "")
        participant = process_participant.get(pid, "")
        process = BpmnProcess(
            id=pid,
            name=proc.get("name") or participant or pid,
            participant=participant,
            documentation=_text(proc.find(f"{_Q}documentation")),
        )
        # lanes imbriquées : la plus profonde, visitée en dernier, l'emporte
        for lane in proc.iter(f"{_Q}lane"):
            for ref in lane.findall(f"{_Q}flowNodeRef"):
                process.lanes[_text(ref)] = lane.get("name", "")

        for child in proc:
            tag = _local(child.tag)
            if tag == "sequenceFlow":
                sequence_flows.append(
                    Flow(
                        id=child.get("id", ""),
                        source=child.get("sourceRef", ""),
                        target=child.get("targetRef", ""),
                        name=child.get("name", ""),
                        condition=_text(child.find(f"{_Q}conditionExpression")),
                    )
                )
            elif tag in ACTIVITY_KINDS or tag in GATEWAY_KINDS or tag in EVENT_TAGS:
                node_id = child.get("id", "")
                nodes[node_id] = FlowNode(
                    id=node_id,
                    tag=tag,
                    name=" ".join(child.get("name", "").split()),
                    process_id=pid,
                    documentation=_text(child.find(f"{_Q}documentation")),
                )
                process.node_ids.append(node_id)
                for assoc in child.findall(f"{_Q}dataInputAssociation"):
                    raw_inputs.setdefault(node_id, []).extend(
                        _text(r) for r in assoc.findall(f"{_Q}sourceRef")
                    )
                for assoc in child.findall(f"{_Q}dataOutputAssociation"):
                    raw_outputs.setdefault(node_id, []).extend(
                        _text(r) for r in assoc.findall(f"{_Q}targetRef")
                    )
        processes.append(process)

    message_flows = [
        Flow(
            id=m.get("id", ""),
            source=m.get("sourceRef", ""),
            target=m.get("targetRef", ""),
            name=m.get("name", ""),
        )
        for m in root.iter(f"{_Q}messageFlow")
    ]

    def names(refs: dict[str, list[str]]) -> dict[str, list[str]]:
        # les associations vers un objet inconnu (spécifications de données
        # internes à la tâche) sont ignorées
        return {
            node: [data_names[r] or r for r in ids if r in data_names]
            for node, ids in refs.items()
        }

    return BpmnModel(
        processes=processes,
        nodes=nodes,
        sequence_flows=sequence_flows,
        message_flows=message_flows,
        participants=participants,
        data_inputs=names(raw_inputs),
        data_outputs=names(raw_outputs),
    )


def _quote(text: str) -> str:
    return f"« {text} »"


class _Describer:
    """Construit le texte d'un processus et de chacune de ses tâches."""

    def __init__(self, model: BpmnModel) -> None:
        self._m = model
        self._process_of = {p.id: p for p in model.processes}

    def _endpoint(self, ref: str) -> str:
        """Nom lisible d'une extrémité de flux de message (nœud ou pool)."""
        node = self._m.nodes.get(ref)
        if node is not None:
            pool = self._process_of[node.process_id].participant
            return f"{_quote(node.label)} ({pool})" if pool else _quote(node.label)
        return _quote(self._m.participants.get(ref) or ref)

    def _backward(self, flow: Flow, depth: int = 0) -> list[str]:
        source = self._m.nodes.get(flow.source)
        if source is None:
            return [_quote(flow.source)]
        if source.is_gateway and depth < _MAX_DEPTH:
            suffix = ""
            if len(self._m.outgoing(source.id)) > 1 and flow.branch_label:
                suffix = (
                    f" (décision {_quote(source.label)} : {_quote(flow.branch_label)})"
                )
            items = [
                item + suffix
                for previous in self._m.incoming(source.id)
                for item in self._backward(previous, depth + 1)
            ]
            return items or [_quote(source.label) + suffix]
        if source.tag == "startEvent":
            start = (
                f"début du processus {_quote(source.name)}"
                if source.name
                else "début du processus"
            )
            messages = [
                f"message {_quote(m.name or m.id)} reçu de {self._endpoint(m.source)}"
                for m in self._m.message_flows
                if m.target == source.id
            ]
            return [f"{start}, sur {', '.join(messages)}" if messages else start]
        if source.tag in EVENT_TAGS:
            return [f"événement {_quote(source.label)}"]
        return [_quote(source.label)]

    def _through_joins(self, node_id: str) -> FlowNode | None:
        """Suit les passerelles de jonction (une seule sortie) jusqu'au nœud utile."""
        node = self._m.nodes.get(node_id)
        depth = 0
        while (
            node is not None
            and node.is_gateway
            and len(self._m.outgoing(node.id)) == 1
            and depth < _MAX_DEPTH
        ):
            node = self._m.nodes.get(self._m.outgoing(node.id)[0].target)
            depth += 1
        return node

    @staticmethod
    def _target_label(node: FlowNode | None, fallback: str) -> str:
        if node is None:
            return _quote(fallback)
        if node.tag == "endEvent":
            return (
                f"fin du processus {_quote(node.name)}"
                if node.name
                else "fin du processus"
            )
        if node.is_gateway:
            return f"décision {_quote(node.label)}"
        if node.tag in EVENT_TAGS:
            return f"événement {_quote(node.label)}"
        return _quote(node.label)

    def _branch_target(self, flow: Flow, depth: int) -> str:
        node = self._through_joins(flow.target)
        if node is not None and node.is_gateway and len(self._m.outgoing(node.id)) > 1:
            if depth >= _MAX_DEPTH:
                return f"décision {_quote(node.label)}"
            if node.tag == "parallelGateway":
                return "en parallèle : " + ", ".join(
                    self._branch_target(b, depth + 1) for b in self._m.outgoing(node.id)
                )
            return f"(décision {self._decision(node, depth + 1)})"
        return self._target_label(node, flow.target)

    def _decision(self, gateway: FlowNode, depth: int = 0) -> str:
        """« Passerelle » : si « condition », cible ; ... (décisions imbriquées détaillées)."""
        branches = [
            (b.branch_label, self._branch_target(b, depth))
            for b in self._m.outgoing(gateway.id)
        ]
        return f"{_quote(gateway.label)} : " + " ; ".join(
            f"si {_quote(label)}, {target}" if label else f"sinon, {target}"
            for label, target in branches
        )

    def describe_process(self, process: BpmnProcess, tasks: list[FlowNode]) -> str:
        lanes = list(dict.fromkeys(v for v in process.lanes.values() if v))
        lines = [f"Processus : {process.name}"]
        if process.participant and process.participant != process.name:
            lines.append(f"Organisation : {process.participant}")
        if lanes:
            lines.append(f"Acteurs : {', '.join(lanes)}")
        if process.documentation:
            lines.append(f"Description : {process.documentation}")
        lines.append("Tâches : " + " ; ".join(t.label for t in tasks))
        return "\n".join(lines)

    def describe_task(self, task: FlowNode) -> str:
        process = self._process_of[task.process_id]
        lane = process.lanes.get(task.id, "")
        actor = lane or process.participant or "non précisé"
        if lane and process.participant and process.participant != lane:
            actor = f"{lane} ({process.participant})"

        inputs = [item for f in self._m.incoming(task.id) for item in self._backward(f)]
        inputs += [
            f"message {_quote(m.name or m.id)} reçu de {self._endpoint(m.source)}"
            for m in self._m.message_flows
            if m.target == task.id
        ]
        inputs += [f"donnée {_quote(d)}" for d in self._m.data_inputs.get(task.id, [])]

        outputs: list[str] = []
        decisions: list[str] = []
        for flow in self._m.outgoing(task.id):
            node = self._through_joins(flow.target)
            if node is not None and node.is_gateway and node.tag != "parallelGateway":
                decisions.append(self._decision(node))
            else:
                outputs.append(self._branch_target(flow, 0))
        outputs += [
            f"message {_quote(m.name or m.id)} envoyé à {self._endpoint(m.target)}"
            for m in self._m.message_flows
            if m.source == task.id
        ]
        outputs += [
            f"donnée {_quote(d)}" for d in self._m.data_outputs.get(task.id, [])
        ]

        lines = [
            f"Tâche : {task.label}",
            f"Type : {ACTIVITY_KINDS[task.tag]}",
            f"Acteur : {actor}",
            "Entrées : " + (" ; ".join(inputs) if inputs else "aucune"),
        ]
        if outputs:
            lines.append("Sorties : " + " ; ".join(outputs))
        elif decisions:
            lines.append("Sorties : selon la décision ci-dessous")
        else:
            lines.append("Sorties : aucune")
        if decisions:
            lines.append("Décision : " + " | ".join(decisions))
        if task.documentation:
            lines.append(f"Description : {task.documentation}")
        return "\n".join(lines)


def bpmn_document(model: BpmnModel, source_path: str) -> ParsedDocument:
    """Convertit un graphe BPMN en `ParsedDocument` : par processus, un élément
    `PROCESS` puis un élément `TASK` par activité, dans l'ordre du flux.

    Raises:
        EmptyDocumentError: si le fichier ne contient aucune activité.
    """
    describer = _Describer(model)
    elements: list[DocumentElement] = []
    for process in model.processes:
        tasks = model.ordered_activities(process)
        if not tasks:
            continue
        elements.append(
            DocumentElement(
                element_id=process.id,
                element_type=ElementType.PROCESS,
                text=describer.describe_process(process, tasks),
            )
        )
        elements += [
            DocumentElement(
                element_id=task.id,
                element_type=ElementType.TASK,
                text=describer.describe_task(task),
            )
            for task in tasks
        ]
    if not elements:
        raise EmptyDocumentError(f"Aucune tâche BPMN dans '{source_path}'")
    return ParsedDocument(
        document_id=str(uuid.uuid4()),
        source_path=source_path,
        source_format="bpmn",
        elements=elements,
    )


class BpmnParser(DocumentParser):
    """Parser pour les modèles de processus BPMN 2.0 (`.bpmn`)."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".bpmn"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CorruptFileError(
                f"Impossible de lire le BPMN '{path}': {exc}"
            ) from exc
        return bpmn_document(parse_bpmn(raw, str(path)), str(path))
