"""Module d'orchestration multi-agents (Sprint 3).

Pipeline routage → retrieval → synthèse citée, construit comme une suite de
fonctions `AgentState -> AgentState` (`orchestration.nodes`) afin de pouvoir
être porté vers un `StateGraph` LangGraph sans changer la logique métier.
Point d'entrée public : `orchestration.graph.answer_query`.
"""
