"""
SAAGA Engine Single Runner
==========================
Executes a single CTF defense scenario with step-by-step console logging and telemetry.
"""
from __future__ import annotations

from typing import Any, Optional

from saaga.agents.controller import RedTeamingController
from saaga.agents.generator import AttackPromptGenerator
from saaga.agents.planner import RedTeamingPlanner
from saaga.core.scenario import DefenseScenario
from saaga.evaluators.extractor import SensitiveInfoExtractor
from saaga.evaluators.judge import StopPointIdentifier
from saaga.fuzzing.fallback import MutationFallback
from saaga.memory.kb import StrategyKnowledgeBase
from saaga.memory.rag import DefenseRetriever
from saaga.memory.updater import KBUpdater
from saaga.providers.base import BaseLLMProvider


def run_single_scenario_verbose(
    scenario: DefenseScenario,
    victim_provider: BaseLLMProvider,
    planner_provider: Optional[BaseLLMProvider] = None,
    generator_provider: Optional[BaseLLMProvider] = None,
    max_attempts: int = 20,
    enable_fallback: bool = True,
    scenario_id: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute a single red-teaming scenario with full logging."""
    p_prov = planner_provider or victim_provider
    g_prov = generator_provider or victim_provider

    kb = StrategyKnowledgeBase()
    retriever = DefenseRetriever()
    planner = RedTeamingPlanner(p_prov, kb=kb, retriever=retriever)
    generator = AttackPromptGenerator(g_prov, retriever=retriever)
    extractor = SensitiveInfoExtractor()
    judge = StopPointIdentifier()
    fallback = MutationFallback() if enable_fallback else None
    kb_updater = KBUpdater(mode="run")

    controller = RedTeamingController(
        victim_provider=victim_provider,
        planner=planner,
        generator=generator,
        extractor=extractor,
        judge=judge,
        fallback=fallback,
        kb_updater=kb_updater,
        max_attempts=max_attempts,
        verbose=verbose,
    )

    return controller.run_scenario(scenario, scenario_id=scenario_id)
