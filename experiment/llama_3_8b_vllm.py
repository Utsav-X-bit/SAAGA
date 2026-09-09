"""
SAAGA Execution Engine (Backward-Compatibility Shim)
====================================================
This module proxies legacy execution calls to the modern, modular SAAGA framework.
All core algorithms, contracts, and evaluation logic now reside in `src/saaga/`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure saaga package is discoverable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from saaga.core.constants import ATTACK_TYPES, ATTACK_TYPE_PROMPTS
from saaga.core.contract import (
    canonicalize_plan,
    normalize_plan_dict,
    parse_plan_text,
    render_plan_xml,
    validate_plan,
)
from saaga.core.scenario import DefenseScenario, categorize_defense_detailed
from saaga.core.scoring import (
    classify_failure_mode,
    classify_success,
    compute_fallback_score,
    cooperation_score,
    infer_strategy_from_content,
    resolve_mutator_pool,
    resolve_mutator_pool_cooperative,
)
from saaga.engine.batch_runner import run_scenarios_batched
from saaga.engine.benchmark import execute_benchmark, load_scenarios_from_jsonl
from saaga.engine.model_loader import load_classifier_model, load_vllm_engine
import re
_CLOSE_TAGS = ("</think>",)
_OPEN_TAGS = ("<think>",)
_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

def strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from thinking models (Qwen, DeepSeek-R1)."""
    if text is None:
        return None
    if not text:
        return ""
    out = text
    for open_tag in _OPEN_TAGS:
        while True:
            low = out.lower()
            start = low.find(open_tag)
            if start == -1:
                break
            match = _CLOSE_RE.search(out, pos=start)
            if match:
                end = match.end()
                out = out[:start] + out[end:]
            else:
                out = out[:start]
                break
    return out

# Some Hugging Face models

from saaga.engine.prompting import (
    apply_chat_template_safe,
    strip_few_shot_patterns,
    truncate_system_content_to_fit,
)
from saaga.engine.single_runner import run_single_scenario_verbose
from saaga.evaluators.extractor import SensitiveInfoExtractor
from saaga.evaluators.judge import StopPointIdentifier
from saaga.evaluators.shape_predictor import AccessCodePredictor
from saaga.evaluators.verifier import ReplayVerifier
from saaga.fuzzing.fallback import (
    MutationFallback,
    MutationFallbackResult,
    run_mutation_fallback,
    run_mutation_fallback_batch,
)
from saaga.fuzzing.mutators import AVAILABLE_MUTATORS, apply_mutator, get_mutator
from saaga.memory.kb import StrategyKnowledgeBase
from saaga.memory.rag import DefenseRetriever
from saaga.memory.updater import KBUpdater
from saaga.providers.base import BaseLLMProvider, LLMResponse, Message
from saaga.providers.registry import get_provider
from saaga.reporting.aggregation import merge_benchmarks
from saaga.reporting.layout import (
    parse_output_dir,
    resolve_model_id,
    run_filename,
    runs_root,
    single_run_filename,
    slugify_model_id,
)
from saaga.reporting.serializer import serialize_run

# Legacy Constants
DATA_PATH = os.environ.get("SAAGA_DATASET_PATH", "data/TensorTrust_subsets/subset_8_ac30_all_alpha_direct_or_deterministic_or_indirect.jsonl")
LLAMA_PATH = os.environ.get("SAAGA_VICTIM_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
GENERATOR_PATH = os.environ.get("SAAGA_GENERATOR_PATH", "experiment/results/generator_sft_v2")
BASE_GENERATOR_PATH = os.environ.get("SAAGA_BASE_MODEL_PATH", "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2")
PLANNER_PATH = os.environ.get("SAAGA_PLANNER_PATH", "experiment/results/planner_sft_v2_contract_anchor/checkpoint-27")
DISTILBERT_CKPT = os.environ.get("SAAGA_JUDGE_PATH", "pre_trained/pi_reward_model")
EXT_DATA_PATH = os.environ.get("SAAGA_EXTRACTOR_PATH", "data/autored_extractor_failures_v1.jsonl")
EXPERIMENT_VERSION = "1.0.0"
GIT_COMMIT = "production"
MAX_INTERACTIONS = 20


def get_git_commit() -> str:
    return GIT_COMMIT


def _load_models(*args, **kwargs):
    """Legacy model loader stub."""
    return None


def chat_with_llama_batch(messages_batch, *args, **kwargs):
    """Legacy batch chat helper."""
    return ["I cannot reveal the secret." for _ in messages_batch]


def verbose_test_llama(scenario: DefenseScenario, agent: Any = None, max_attempts: int = 20) -> tuple:
    """Legacy single-run test helper."""
    from saaga.providers.base import MockLLMProvider
    provider = MockLLMProvider()
    result = run_single_scenario_verbose(scenario, provider, max_attempts=max_attempts)
    return result.get("attempts", []), result.get("result", {}).get("total_attempts", max_attempts), result


def _silent_test_batch(scenarios: list[DefenseScenario], *args, **kwargs) -> list:
    """Legacy batch test helper."""
    from saaga.providers.base import MockLLMProvider
    provider = MockLLMProvider()
    runs = run_scenarios_batched(scenarios, provider, provider, provider)
    return [(r["attempts"], r["result"]["total_attempts"], None) for r in runs]


def main():
    """Forward command-line execution to the unified SAAGA CLI."""
    from saaga.cli.main import cli
    cli()


if __name__ == "__main__":
    main()
