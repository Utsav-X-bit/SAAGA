"""
SAAGA Execution Engine Subpackage
=================================
High-performance batched and single-scenario execution engines,
prompt processing, model loaders, and benchmark orchestrators.
"""
from saaga.engine.prompting import (
    strip_think_blocks,
    strip_few_shot_patterns,
    apply_chat_template_safe,
    truncate_system_content_to_fit,
)
from saaga.engine.model_loader import (
    load_vllm_engine,
    load_classifier_model,
)
from saaga.engine.batch_runner import run_scenarios_batched
from saaga.engine.single_runner import run_single_scenario_verbose
from saaga.engine.benchmark import (
    load_scenarios_from_jsonl,
    execute_benchmark,
)

__all__ = [
    "strip_think_blocks",
    "strip_few_shot_patterns",
    "apply_chat_template_safe",
    "truncate_system_content_to_fit",
    "load_vllm_engine",
    "load_classifier_model",
    "run_scenarios_batched",
    "run_single_scenario_verbose",
    "load_scenarios_from_jsonl",
    "execute_benchmark",
]
