"""
SAAGA: Strategic and Adaptive Attack Generation for Red Teaming of LLMs
========================================================================
Production-grade red teaming, CTF evaluation, and prompt defense fuzzing framework.
"""

__version__ = "1.0.0"
__author__ = "SAAGA Research Team"

from saaga.core.scenario import DefenseScenario
from saaga.core.contract import canonicalize_plan, validate_plan, render_plan_xml
from saaga.core.scoring import classify_success, classify_failure_mode, cooperation_score

__all__ = [
    "DefenseScenario",
    "canonicalize_plan",
    "validate_plan",
    "render_plan_xml",
    "classify_success",
    "classify_failure_mode",
    "cooperation_score",
]
