"""
Shared planner output contract, XML parser, and canonicalization pipeline.
==========================================================================
Centralizes parsing, normalization, validation, and serialization for the
Planner's structured <plan> output to guarantee deterministic execution across
different model providers.
"""
from __future__ import annotations

import json
import re
from typing import Any

from saaga.core.constants import (
    KNOWN_ACCESS_TYPES,
    KNOWN_FAILURE_REASONS,
    KNOWN_POLICIES,
    KNOWN_STRATEGIES,
    KNOWN_STYLES,
)


def extract_tag(text: str, tag: str) -> str | None:
    """Extract inner content of an XML-like tag from text, case-insensitively.

    Args:
        text: Target text potentially containing tags.
        tag: Tag name without brackets (e.g. 'strategy').

    Returns:
        Stripped inner content if found, else None.
    """
    if not text:
        return None
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def strip_markdown_enclosures(text: str) -> str:
    """Strip markdown code block wrappers (e.g. ```xml ... ```) if present."""
    if not text:
        return ""
    stripped = text.strip()
    match = re.search(r"```(?:xml)?\s*(<plan>.*?</plan>)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped


def parse_plan_text(output: str) -> dict[str, Any]:
    """Parse raw LLM output into a dictionary of extracted plan components.

    Handles XML tags, JSON fallbacks for step lists, and markdown wrappers.

    Args:
        output: Raw completion string from planner LLM.

    Returns:
        Dictionary with extracted fields (unnormalized).
    """
    clean_output = strip_markdown_enclosures(output or "")

    confidence_raw = extract_tag(clean_output, "confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else -1.0
    except (TypeError, ValueError):
        confidence = -1.0

    prim_block = extract_tag(clean_output, "primitive_sequence") or ""
    primitives = re.findall(r"<step>(.*?)</step>", prim_block, re.DOTALL | re.IGNORECASE)
    if not primitives and prim_block.strip():
        # Fallback 1: JSON array within the primitive_sequence tag
        try:
            parsed = json.loads(prim_block)
            if isinstance(parsed, list):
                primitives = [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            # Fallback 2: Comma or newline separated lines
            lines = [line.strip().lstrip("-*123456789. ") for line in prim_block.splitlines()]
            primitives = [line for line in lines if line]

    return {
        "strategy": extract_tag(clean_output, "strategy"),
        "primitives": [p.strip() for p in primitives if p.strip()],
        "style": extract_tag(clean_output, "style"),
        "expected_access_type": extract_tag(clean_output, "expected_access_type"),
        "expected_access_code_type": extract_tag(clean_output, "expected_access_code_type"),
        "expected_access_code": extract_tag(clean_output, "expected_access_code"),
        "expected_output": extract_tag(clean_output, "expected_output"),
        "retry_policy": extract_tag(clean_output, "retry_policy"),
        "confidence": confidence,
        "failure_reason": extract_tag(clean_output, "failure_reason"),
    }


def normalize_plan_dict(plan: dict[str, Any], output: str = "") -> dict[str, Any]:
    """Normalize extracted plan dictionary field names, aliases, and defaults.

    Args:
        plan: Dictionary of extracted or provided plan fields.
        output: Optional original output string for fallback parsing.

    Returns:
        Normalized dictionary with unified keys.
    """
    normalized = dict(plan)

    if normalized.get("expected_access_type") in (None, ""):
        alias = (
            normalized.get("expected_access_code_type")
            or normalized.get("expected_access_code")
            or normalized.get("expected_output")
        )
        if alias:
            normalized["expected_access_type"] = alias

    failure_reason = normalized.get("failure_reason")
    if (
        failure_reason in (None, "", "n/a", "N/A", "NA", "none")
        or failure_reason not in KNOWN_FAILURE_REASONS
    ):
        normalized["failure_reason"] = "none"

    return normalized


def canonicalize_plan(plan: dict[str, Any], output: str = "") -> dict[str, Any]:
    """Validate, constrain, and canonicalize plan fields against strict taxonomy.

    Guarantees that all fields exist, are strictly typed, and fall within valid
    domain ranges (e.g. valid strategy, 1-5 primitives, clamped confidence).

    Args:
        plan: Raw or partially parsed plan dictionary.
        output: Optional raw output string used if plan dictionary is empty.

    Returns:
        Canonical plan dictionary ready for execution or serialization.
    """
    if not plan and output:
        plan = parse_plan_text(output)

    candidate = normalize_plan_dict(plan, output)

    strategy = candidate.get("strategy")
    if strategy not in KNOWN_STRATEGIES:
        strategy = "instruction_leak"

    primitives_val = candidate.get("primitives") or []
    if isinstance(primitives_val, str):
        try:
            parsed = json.loads(primitives_val)
            if isinstance(parsed, list):
                primitives = [str(item).strip() for item in parsed if str(item).strip()]
            else:
                primitives = [primitives_val.strip()]
        except Exception:
            primitives = [primitives_val.strip()]
    elif isinstance(primitives_val, list):
        primitives = [str(item).strip() for item in primitives_val if str(item).strip()]
    else:
        primitives = []

    if not primitives:
        primitives = ["framing/educational_context"]
    primitives = primitives[:5]

    style = candidate.get("style")
    if style not in KNOWN_STYLES:
        style = "direct"

    access_type = candidate.get("expected_access_type")
    if access_type not in KNOWN_ACCESS_TYPES:
        access_type = "UNKNOWN"

    retry_policy = candidate.get("retry_policy")
    if retry_policy not in KNOWN_POLICIES:
        retry_policy = "explore"

    failure_reason = candidate.get("failure_reason")
    if failure_reason not in KNOWN_FAILURE_REASONS:
        failure_reason = "none"

    confidence_val = candidate.get("confidence")
    try:
        confidence = float(confidence_val) if confidence_val is not None else 0.50
        if confidence < 0.0:
            confidence = 0.50
    except (TypeError, ValueError):
        confidence = 0.50
    confidence = max(0.0, min(1.0, confidence))

    return {
        "strategy": strategy,
        "primitives": primitives,
        "style": style,
        "expected_access_type": access_type,
        "retry_policy": retry_policy,
        "confidence": confidence,
        "failure_reason": failure_reason,
    }


def validate_plan(plan: dict[str, Any], output: str = "", *, strict: bool = False) -> list[str]:
    """Validate a plan dictionary and its corresponding raw output text.

    Args:
        plan: Plan dictionary to check.
        output: Raw LLM output string containing <plan> tags.
        strict: If True, tests the raw plan dict without normalization.

    Returns:
        List of validation error strings. An empty list indicates a valid plan.
    """
    candidate = plan if strict else normalize_plan_dict(plan, output)

    errors: list[str] = []
    if output and ("<plan>" not in output.lower() or "</plan>" not in output.lower()):
        errors.append("missing <plan> wrapper")
    if candidate.get("strategy") not in KNOWN_STRATEGIES:
        errors.append(f"unknown strategy: {candidate.get('strategy')}")
    prims = candidate.get("primitives")
    if not isinstance(prims, list) or not (1 <= len(prims) <= 5):
        errors.append(f"invalid primitive count: {len(prims) if isinstance(prims, list) else 0}")
    if candidate.get("style") not in KNOWN_STYLES:
        errors.append(f"unknown style: {candidate.get('style')}")
    if candidate.get("retry_policy") not in KNOWN_POLICIES:
        errors.append(f"unknown retry_policy: {candidate.get('retry_policy')}")
    if candidate.get("expected_access_type") not in KNOWN_ACCESS_TYPES:
        errors.append(f"unknown access type: {candidate.get('expected_access_type')}")
    if candidate.get("failure_reason") not in KNOWN_FAILURE_REASONS:
        errors.append(f"unknown failure_reason: {candidate.get('failure_reason')}")
    try:
        conf = float(candidate.get("confidence", -1))
        if not (0.0 <= conf <= 1.0):
            errors.append(f"confidence out of range: {conf}")
    except (TypeError, ValueError):
        errors.append(f"confidence not a valid float: {candidate.get('confidence')}")

    if any(v in (None, "") for k, v in candidate.items() if k != "primitives"):
        errors.append("one or more plan fields are empty")
    return errors


def render_plan_xml(plan: dict[str, Any]) -> str:
    """Serialize a plan dictionary into canonical XML format for generation agents.

    Args:
        plan: Plan dictionary (will be canonicalized automatically).

    Returns:
        Standardized <plan> XML string.
    """
    canonical = canonicalize_plan(plan)
    primitive_steps = "\n".join(f"    <step>{step}</step>" for step in canonical["primitives"])
    return (
        "<plan>\n"
        f"  <strategy>{canonical['strategy']}</strategy>\n"
        "  <primitive_sequence>\n"
        f"{primitive_steps}\n"
        "  </primitive_sequence>\n"
        f"  <style>{canonical['style']}</style>\n"
        f"  <expected_access_type>{canonical['expected_access_type']}</expected_access_type>\n"
        f"  <retry_policy>{canonical['retry_policy']}</retry_policy>\n"
        f"  <confidence>{canonical['confidence']:.2f}</confidence>\n"
        f"  <failure_reason>{canonical['failure_reason']}</failure_reason>\n"
        "</plan>"
    )
