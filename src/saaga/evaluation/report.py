"""
Evaluation Scorecard Report Writer
==================================
Renders a defense-strength scorecard into `summary.json` (machine-readable) and
`report.md` (human-readable) under the evaluation output directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from saaga.evaluation.defense_scorer import DefenseScoreCard, TierStats


def _render_tier_table(tiers: dict[str, TierStats]) -> str:
    """Render the per-tier scorecard as a markdown table."""
    header = (
        "| Tier | Scenarios | Broken | DSS | MTB (attempts) | MSB (severity) | Leak Resistance |\n"
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for name, s in tiers.items():
        mtb = f"{s.mtb:.2f}" if s.mtb is not None else "—"
        msb = f"{s.msb:.3f}" if s.msb is not None else "—"
        rows.append(
            f"| `{name}` | {s.total} | {s.broken} | **{s.dss:.1f}** | {mtb} | {msb} | {s.leak_resistance:.1f}% |"
        )
    return header + "\n".join(rows) + "\n"


def _render_card(title: str, card: DefenseScoreCard) -> str:
    head = card.to_dict()["headline"]
    return (
        f"### {title}\n\n"
        f"**Victim model:** `{card.victim_model}`  "
        f"**Attempt cap:** {card.max_attempts}  "
        f"**Scenarios:** {card.total_scenarios}\n\n"
        f"| Headline Metric | Value |\n"
        f"| :--- | ---: |\n"
        f"| Weighted Defense Strength Score (DSS_w) | **{head['weighted_defense_strength_score']:.1f}** / 100 |\n"
        f"| Overall Defense Strength (DSS) | {head['overall_defense_strength_score']:.1f} / 100 |\n"
        f"| Secret Protection (direct/indirect) | {head['secret_protection']:.1f} / 100 |\n"
        f"| Compliance Resistance (not_recoverable) | {head['compliance_resistance']:.1f} / 100 |\n\n"
        + _render_tier_table(card.tiers)
    )


def render_markdown(result: dict[str, Any]) -> str:
    """Render the full evaluation report markdown."""
    card: DefenseScoreCard = result["scorecard"]
    meta = result.get("meta", {})
    mode = meta.get("mode", "adaptive")
    seed = meta.get("seed", 42)
    tiers = ", ".join(meta.get("tiers", []))
    counts = meta.get("scenario_counts", {})

    body = [
        "# SAAGA Defense Strength Evaluation",
        "",
        f"**Mode:** `{mode}`  **Seed:** {seed}  **Tiers:** `{tiers}`",
        "",
        f"**Scenario sample counts:** {json.dumps(counts)}",
        "",
        "> Strength is measured *relative to the fixed SAAGA attacker* "
        "(planner + generator LoRA adapters, `max_attempts` attempts per scenario). "
        "Higher DSS = stronger defense. MTB = mean attempts-to-break (higher is stronger). "
        "MSB = mean severity of break (lower is better; 1.0 = secret leaked verbatim).",
        "",
        _render_card("Adaptive Attack Scorecard", card),
    ]

    if "static_scorecard" in result:
        body.append(_render_card("Static Screen Scorecard (cheap, no planner/generator)", result["static_scorecard"]))

    return "\n".join(body) + "\n"


def write_scorecard(
    result: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write summary.json and report.md under output_dir.

    Args:
        result: the dict returned by `run_evaluation`.
        output_dir: directory to write artifacts into (created if missing).

    Returns:
        the output directory Path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    card: DefenseScoreCard = result["scorecard"]
    payload: dict[str, Any] = {
        "victim_model": card.victim_model,
        "meta": result.get("meta", {}),
        "scorecard": card.to_dict(),
    }
    if "static_scorecard" in result:
        payload["static_scorecard"] = result["static_scorecard"].to_dict()

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(render_markdown(result))

    return out