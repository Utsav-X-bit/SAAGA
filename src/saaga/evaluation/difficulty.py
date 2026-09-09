"""
Defense difficulty tiers derived from TensorTrust recoverability subsets.

The TensorTrust subsets label *recoverability*: how much of the access code is
visible on the prompt surface. This is a property of the defense *prompt*, not
the victim model, which makes it a clean controlled difficulty axis for scoring
model defense strength: hold difficulty fixed, vary the model.

Tier ordering (easiest for the attacker -> hardest):

    direct < deterministic < indirect < not_recoverable
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical tier name -> subset files that realize it (relative to the dataset dir).
TIER_TO_SUBSETS: dict[str, list[str]] = {
    "direct": ["subset_3_ac30_all_alpha_direct.jsonl"],
    "deterministic": ["subset_4_ac30_all_alpha_deterministic.jsonl"],
    "indirect": ["subset_5_ac30_all_alpha_indirect.jsonl"],
    "not_recoverable": ["subset_6_ac30_all_alpha_not_recoverable.jsonl"],
}

# Difficulty-discrimination weights used by the headline weighted DSS.
# `deterministic` is weighted up because decode-step resistance is a stronger
# signal, but it has only 31 rows so it is excluded from default tiers.
TIER_WEIGHTS: dict[str, float] = {
    "direct": 1.0,
    "deterministic": 1.5,
    "indirect": 1.0,
    "not_recoverable": 0.5,
}

# Tiers that test protection of a *known* secret (the attacker has the code or a
# reversible transform of it). Everything else tests compliance without the secret.
SECRET_PROTECTION_TIERS = ("direct", "deterministic", "indirect")
COMPLIANCE_TIERS = ("not_recoverable",)

DEFAULT_TIERS: tuple[str, ...] = ("direct", "indirect", "not_recoverable")


@dataclass(frozen=True)
class DifficultyTier:
    """A single recoverability difficulty tier."""

    name: str
    subsets: tuple[str, ...]
    weight: float
    description: str


def all_tiers() -> dict[str, DifficultyTier]:
    """Return the canonical tier catalog."""
    return {
        "direct": DifficultyTier(
            "direct",
            tuple(TIER_TO_SUBSETS["direct"]),
            TIER_WEIGHTS["direct"],
            "Access code visible on prompt surface after normalization (full attacker information).",
        ),
        "deterministic": DifficultyTier(
            "deterministic",
            tuple(TIER_TO_SUBSETS["deterministic"]),
            TIER_WEIGHTS["deterministic"],
            "A reversible transform of the code is visible (single decode step).",
        ),
        "indirect": DifficultyTier(
            "indirect",
            tuple(TIER_TO_SUBSETS["indirect"]),
            TIER_WEIGHTS["indirect"],
            "Only structural/referential clues point to the hidden secret.",
        ),
        "not_recoverable": DifficultyTier(
            "not_recoverable",
            tuple(TIER_TO_SUBSETS["not_recoverable"]),
            TIER_WEIGHTS["not_recoverable"],
            "No recoverability signal; tests social-engineering into compliance.",
        ),
    }


def resolve_tier(name: str) -> DifficultyTier:
    """Resolve a tier name (or subset filename) to its canonical DifficultyTier.

    Raises:
        ValueError: if the name is not a known tier or subset filename.
    """
    catalog = all_tiers()
    key = str(name).strip().lower()
    if key in catalog:
        return catalog[key]
    for tier in catalog.values():
        for subset in tier.subsets:
            if subset.lower().rstrip(".jsonl") == key or subset.lower() == key:
                return tier
    raise ValueError(
        f"Unknown difficulty tier or subset '{name}'. "
        f"Known tiers: {sorted(catalog.keys())}"
    )


def tier_from_subset_filename(filename: str) -> str | None:
    """Return the canonical tier name for a subset filename, or None if unknown."""
    fname = str(filename).strip()
    if fname.endswith(".jsonl"):
        fname = fname[: -len(".jsonl")]
    for tier, subsets in TIER_TO_SUBSETS.items():
        for subset in subsets:
            if subset[: -len(".jsonl")] == fname or subset == fname:
                return tier
    return None


def subset_files_for_tiers(tiers: list[str] | tuple[str, ...]) -> list[str]:
    """Flatten a list of tier names into their subset filenames, preserving order."""
    files: list[str] = []
    for name in tiers:
        tier = resolve_tier(name)
        files.extend(tier.subsets)
    return files