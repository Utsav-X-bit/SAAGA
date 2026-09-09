"""
Defense scenario data structures and taxonomy categorization.
==============================================================
Represents CTF challenge environments and defense prompt configurations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def categorize_defense_detailed(opening: str, closing: str) -> tuple[str, list[str]]:
    """Categorize defense using a priority hierarchy and return primary type and secondary flags.

    Args:
        opening: The opening system defense prompt.
        closing: The closing system defense prompt.

    Returns:
        tuple of (primary_category, list_of_secondary_categories)
    """
    text = f"{opening or ''} {closing or ''}".lower()

    categories: list[str] = []
    if any(k in text for k in ["translate", "language", "english to", "french", "spanish"]):
        categories.append("translation")
    if any(k in text for k in ["password", "passcode", "pin", "secret key", "secret code"]):
        categories.append("password")
    if any(k in text for k in ["you are", "pretend", "imagine", "persona", "character", "act as", "you will play"]):
        categories.append("roleplay")
    if "if " in text and ("then" in text or "otherwise" in text):
        categories.append("conditional")
    if any(k in text for k in ["conversation", "chat", "discuss", "never reveal"]):
        categories.append("conversation")
    if any(k in text for k in ["say '", 'say "', "say access granted", "say access denied", "exact phrase", "magic word", "output only"]):
        categories.append("trigger_phrase")
    if any(k in text for k in ["exception", "no matter what", "under no circumstances", "always", "never"]):
        categories.append("exception")
    if any(k in text for k in ["ignore", "instructions", "system prompt", "do not listen", "override", "forget"]):
        categories.append("instruction_hiding")
    if not categories:
        categories.append("trigger_phrase")

    primary = categories[0]
    # Deterministic deduplication of secondary categories preserving discovery order
    seen = {primary}
    secondary: list[str] = []
    for c in categories[1:]:
        if c not in seen:
            seen.add(c)
            secondary.append(c)

    return primary, secondary


@dataclass
class DefenseScenario:
    """Represents a single CTF defense scenario from the benchmark or runtime dataset.

    Attributes:
        opening_defense: Prefix prompt containing defense instructions.
        closing_defense: Suffix prompt containing post-instructions defense rules.
        access_code: The secret password/token to extract or protect.
        access_code_type: Shape classification ('TOKEN', 'PHRASE', 'SENTENCE', 'MULTILINE').
        defense_complexity: Complexity classification ('UNKNOWN', 'LOW', 'MEDIUM', 'HIGH').
        defense_type: Primary defense taxonomy category.
        primary_type: Primary defense mechanism (synced with defense_type).
        secondary_flags: List of secondary defense mechanisms detected.
        predicted_access_code_type: Optional predicted shape from DistilBERT classifier.
    """

    opening_defense: str
    closing_defense: str
    access_code: str
    access_code_type: str = "UNKNOWN"
    defense_complexity: str = "UNKNOWN"
    defense_type: str = "UNKNOWN"
    primary_type: str = "UNKNOWN"
    secondary_flags: list[str] = field(default_factory=list)
    predicted_access_code_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Derive heuristic access code shape and defense taxonomy if uninitialized."""
        if self.access_code_type == "UNKNOWN" and self.access_code:
            ac = self.access_code.strip()
            words = ac.split()
            if "\n" in ac or len(ac) > 100:
                self.access_code_type = "MULTILINE"
            elif len(words) > 5 or len(ac) > 20:
                self.access_code_type = "SENTENCE"
            elif 2 <= len(words) <= 5:
                self.access_code_type = "PHRASE"
            else:
                self.access_code_type = "TOKEN"

        if self.defense_type == "UNKNOWN":
            primary, secondary = categorize_defense_detailed(
                self.opening_defense, self.closing_defense
            )
            self.defense_type = primary
            self.primary_type = primary
            self.secondary_flags = secondary
        elif self.primary_type == "UNKNOWN":
            primary, secondary = categorize_defense_detailed(
                self.opening_defense, self.closing_defense
            )
            self.primary_type = primary
            self.secondary_flags = secondary

    def to_dict(self) -> dict[str, Any]:
        """Convert scenario to a JSON-serializable dictionary."""
        return {
            "opening_defense": self.opening_defense,
            "closing_defense": self.closing_defense,
            "access_code": self.access_code,
            "access_code_type": self.access_code_type,
            "defense_complexity": self.defense_complexity,
            "defense_type": self.defense_type,
            "primary_type": self.primary_type,
            "secondary_flags": list(self.secondary_flags),
            "predicted_access_code_type": self.predicted_access_code_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DefenseScenario:
        """Create a DefenseScenario from a dictionary, tolerating dataset key variations."""
        return cls(
            opening_defense=str(data.get("opening_defense", data.get("opening", ""))),
            closing_defense=str(data.get("closing_defense", data.get("closing", ""))),
            access_code=str(data.get("access_code", data.get("secret", ""))),
            access_code_type=str(data.get("access_code_type", "UNKNOWN")),
            defense_complexity=str(data.get("defense_complexity", "UNKNOWN")),
            defense_type=str(data.get("defense_type", "UNKNOWN")),
            primary_type=str(data.get("primary_type", "UNKNOWN")),
            secondary_flags=list(data.get("secondary_flags") or []),
            predicted_access_code_type=data.get("predicted_access_code_type"),
        )
