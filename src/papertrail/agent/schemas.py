"""The case summary the agent must produce. Typology names are the ones in HI-Small_Patterns.txt (checked in Phase 1),
plus UNCLEAR for when the evidence does not support a pattern."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Typology = Literal["FAN-OUT", "FAN-IN", "CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "STACK", "BIPARTITE", "RANDOM", "UNCLEAR"]
Action = Literal["escalate_SAR", "enhanced_review", "close_false_positive"]
TYPOLOGIES: tuple[str, ...] = Typology.__args__
ACTIONS: tuple[str, ...] = Action.__args__


class Evidence(BaseModel):
    claim: str = Field(min_length=3, max_length=400)
    source_id: str                                    # must match a tool output


class CaseSummary(BaseModel):
    account_id: str
    risk_score: float
    suspected_typology: Typology
    evidence: list[Evidence] = Field(min_length=1, max_length=8)
    narrative: str = Field(min_length=10, max_length=1500)
    recommended_action: Action
