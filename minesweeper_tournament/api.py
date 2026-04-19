from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AgentState(str, Enum):
    ACTIVE = "active"
    TIMED_OUT = "timed_out"
    HIT_MINE = "hit_mine"
    INVALID_MOVE = "invalid_move"
    CRASHED = "crashed"
    DISQUALIFIED = "disqualified"


@dataclass(frozen=True)
class Action:
    row: int
    col: int


@dataclass(frozen=True)
class Observation:
    round_index: int
    turn_index: int
    board: tuple[tuple[str, ...], ...]
    legal_actions: tuple[Action, ...]
    scores: dict[str, int]
    active_agents: tuple[str, ...]
    disqualified_agents: tuple[str, ...]


@dataclass(frozen=True)
class RoundContext:
    round_index: int
    board_size: int
    num_mines: int
    agent_order: tuple[str, ...]
    total_rounds: int


@dataclass(frozen=True)
class TurnRecord:
    round_index: int
    turn_index: int
    agent_name: str
    action: Optional[Action]
    outcome: str
    message: str
    board: tuple[tuple[str, ...], ...]
    scores: dict[str, int]


@dataclass(frozen=True)
class RoundOutcome:
    round_index: int
    winner_names: tuple[str, ...]
    turn_records: tuple[TurnRecord, ...]
    surviving_agents: tuple[str, ...]
    timed_out_agents: tuple[str, ...]
    mine_hit_agents: tuple[str, ...]
    invalid_move_agents: tuple[str, ...]
    crashed_agents: tuple[str, ...]
    final_board: tuple[tuple[str, ...], ...]
    completed: bool
    message: str


@dataclass(frozen=True)
class TournamentConfig:
    board_size: int = 5
    num_mines: int = 5
    num_rounds: int = 10
    turn_timeout_seconds: float = 1.0
    random_seed: Optional[int] = None
    reshuffle_each_round: bool = True


@dataclass(frozen=True)
class TournamentResult:
    config: TournamentConfig
    scores: dict[str, int]
    rounds: tuple[RoundOutcome, ...]
    disqualified_agents: tuple[str, ...]


class MinesweeperAgent:
    """
    Base class for tournament agents.

    Override `choose_action` in your agent class. The other hooks are optional
    and can help with debugging or per-round initialization.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def new_round(self, context: RoundContext) -> None:
        """Called once at the start of each round."""

    def choose_action(self, observation: Observation) -> Action:
        raise NotImplementedError

    def round_finished(self, outcome: RoundOutcome) -> None:
        """Called once after a round ends."""


@dataclass(frozen=True)
class _WorkerRequest:
    command: str
    payload: object = None


@dataclass(frozen=True)
class _WorkerResponse:
    ok: bool
    result: object = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class _AgentLifecycleState:
    timed_out: bool = False
    disqualified: bool = False
    last_error: Optional[str] = None
    round_state: AgentState = AgentState.ACTIVE
    total_points: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
