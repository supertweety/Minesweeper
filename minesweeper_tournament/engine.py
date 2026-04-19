from __future__ import annotations

import multiprocessing
import random
import traceback
from multiprocessing.connection import Connection
from typing import Callable

from mine_board import MineBoard

from .api import (
    Action,
    AgentState,
    MinesweeperAgent,
    Observation,
    RoundContext,
    RoundOutcome,
    TournamentConfig,
    TournamentResult,
    TurnRecord,
    _AgentLifecycleState,
    _WorkerRequest,
    _WorkerResponse,
)


AgentFactory = Callable[[], MinesweeperAgent]


def _board_to_tuple(board: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(row) for row in board)


def _worker_main(factory: AgentFactory, conn: Connection) -> None:
    try:
        agent = factory()
    except Exception as exc:  # pragma: no cover - defensive bootstrap path
        conn.send(
            _WorkerResponse(
                ok=False,
                error_type=exc.__class__.__name__,
                error_message="".join(
                    traceback.format_exception_only(exc.__class__, exc)
                ).strip(),
            )
        )
        conn.close()
        return

    while True:
        try:
            request = conn.recv()
        except EOFError:
            break

        if not isinstance(request, _WorkerRequest):
            conn.send(
                _WorkerResponse(
                    ok=False,
                    error_type="ProtocolError",
                    error_message="worker received an invalid request object",
                )
            )
            continue

        try:
            if request.command == "new_round":
                agent.new_round(request.payload)
                conn.send(_WorkerResponse(ok=True, result=None))
            elif request.command == "choose_action":
                action = agent.choose_action(request.payload)
                conn.send(_WorkerResponse(ok=True, result=action))
            elif request.command == "round_finished":
                agent.round_finished(request.payload)
                conn.send(_WorkerResponse(ok=True, result=None))
            elif request.command == "shutdown":
                conn.send(_WorkerResponse(ok=True, result=None))
                break
            else:
                conn.send(
                    _WorkerResponse(
                        ok=False,
                        error_type="ProtocolError",
                        error_message=f"unknown worker command: {request.command}",
                    )
                )
        except Exception as exc:  # pragma: no cover - exercised via integration
            conn.send(
                _WorkerResponse(
                    ok=False,
                    error_type=exc.__class__.__name__,
                    error_message="".join(traceback.format_exception(exc)).strip(),
                )
            )
            break

    conn.close()


class _AgentSession:
    def __init__(self, factory: AgentFactory, mp_context: multiprocessing.context.BaseContext):
        self._factory = factory
        self._mp_context = mp_context
        self._process = None
        self._parent_conn = None

    def start(self) -> None:
        parent_conn, child_conn = self._mp_context.Pipe()
        self._process = self._mp_context.Process(
            target=_worker_main,
            args=(self._factory, child_conn),
            daemon=True,
        )
        self._process.start()
        child_conn.close()
        self._parent_conn = parent_conn

    def call(self, command: str, payload: object, timeout_seconds: float) -> _WorkerResponse:
        if self._process is None or self._parent_conn is None:
            raise RuntimeError("session has not been started")

        self._parent_conn.send(_WorkerRequest(command=command, payload=payload))
        if not self._parent_conn.poll(timeout_seconds):
            self.terminate()
            return _WorkerResponse(
                ok=False,
                error_type="TimeoutError",
                error_message=f"agent exceeded {timeout_seconds:.3f}s on {command}",
            )
        return self._parent_conn.recv()

    def terminate(self) -> None:
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except OSError:
                pass
            self._parent_conn = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=0.2)
            self._process = None

    def close_gracefully(self, timeout_seconds: float) -> _WorkerResponse:
        if self._process is None:
            return _WorkerResponse(ok=True, result=None)
        response = self.call("shutdown", None, timeout_seconds)
        self.terminate()
        return response


class TournamentRunner:
    def __init__(
        self,
        agent_factories: dict[str, AgentFactory],
        config: TournamentConfig,
    ):
        if not agent_factories:
            raise ValueError("at least one agent must be provided")
        if config.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")

        self._agent_factories = dict(agent_factories)
        self._config = config
        self._rng = random.Random(config.random_seed)
        self._mp_context = self._build_mp_context()

    def _build_mp_context(self) -> multiprocessing.context.BaseContext:
        available_methods = multiprocessing.get_all_start_methods()
        if "fork" in available_methods:
            return multiprocessing.get_context("fork")
        return multiprocessing.get_context()

    def run(self) -> TournamentResult:
        scores = {name: 0 for name in self._agent_factories}
        lifecycle = {
            name: _AgentLifecycleState(total_points=0) for name in self._agent_factories
        }
        round_results: list[RoundOutcome] = []

        for round_index in range(1, self._config.num_rounds + 1):
            active_names = list(lifecycle.keys())
            if len(active_names) == 0:
                break

            order = list(active_names)
            if self._config.reshuffle_each_round:
                self._rng.shuffle(order)

            outcome = self._run_round(round_index, order, scores, lifecycle)
            round_results.append(outcome)

        result = TournamentResult(
            config=self._config,
            scores=dict(scores),
            rounds=tuple(round_results),
            disqualified_agents=(),
        )
        return result

    def _run_round(
        self,
        round_index: int,
        agent_order: list[str],
        scores: dict[str, int],
        lifecycle: dict[str, _AgentLifecycleState],
    ) -> RoundOutcome:
        board_seed = self._rng.randrange(0, 10**9)
        board_rng = random.Random(board_seed)
        board = MineBoard(
            size=self._config.board_size,
            num_mines=self._config.num_mines,
            rng=board_rng,
        )

        sessions: dict[str, _AgentSession] = {}
        active_in_round: list[str] = []
        timed_out_agents: list[str] = []
        mine_hit_agents: list[str] = []
        invalid_move_agents: list[str] = []
        crashed_agents: list[str] = []
        turn_records: list[TurnRecord] = []
        detonated_mines: set[tuple[int, int]] = set()

        for name in agent_order:
            session = _AgentSession(self._agent_factories[name], self._mp_context)
            session.start()
            context = RoundContext(
                round_index=round_index,
                board_size=self._config.board_size,
                num_mines=self._config.num_mines,
                agent_order=tuple(agent_order),
                total_rounds=self._config.num_rounds,
            )
            response = session.call(
                "new_round",
                context,
                self._config.turn_timeout_seconds,
            )
            if not response.ok:
                session.terminate()
                if response.error_type == "TimeoutError":
                    lifecycle[name].round_state = AgentState.TIMED_OUT
                    timed_out_agents.append(name)
                else:
                    lifecycle[name].round_state = AgentState.CRASHED
                    lifecycle[name].last_error = response.error_message
                    crashed_agents.append(name)
                continue

            sessions[name] = session
            active_in_round.append(name)
            lifecycle[name].round_state = AgentState.ACTIVE

        turn_index = 0
        while not board.is_solved() and active_in_round:
            current_name = active_in_round[turn_index % len(active_in_round)]
            observation = self._build_observation(
                round_index=round_index,
                turn_index=turn_index,
                board=board,
                scores=scores,
                active_agents=active_in_round,
                lifecycle=lifecycle,
                detonated_mines=detonated_mines,
            )

            response = sessions[current_name].call(
                "choose_action",
                observation,
                self._config.turn_timeout_seconds,
            )
            if not response.ok:
                if response.error_type == "TimeoutError":
                    lifecycle[current_name].round_state = AgentState.TIMED_OUT
                    timed_out_agents.append(current_name)
                    message = response.error_message or "agent timed out"
                    outcome = "timeout"
                else:
                    lifecycle[current_name].round_state = AgentState.CRASHED
                    lifecycle[current_name].last_error = response.error_message
                    crashed_agents.append(current_name)
                    message = response.error_message or "agent crashed"
                    outcome = "crash"

                turn_records.append(
                    TurnRecord(
                        round_index=round_index,
                        turn_index=turn_index,
                        agent_name=current_name,
                        action=None,
                        outcome=outcome,
                        message=message,
                        board=_board_to_tuple(board.board()),
                        scores=dict(scores),
                    )
                )
                sessions[current_name].terminate()
                active_in_round.remove(current_name)
                continue

            action = response.result
            action_validation_error = self._validate_action(
                action,
                board,
                detonated_mines=detonated_mines,
            )
            if action_validation_error is not None:
                invalid_move_agents.append(current_name)
                lifecycle[current_name].round_state = AgentState.INVALID_MOVE
                turn_records.append(
                    TurnRecord(
                        round_index=round_index,
                        turn_index=turn_index,
                        agent_name=current_name,
                        action=action if isinstance(action, Action) else None,
                        outcome="invalid_move",
                        message=action_validation_error,
                        board=_board_to_tuple(board.board()),
                        scores=dict(scores),
                    )
                )
                sessions[current_name].terminate()
                active_in_round.remove(current_name)
                continue

            hit_safe_tile = board.perform_action(action.row, action.col)
            if hit_safe_tile:
                turn_records.append(
                    TurnRecord(
                        round_index=round_index,
                        turn_index=turn_index,
                        agent_name=current_name,
                        action=action,
                        outcome="safe",
                        message="safe tile uncovered",
                        board=_board_to_tuple(board.board()),
                        scores=dict(scores),
                    )
                )
                turn_index += 1
                continue

            mine_hit_agents.append(current_name)
            lifecycle[current_name].round_state = AgentState.HIT_MINE
            detonated_mines.add((action.row, action.col))
            for other_name in active_in_round:
                if other_name != current_name:
                    scores[other_name] += 1
                    lifecycle[other_name].total_points = scores[other_name]

            turn_records.append(
                TurnRecord(
                    round_index=round_index,
                    turn_index=turn_index,
                    agent_name=current_name,
                    action=action,
                    outcome="mine",
                    message="agent clicked on a mine; all other active agents scored",
                    board=_board_to_tuple(board.board()),
                    scores=dict(scores),
                )
            )
            sessions[current_name].terminate()
            active_in_round.remove(current_name)

        if board.is_solved():
            winner_names = tuple(active_in_round)
            message = "all safe tiles were uncovered"
            completed = True
        elif not active_in_round:
            winner_names = ()
            message = "round ended because no active agents remained"
            completed = False
        else:
            winner_names = tuple(active_in_round)
            message = "round ended unexpectedly"
            completed = False

        round_outcome = RoundOutcome(
            round_index=round_index,
            winner_names=winner_names,
            turn_records=tuple(turn_records),
            surviving_agents=tuple(active_in_round),
            timed_out_agents=tuple(timed_out_agents),
            mine_hit_agents=tuple(mine_hit_agents),
            invalid_move_agents=tuple(invalid_move_agents),
            crashed_agents=tuple(crashed_agents),
            final_board=_board_to_tuple(board.board()),
            completed=completed,
            message=message,
        )

        for name, session in sessions.items():
            if name in active_in_round:
                response = session.call(
                    "round_finished",
                    round_outcome,
                    self._config.turn_timeout_seconds,
                )
                if not response.ok and response.error_type != "TimeoutError":
                    lifecycle[name].round_state = AgentState.CRASHED
                    lifecycle[name].last_error = response.error_message
                session.terminate()

        return round_outcome

    def _build_observation(
        self,
        round_index: int,
        turn_index: int,
        board: MineBoard,
        scores: dict[str, int],
        active_agents: list[str],
        lifecycle: dict[str, _AgentLifecycleState],
        detonated_mines: set[tuple[int, int]],
    ) -> Observation:
        legal_actions = []
        for row_index in range(board.size()):
            for col_index in range(board.size()):
                if (
                    board.is_hidden(row_index, col_index)
                    and (row_index, col_index) not in detonated_mines
                ):
                    legal_actions.append(Action(row=row_index, col=col_index))
        return Observation(
            round_index=round_index,
            turn_index=turn_index,
            board=_board_to_tuple(board.board()),
            legal_actions=tuple(legal_actions),
            scores=dict(scores),
            active_agents=tuple(active_agents),
            disqualified_agents=tuple(
                name for name, state in lifecycle.items() if state.disqualified
            ),
        )

    def _validate_action(
        self,
        action: object,
        board: MineBoard,
        detonated_mines: set[tuple[int, int]],
    ) -> str | None:
        if not isinstance(action, Action):
            return "agent must return an Action(row=..., col=...) instance"
        if not (0 <= action.row < board.size() and 0 <= action.col < board.size()):
            return "selected action is outside the board"
        if (action.row, action.col) in detonated_mines:
            return "selected action targets an already detonated mine"
        if not board.is_hidden(action.row, action.col):
            return "selected action targets a revealed cell"
        return None
