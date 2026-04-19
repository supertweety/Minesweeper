from __future__ import annotations

import json
import uuid


def hidden_board(size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple("?" for _ in range(size)) for _ in range(size))


def build_dashboard_payload(result) -> dict[str, object]:
    agent_names = list(result.scores.keys())
    board_size = result.config.board_size

    current_scores = {name: 0 for name in agent_names}
    current_stats = {
        name: {
            "safe": 0,
            "mine": 0,
            "timeout": 0,
            "crash": 0,
            "invalid_move": 0,
            "round_survived": 0,
        }
        for name in agent_names
    }

    rounds_payload = []
    timeline = []

    def append_snapshot(snapshot: dict[str, object]) -> int:
        snapshot["global_index"] = len(timeline)
        timeline.append(snapshot)
        return snapshot["global_index"]

    for round_outcome in result.rounds:
        round_snapshots = []
        start_snapshot = {
            "round_index": round_outcome.round_index,
            "turn_index": 0,
            "event_index": -1,
            "agent_name": None,
            "outcome": "round_start",
            "message": "Round start",
            "board": [list(row) for row in hidden_board(board_size)],
            "scores": dict(current_scores),
            "stats": json.loads(json.dumps(current_stats)),
        }
        append_snapshot(start_snapshot)
        round_snapshots.append(start_snapshot)

        for event_index, turn in enumerate(round_outcome.turn_records, start=1):
            current_scores = dict(turn.scores)
            if turn.outcome in current_stats[turn.agent_name]:
                current_stats[turn.agent_name][turn.outcome] += 1

            snapshot = {
                "round_index": round_outcome.round_index,
                "turn_index": event_index,
                "event_index": event_index - 1,
                "agent_name": turn.agent_name,
                "outcome": turn.outcome,
                "message": turn.message,
                "board": [list(row) for row in turn.board],
                "scores": dict(current_scores),
                "stats": json.loads(json.dumps(current_stats)),
            }
            append_snapshot(snapshot)
            round_snapshots.append(snapshot)

        for survivor in round_outcome.surviving_agents:
            current_stats[survivor]["round_survived"] += 1

        if round_snapshots:
            final_stats = json.loads(json.dumps(current_stats))
            round_snapshots[-1]["stats"] = final_stats
            timeline[round_snapshots[-1]["global_index"]]["stats"] = json.loads(
                json.dumps(final_stats)
            )

        rounds_payload.append(
            {
                "round_index": round_outcome.round_index,
                "completed": round_outcome.completed,
                "message": round_outcome.message,
                "winner_names": list(round_outcome.winner_names),
                "timed_out_agents": list(round_outcome.timed_out_agents),
                "mine_hit_agents": list(round_outcome.mine_hit_agents),
                "invalid_move_agents": list(round_outcome.invalid_move_agents),
                "crashed_agents": list(round_outcome.crashed_agents),
                "surviving_agents": list(round_outcome.surviving_agents),
                "snapshots": round_snapshots,
                "event_log": [
                    {
                        "turn_index": idx,
                        "agent_name": snapshot["agent_name"],
                        "outcome": snapshot["outcome"],
                        "message": snapshot["message"],
                        "scores": snapshot["scores"],
                    }
                    for idx, snapshot in enumerate(round_snapshots)
                ],
            }
        )

    score_series = {name: [] for name in agent_names}
    labels = []
    for snapshot in timeline:
        labels.append(f"R{snapshot['round_index']}T{snapshot['turn_index']}")
        for name in agent_names:
            score_series[name].append(snapshot["scores"][name])

    final_agent_stats = {
        name: timeline[-1]["stats"][name] if timeline else current_stats[name]
        for name in agent_names
    }

    return {
        "config": {
            "board_size": result.config.board_size,
            "num_mines": result.config.num_mines,
            "num_rounds": result.config.num_rounds,
            "turn_timeout_seconds": result.config.turn_timeout_seconds,
            "random_seed": result.config.random_seed,
        },
        "agent_names": agent_names,
        "scores": result.scores,
        "disqualified_agents": list(result.disqualified_agents),
        "rounds": rounds_payload,
        "timeline_labels": labels,
        "score_series": score_series,
        "final_agent_stats": final_agent_stats,
    }


def render_tournament_dashboard(payload: dict[str, object]) -> None:
    from IPython.display import HTML, display

    root_id = f"tournament-dashboard-{uuid.uuid4().hex}"
    payload_json = json.dumps(payload)
    html = f"""
    <div id="{root_id}" style="font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #0f172a;">
      <div style="display:flex; flex-wrap:wrap; gap:16px; align-items:flex-end; margin-bottom:16px;">
        <label style="display:flex; flex-direction:column; gap:6px; font-weight:600;">
          Round
          <select data-role="round-select" style="padding:8px 10px; border:1px solid #cbd5e1; border-radius:10px; background:white;"></select>
        </label>
        <label style="display:flex; flex:1 1 320px; flex-direction:column; gap:6px; font-weight:600; min-width:320px;">
          Turn
          <input data-role="turn-slider" type="range" min="0" max="0" value="0" style="width:100%;" />
        </label>
        <div style="display:flex; gap:8px; align-items:center;">
          <button data-role="prev-turn" style="padding:8px 14px; border-radius:10px; border:1px solid #cbd5e1; background:white; cursor:pointer;">Previous</button>
          <button data-role="next-turn" style="padding:8px 14px; border-radius:10px; border:1px solid #cbd5e1; background:white; cursor:pointer;">Next</button>
        </div>
        <div data-role="turn-label" style="font-weight:700; color:#334155;"></div>
      </div>

      <div style="display:grid; grid-template-columns:minmax(320px, 420px) minmax(340px, 1fr); gap:18px; align-items:start;">
        <div>
          <div data-role="board-panel"></div>
          <div data-role="event-card" style="margin-top:14px;"></div>
        </div>
        <div>
          <div data-role="score-cards" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px;"></div>
          <div data-role="round-summary" style="margin-top:14px;"></div>
          <div data-role="event-log" style="margin-top:14px;"></div>
        </div>
      </div>

      <div style="margin-top:24px; display:grid; grid-template-columns:1fr; gap:20px;">
        <div>
          <div style="font-size:20px; font-weight:800; margin-bottom:8px;">Cumulative score progression</div>
          <div data-role="score-chart"></div>
        </div>
        <div>
          <div style="font-size:20px; font-weight:800; margin-bottom:8px;">Cumulative agent statistics</div>
          <div data-role="stats-table"></div>
        </div>
      </div>
    </div>

    <script>
    (() => {{
      const payload = {payload_json};
      const root = document.getElementById({json.dumps(root_id)});
      if (!root) return;

      const roundSelect = root.querySelector('[data-role="round-select"]');
      const turnSlider = root.querySelector('[data-role="turn-slider"]');
      const prevTurn = root.querySelector('[data-role="prev-turn"]');
      const nextTurn = root.querySelector('[data-role="next-turn"]');
      const turnLabel = root.querySelector('[data-role="turn-label"]');
      const boardPanel = root.querySelector('[data-role="board-panel"]');
      const eventCard = root.querySelector('[data-role="event-card"]');
      const scoreCards = root.querySelector('[data-role="score-cards"]');
      const roundSummary = root.querySelector('[data-role="round-summary"]');
      const eventLog = root.querySelector('[data-role="event-log"]');
      const scoreChart = root.querySelector('[data-role="score-chart"]');
      const statsTable = root.querySelector('[data-role="stats-table"]');

      const numberColors = {{
        '1': '#2563eb',
        '2': '#16a34a',
        '3': '#dc2626',
        '4': '#7c3aed',
        '5': '#b45309',
        '6': '#0f766e',
        '7': '#111827',
        '8': '#475569'
      }};
      const lineColors = ['#2563eb', '#16a34a', '#dc2626', '#7c3aed', '#ea580c', '#0891b2'];

      payload.rounds.forEach((round, index) => {{
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `Round ${{round.round_index}}`;
        roundSelect.appendChild(option);
      }});

      const state = {{ roundIndex: 0, turnIndex: 0 }};

      function currentRound() {{
        return payload.rounds[state.roundIndex];
      }}

      function currentSnapshot() {{
        return currentRound().snapshots[state.turnIndex];
      }}

      function currentGlobalIndex() {{
        return currentSnapshot().global_index;
      }}

      function setRound(index) {{
        state.roundIndex = index;
        state.turnIndex = 0;
        const round = currentRound();
        turnSlider.max = String(Math.max(0, round.snapshots.length - 1));
        turnSlider.value = '0';
        render();
      }}

      function setTurn(index) {{
        state.turnIndex = index;
        turnSlider.value = String(index);
        render();
      }}

      function formatBoard(board) {{
        const size = board.length;
        const cells = [];
        for (let row = 0; row < size; row++) {{
          for (let col = 0; col < size; col++) {{
            const value = board[row][col];
            let background = '#ffffff';
            let color = '#0f172a';
            let label = value === ' ' ? '·' : value;
            if (value === '?') {{
              background = '#1d3557';
              color = '#f8fafc';
            }} else if (value === ' ') {{
              background = '#f1f5f9';
              color = '#94a3b8';
            }} else {{
              background = '#ffffff';
              color = numberColors[value] || '#0f172a';
            }}
            cells.push(`<div style="width:52px;height:52px;display:flex;align-items:center;justify-content:center;border-radius:12px;border:1px solid #cbd5e1;background:${{background}};color:${{color}};font-size:24px;font-weight:800;box-shadow:0 8px 18px rgba(15,23,42,0.06);">${{label}}</div>`);
          }}
        }}
        return `<div style="display:inline-block;padding:18px;background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);border:1px solid #e2e8f0;border-radius:18px;">` +
          `<div style="display:grid;grid-template-columns:repeat(${{size}},52px);gap:8px;">${{cells.join('')}}</div>` +
          `</div>`;
      }}

      function renderBoard() {{
        const snapshot = currentSnapshot();
        const round = currentRound();
        boardPanel.innerHTML = `
          <div style="font-size:22px;font-weight:800;margin-bottom:10px;">Board state</div>
          <div style="font-size:14px;color:#475569;margin-bottom:12px;">${{round.message}}</div>
          ${{formatBoard(snapshot.board)}}
        `;
      }}

      function renderEventCard() {{
        const snapshot = currentSnapshot();
        const actor = snapshot.agent_name ? snapshot.agent_name : 'system';
        eventCard.innerHTML = `
          <div style="padding:16px 18px;border:1px solid #e2e8f0;border-radius:16px;background:#fff;">
            <div style="font-size:18px;font-weight:800;margin-bottom:10px;">Current event</div>
            <div style="display:grid;grid-template-columns:120px 1fr;gap:8px 12px;font-size:14px;">
              <div style="color:#64748b;font-weight:700;">Actor</div><div>${{actor}}</div>
              <div style="color:#64748b;font-weight:700;">Outcome</div><div>${{snapshot.outcome}}</div>
              <div style="color:#64748b;font-weight:700;">Message</div><div>${{snapshot.message}}</div>
              <div style="color:#64748b;font-weight:700;">Global step</div><div>${{snapshot.global_index}}</div>
            </div>
          </div>
        `;
      }}

      function renderScoreCards() {{
        const snapshot = currentSnapshot();
        scoreCards.innerHTML = payload.agent_names.map((agent, index) => {{
          const score = snapshot.scores[agent];
          const accent = lineColors[index % lineColors.length];
          return `
            <div style="padding:16px 18px;border:1px solid #e2e8f0;border-radius:16px;background:#fff;">
              <div style="font-size:14px;color:#64748b;font-weight:700;">${{agent}}</div>
              <div style="font-size:32px;font-weight:900;color:${{accent}};margin-top:6px;">${{score}}</div>
              <div style="font-size:12px;color:#475569;">current cumulative points</div>
            </div>
          `;
        }}).join('');
      }}

      function renderRoundSummary() {{
        const round = currentRound();
        roundSummary.innerHTML = `
          <div style="padding:16px 18px;border:1px solid #e2e8f0;border-radius:16px;background:#fff;">
            <div style="font-size:18px;font-weight:800;margin-bottom:10px;">Round summary</div>
            <div style="display:grid;grid-template-columns:140px 1fr;gap:8px 12px;font-size:14px;">
              <div style="color:#64748b;font-weight:700;">Completed</div><div>${{round.completed}}</div>
              <div style="color:#64748b;font-weight:700;">Winners</div><div>${{round.winner_names.length ? round.winner_names.join(', ') : 'none'}}</div>
              <div style="color:#64748b;font-weight:700;">Timed out</div><div>${{round.timed_out_agents.length ? round.timed_out_agents.join(', ') : 'none'}}</div>
              <div style="color:#64748b;font-weight:700;">Hit mine</div><div>${{round.mine_hit_agents.length ? round.mine_hit_agents.join(', ') : 'none'}}</div>
              <div style="color:#64748b;font-weight:700;">Invalid move</div><div>${{round.invalid_move_agents.length ? round.invalid_move_agents.join(', ') : 'none'}}</div>
              <div style="color:#64748b;font-weight:700;">Crashed</div><div>${{round.crashed_agents.length ? round.crashed_agents.join(', ') : 'none'}}</div>
            </div>
          </div>
        `;
      }}

      function renderEventLog() {{
        const round = currentRound();
        const rows = round.event_log.map((entry, index) => {{
          const active = index === state.turnIndex ? 'background:#eff6ff;' : '';
          const actor = entry.agent_name || 'system';
          return `
            <tr style="${{active}}">
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{entry.turn_index}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{actor}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{entry.outcome}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{entry.message}}</td>
            </tr>
          `;
        }}).join('');
        eventLog.innerHTML = `
          <div style="padding:16px 18px;border:1px solid #e2e8f0;border-radius:16px;background:#fff;">
            <div style="font-size:18px;font-weight:800;margin-bottom:10px;">Round event log</div>
            <div style="max-height:280px;overflow:auto;">
              <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <thead>
                  <tr>
                    <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Turn</th>
                    <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Agent</th>
                    <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Outcome</th>
                    <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Message</th>
                  </tr>
                </thead>
                <tbody>${{rows}}</tbody>
              </table>
            </div>
          </div>
        `;
      }}

      function renderScoreChart() {{
        const width = 920;
        const height = 260;
        const left = 42;
        const right = 24;
        const top = 20;
        const bottom = 34;
        const currentIndex = currentGlobalIndex();
        const allValues = [].concat(...Object.values(payload.score_series));
        const maxValue = Math.max(1, ...allValues);
        const xMax = Math.max(1, payload.timeline_labels.length - 1);

        function x(index) {{
          return left + (index / xMax) * (width - left - right);
        }}
        function y(value) {{
          return height - bottom - (value / maxValue) * (height - top - bottom);
        }}

        const lines = payload.agent_names.map((agent, index) => {{
          const points = payload.score_series[agent].map((value, step) => `${{x(step).toFixed(2)}},${{y(value).toFixed(2)}}`).join(' ');
          const color = lineColors[index % lineColors.length];
          return `
            <polyline points="${{points}}" fill="none" stroke="${{color}}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></polyline>
          `;
        }}).join('');

        const legend = payload.agent_names.map((agent, index) => `
          <div style="display:flex;align-items:center;gap:8px;margin-right:16px;">
            <span style="display:inline-block;width:14px;height:14px;border-radius:999px;background:${{lineColors[index % lineColors.length]}};"></span>
            <span>${{agent}}</span>
          </div>
        `).join('');

        scoreChart.innerHTML = `
          <svg viewBox="0 0 ${{width}} ${{height}}" width="100%" style="background:#fff;border:1px solid #e2e8f0;border-radius:16px;">
            <line x1="${{left}}" y1="${{height - bottom}}" x2="${{width - right}}" y2="${{height - bottom}}" stroke="#cbd5e1" stroke-width="2"></line>
            <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{height - bottom}}" stroke="#cbd5e1" stroke-width="2"></line>
            ${{lines}}
            <line x1="${{x(currentIndex)}}" y1="${{top}}" x2="${{x(currentIndex)}}" y2="${{height - bottom}}" stroke="#0f172a" stroke-dasharray="6 4" stroke-width="2"></line>
            <text x="${{left + 6}}" y="${{top + 10}}" font-size="12" fill="#475569">${{maxValue}}</text>
            <text x="${{left + 6}}" y="${{height - bottom - 4}}" font-size="12" fill="#475569">0</text>
          </svg>
          <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;font-size:14px;">${{legend}}</div>
        `;
      }}

      function renderStatsTable() {{
        const snapshot = currentSnapshot();
        const rows = payload.agent_names.map((agent) => {{
          const stats = snapshot.stats[agent];
          const isDisqualified = payload.disqualified_agents.includes(agent);
          return `
            <tr>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;font-weight:700;">${{agent}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{snapshot.scores[agent]}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{stats.safe}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{stats.mine}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{stats.timeout}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{stats.crash}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{stats.invalid_move}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{stats.round_survived}}</td>
              <td style="padding:8px 10px;border-bottom:1px solid #e2e8f0;">${{isDisqualified ? 'yes' : 'no'}}</td>
            </tr>
          `;
        }}).join('');
        statsTable.innerHTML = `
          <div style="padding:16px 18px;border:1px solid #e2e8f0;border-radius:16px;background:#fff;overflow:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
              <thead>
                <tr>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Agent</th>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Points</th>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Safe moves</th>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Mine hits</th>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Timeouts</th>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Crashes</th>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Invalid moves</th>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Rounds survived</th>
                  <th style="text-align:left;padding:8px 10px;border-bottom:1px solid #cbd5e1;">Disqualified</th>
                </tr>
              </thead>
              <tbody>${{rows}}</tbody>
            </table>
          </div>
        `;
      }}

      function render() {{
        const round = currentRound();
        const snapshot = currentSnapshot();
        turnLabel.textContent = `Round ${{round.round_index}}, step ${{state.turnIndex}} / ${{round.snapshots.length - 1}}`;
        roundSelect.value = String(state.roundIndex);
        prevTurn.disabled = state.turnIndex === 0;
        nextTurn.disabled = state.turnIndex === round.snapshots.length - 1;
        renderBoard();
        renderEventCard();
        renderScoreCards();
        renderRoundSummary();
        renderEventLog();
        renderScoreChart();
        renderStatsTable();
      }}

      roundSelect.addEventListener('change', (event) => {{
        setRound(Number(event.target.value));
      }});
      turnSlider.addEventListener('input', (event) => {{
        setTurn(Number(event.target.value));
      }});
      prevTurn.addEventListener('click', () => {{
        if (state.turnIndex > 0) setTurn(state.turnIndex - 1);
      }});
      nextTurn.addEventListener('click', () => {{
        const last = currentRound().snapshots.length - 1;
        if (state.turnIndex < last) setTurn(state.turnIndex + 1);
      }});

      setRound(0);
    }})();
    </script>
    """
    display(HTML(html))
