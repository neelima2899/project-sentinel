# SENTINEL — Reinforcement Learning Experiment (starter)

This is a **separate simplified combat simulator**, not yet connected to the React/Three.js game. The Python RL agent uses PPO and a five-action discrete action space. No trained model or results are bundled. Do not claim performance improvement until experiments have been run.

## Windows PowerShell setup

From the extracted `sentinel-rl-starter` directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python rl/evaluate.py --episodes 100
python rl/train.py --steps 100000
python rl/evaluate.py --episodes 100
```

If `py -3.12` is unavailable, install Python 3.12 or use an existing compatible Python environment. The initial evaluation runs the rule-based baseline; after training, it compares both. Training may take time on CPU.

## Observation and action design

Observations: player and Guardian relative positions to enemy (4 values), normalized player/Guardian/enemy health (3), enemy-to-player and enemy-to-Guardian distances (2), elapsed episode fraction (1). All are bounded in [-1,1].

Actions: 0 approach player, 1 approach Guardian, 2 attack player, 3 attack Guardian, 4 retreat.

Rewards: positive for damage inflicted on player/Guardian, negative for damage taken, small time penalty, terminal win/loss reward. The rewards are *design choices* and may need tuning.

## Important limitations

- The simulated Player uses a fixed approach-and-attack strategy. It does not yet represent varied human playstyles.
- The Guardian uses a simplified attack rule, not the complete browser interception code.
- The simulation is step-based; browser gameplay is frame-based. Direct transfer is **not** validated.
- `evaluation.csv` contains outcomes and action counts, but action counts are not explanations of internal policy reasoning.
- The baseline and PPO should be evaluated over multiple player strategies and seeds before reporting a meaningful comparison.
- A trained `.zip` model cannot simply be dropped into the browser. A separate policy export/inference implementation is required, followed by parity tests.

## Next engineering milestones

1. Move shared combat constants into a specification and align simulator with browser mechanics.
2. Add scripted player styles (inactive, aggressive, retreating) and randomized initial positions.
3. Run multi-seed PPO training and held-out evaluations; chart win rate and damage.
4. Export policy weights to browser-compatible inference, validate action parity with Python.
5. Add a Rule-based/PPO mode switch, telemetry, visual polish, deployment and documentation.
