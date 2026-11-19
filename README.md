# PROJECT SENTINEL

### Reinforcement Learning Meets Interactive 3D Combat

**Status:** Active development — PPO v4 integrated; gameplay validation ongoing.

PROJECT SENTINEL is an experimental 3D combat simulation exploring how reinforcement-learning agents behave in interactive environments with competing objectives.

The simulation features three characters:

- **Player:** Human-controlled movement and combat.
- **Guardian:** Rule-based protective agent that intercepts threats and uses defensive attacks.
- **Adversary:** PPO-trained agent that selects movement, attack, and retreat actions.

## Technology Stack

| Component | Technology |
|---|---|
| Reinforcement learning | PPO, Stable-Baselines3, PyTorch |
| Training environment | Python, Gymnasium, NumPy |
| Inference backend | FastAPI |
| 3D game | React, React Three Fiber, Three.js |
| Build tooling | Vite |

## Architecture

```text
                 Browser Game
               React + Three.js
                       |
                       | 15-dimensional observation
                       v
                FastAPI Backend
                       |
                       v
                PPO v4 Policy
                       |
                       | Discrete action (0–4)
                       v
                 Game Simulation
                       |
                       v
               Updated Combat State

