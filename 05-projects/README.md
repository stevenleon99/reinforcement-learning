# Practical Projects

> Map section: [RL_LEARNING_MAP.md - Section 5](../RL_LEARNING_MAP.md#5-practical-learning-path-hands-on-projects)

## What This Module Covers

Hands-on projects organized by difficulty, progressing from tabular RL to full LLM alignment with DPO/GRPO.

## Project Progression

| Stage | Project | Environment | Key Skills |
|-------|---------|-------------|------------|
| 1 | FrozenLake / Gridworld | `FrozenLake-v1` | Q-Learning, MDPs |
| 1 | Blackjack | `Blackjack-v1` | Monte Carlo methods |
| 2 | CartPole with DQN | `CartPole-v1` | DQN, replay buffers |
| 3 | LunarLander with PPO | `LunarLander-v3` | Policy gradients, PPO |
| 3 | MuJoCo with SAC | `HalfCheetah-v5` | Continuous control, SAC |
| 4 | Atari DQN | `ALE/Breakout-v5` | Vision-based RL, Rainbow |
| 5 | LLM Alignment | HuggingFace TRL | DPO, GRPO, RLHF |

## Quick Start

```bash
pip install gymnasium stable-baselines3 torch matplotlib
```

## Suggested Exercises

- Complete each stage sequentially
- Implement from scratch FIRST, then compare with CleanRL/SB3
- Write up: what worked, what didn't, hyperparameter sensitivity
