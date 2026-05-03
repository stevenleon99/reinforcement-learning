# Actor-Critic Methods

> Map section: [RL_LEARNING_MAP.md - Section 4.2](../RL_LEARNING_MAP.md#actor-critic-methods)

## What This Module Covers

Actor-Critic methods combine a policy network (actor) with a value network (critic). The critic reduces variance while the actor maintains direct policy optimization.

## Key Topics

- Advantage Actor-Critic (A2C / A3C)
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)
- Soft Actor-Critic (SAC) -- maximum entropy RL

## Key Algorithms

| Algorithm | Action Space | Key Feature |
|-----------|-------------|-------------|
| A2C/A3C | Discrete/Continuous | Parallel workers, advantage estimation |
| DDPG | Continuous | Deterministic policy, off-policy |
| TD3 | Continuous | Twin critics, delayed updates |
| SAC | Continuous | Entropy maximization, sample efficient |

## Suggested Exercises

- Implement A2C on CartPole
- Implement DDPG on Pendulum
- Implement SAC on HalfCheetah (MuJoCo)
- Compare TD3 vs SAC on continuous control benchmarks
