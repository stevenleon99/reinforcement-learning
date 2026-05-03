# Deep RL Algorithms

> Map section: [RL_LEARNING_MAP.md - Section 4.2](../../RL_LEARNING_MAP.md#42-deep-rl)

## What This Module Covers

Deep RL algorithms that use neural networks as function approximators: DQN for discrete actions, Policy Gradients for continuous control, and Actor-Critic methods that combine both.

## Submodules

| Algorithm Family | Folder | Key Methods |
|-----------------|--------|-------------|
| Deep Q-Networks | [./dqn/](./dqn/) | DQN, Double DQN, Dueling DQN, Rainbow |
| Policy Gradient | [./policy-gradient/](./policy-gradient/) | REINFORCE, baseline reduction |
| Actor-Critic | [./actor-critic/](./actor-critic/) | A2C, A3C, DDPG, TD3, SAC |

## Suggested Resources

- Sutton & Barto Ch.9-10, 13 (Book)
- CleanRL single-file implementations (GitHub)
- Spinning Up in Deep RL (OpenAI)

## Suggested Exercises

- DQN on CartPole (from scratch)
- REINFORCE on CartPole
- Full PPO on LunarLander
- SAC on MuJoCo HalfCheetah
