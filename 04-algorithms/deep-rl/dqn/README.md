# Deep Q-Networks (DQN)

> Map section: [RL_LEARNING_MAP.md - Section 4.2](../../RL_LEARNING_MAP.md#deep-q-networks-dqn)

## What This Module Covers

DQN launched the deep RL revolution (Mnih et al., 2015). It uses neural networks to approximate Q(s,a) for high-dimensional state spaces like Atari game pixels.

## Key Topics

- Neural network as Q-function approximator
- Experience replay buffer (break temporal correlations)
- Target network (stabilize training)
- Epsilon-greedy exploration
- Double DQN (reduce overestimation)
- Dueling DQN (separate state-value and advantage)
- Prioritized Experience Replay
- Distributional RL (C51, QR-DQN)
- Rainbow DQN (combining all improvements)

## Key Equation

```
L(theta) = E[(r + gamma * max_a' Q_target(s', a'; theta-) - Q(s, a; theta))^2]
```

## Suggested Resources

- Mnih et al., 2015: "Human-level control through deep RL" (Paper)
- CleanRL DQN implementation (GitHub, Beginner)
- SB3 DQN docs (Framework, Intermediate)

## Suggested Exercises

- Implement DQN on CartPole from scratch (PyTorch)
- Add experience replay buffer
- Add target network with soft/hard updates
- Extend to Double DQN
- Train on Atari (ALE/Breakout-v5)
