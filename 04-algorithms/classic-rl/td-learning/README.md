# Temporal Difference (TD) Learning

> Map section: [RL_LEARNING_MAP.md - Section 4.1](../RL_LEARNING_MAP.md#temporal-difference-td-learning)

## What This Module Covers

TD methods combine MC sampling with DP bootstrapping -- the most practical family of tabular RL algorithms.

## Key Topics

- TD(0) for prediction
- SARSA (on-policy TD control)
- Q-Learning (off-policy TD control)
- Expected SARSA
- TD(lambda) and eligibility traces

## Key Equations

```
# TD(0) Update
V(s) <- V(s) + alpha * [r + gamma * V(s') - V(s)]

# SARSA Update
Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]

# Q-Learning Update
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

## Suggested Exercises

- Implement TD(0) on FrozenLake
- Implement SARSA vs Q-Learning on Cliff Walking
- Compare on-policy (SARSA) vs off-policy (Q-Learning) behavior
- Implement Q-Learning on discretized CartPole
