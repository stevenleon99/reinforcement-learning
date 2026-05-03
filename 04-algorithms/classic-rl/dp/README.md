# Dynamic Programming (DP)

> Map section: [RL_LEARNING_MAP.md - Section 4.1](../../RL_LEARNING_MAP.md#dynamic-programming-dp)

## What This Module Covers

DP methods solve known MDPs exactly. While rarely used directly in practice (model is usually unknown), they provide the theoretical foundation for all value-based methods.

## Key Topics

- Policy Evaluation (predicting value for a given policy)
- Policy Improvement (greedy improvement theorem)
- Policy Iteration (evaluate then improve, repeat)
- Value Iteration (directly compute optimal values)
- Asynchronous DP
- Convergence guarantees

## Key Equations

```
# Policy Evaluation (iterative)
V_{k+1}(s) = Sum_a pi(a|s) Sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]

# Value Iteration
V_{k+1}(s) = max_a Sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V_k(s')]
```

## Suggested Exercises

- Implement Policy Iteration on a 4x4 Gridworld
- Implement Value Iteration on FrozenLake
- Compare iteration counts: Policy Iteration vs Value Iteration
- Visualize value functions as they converge
