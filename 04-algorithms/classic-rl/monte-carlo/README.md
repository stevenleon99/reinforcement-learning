# Monte Carlo Methods

> Map section: [RL_LEARNING_MAP.md - Section 4.1](../RL_LEARNING_MAP.md#monte-carlo-mc-methods)

## What This Module Covers

MC methods learn from complete episodes by averaging sample returns. No bootstrapping, no model required -- just pure experience.

## Key Topics

- First-Visit MC Prediction
- Every-Visit MC Prediction
- MC Control with epsilon-greedy exploration
- Off-policy MC with importance sampling
- Episode-based learning (no bootstrapping)

## Key Equations

```
# MC Prediction (first-visit)
V(s) = average(G_t) for all visits to state s

# MC Control update
Q(s,a) = average(G_t) for all first visits to (s,a)
```

## Suggested Exercises

- Implement First-Visit MC Prediction on Blackjack
- Implement MC Control and learn an optimal Blackjack policy
- Compare First-Visit vs Every-Visit MC convergence
- Implement off-policy MC with weighted importance sampling
