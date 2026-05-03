# Policy Gradient Methods

> Map section: [RL_LEARNING_MAP.md - Section 4.2](../../RL_LEARNING_MAP.md#policy-gradient-methods)

## What This Module Covers

Policy gradient methods directly optimize the policy parameters by ascending the gradient of expected return. They handle continuous action spaces naturally.

## Key Topics

- REINFORCE algorithm
- Policy gradient theorem
- Baseline reduction (variance reduction)
- Compatible function approximation
- Natural policy gradient

## Key Equations

```
# REINFORCE
nabla J(theta) = E[Sum_t nabla log pi(a_t|s_t) * G_t]

# With baseline
nabla J(theta) = E[Sum_t nabla log pi(a_t|s_t) * (G_t - b(s_t))]
```

## Suggested Exercises

- Implement REINFORCE on CartPole
- Add a learned baseline (value function)
- Compare variance: with vs without baseline
- Apply to a continuous control task (Pendulum)
