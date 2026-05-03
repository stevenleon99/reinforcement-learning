# Markov Decision Processes (MDPs)

> Map section: [RL_LEARNING_MAP.md - Section 2.4](../RL_LEARNING_MAP.md#24-markov-decision-processes-mdps)

## What This Module Covers

The formal mathematical framework underlying ALL of reinforcement learning. MDPs define the problem; everything else is solution methods.

## Key Topics

- MDP definition: tuple `(S, A, P, R, gamma)`
- Bellman Expectation Equation
- Bellman Optimality Equation
- Policy evaluation and policy improvement
- Value iteration and policy iteration
- Discount factor and its effect on optimal policies

## Key Equations

**Bellman Expectation:**
```
V^pi(s) = Sum_a pi(a|s) Sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V^pi(s')]
```

**Bellman Optimality:**
```
V*(s) = max_a Sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V*(s')]
```

## Suggested Resources

| Resource | Type | Difficulty |
|----------|------|-----------|
| Sutton & Barto Ch.3 | Book | Intermediate |
| David Silver's RL Course, Lecture 2 | Video | Intermediate |
| Puterman: Markov Decision Processes | Book | Advanced |

## Suggested Exercises

- Formulate FrozenLake as an MDP (define S, A, P, R, gamma)
- Implement value iteration on a small gridworld
- Implement policy iteration and compare with value iteration
- Verify that both converge to the same optimal policy
