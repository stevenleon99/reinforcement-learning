# Markov Decision Processes (MDPs)

> Map section: [RL_LEARNING_MAP.md - Section 2.4](../../RL_LEARNING_MAP.md#24-markov-decision-processes-mdps)

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
*problem to solve: what is the best policy*
```
Vπ(s)=∑a​π(a∣s)∑s′​P(s′∣s,a)[R(s,a,s′)+γVπ(s′)]
```
- Vπ(s)V^{\pi}(s)Vπ(s): value of state sss under policy π\piπ
- π(a∣s)\pi(a|s)π(a∣s): probability of taking action aaa in state sss
- P(s′∣s,a)P(s'|s,a)P(s′∣s,a): probability of moving to next state s′s's′
- R(s,a,s′)R(s,a,s')R(s,a,s′): reward received
- γ\gammaγ: discount factor
- Vπ(s′)V^{\pi}(s')Vπ(s′): value of the next state


**Bellman Optimality:**
*problem to solve: what is the best action to take*
```
V∗(s)=maxa​∑s′​P(s′∣s,a)[R(s,a,s′)+γV∗(s′)]
```
- V∗(s)V^*(s)V∗(s): optimal value of state sss
- max⁡a\max_amaxa​: choose the best action
- The rest is the expected reward plus discounted future value
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
