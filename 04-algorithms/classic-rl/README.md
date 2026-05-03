# Classic RL Algorithms

> Map section: [RL_LEARNING_MAP.md - Section 4.1](../../RL_LEARNING_MAP.md#section-4-classic-rl-algorithms)

## What This Module Covers

The foundational RL algorithm families that require no deep learning: Dynamic Programming, Monte Carlo methods, and Temporal Difference learning.

## Submodules

| Algorithm Family | Folder | Key Methods |
|-----------------|--------|-------------|
| Dynamic Programming | [./dp/](./dp/) | Policy Iteration, Value Iteration |
| Monte Carlo | [./monte-carlo/](./monte-carlo/) | First-Visit MC, Every-Visit MC, MC Control |
| Temporal Difference | [./td-learning/](./td-learning/) | TD(0), SARSA, Q-Learning |

## Suggested Resources

- Sutton & Barto Ch.4-7 (Book)
- David Silver Lectures 3-5 (Video)
- CleanRL (reference implementations)

## Suggested Exercises

- FrozenLake with Value Iteration
- Blackjack with Monte Carlo Control
- CartPole with Q-Learning (discretized)
- Compare convergence of DP vs MC vs TD
