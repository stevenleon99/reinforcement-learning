# Probability & Statistics for RL

> Map section: [RL_LEARNING_MAP.md - Section 2.2](../RL_LEARNING_MAP.md#22-probability--statistics)

## What This Module Covers

Probability theory and statistics as the language of uncertainty in RL. Every RL problem involves stochastic transitions, random rewards, and probabilistic policies.

## Key Topics

- Probability distributions (Bernoulli, Categorical, Gaussian, Beta, Dirichlet)
- Conditional probability and Bayes' theorem
- Expectation and variance
- Law of large numbers and central limit theorem
- Markov chains and stationary distributions
- Monte Carlo sampling
- Concentration inequalities (Hoeffding, Chebyshev)

## How It's Used in RL

- State transitions: `P(s'|s,a)` is stochastic
- Rewards are random variables
- Policies define probability distributions: `pi(a|s)`
- Value functions are expectations: `V(s) = E[G_t | S_t = s]`
- Policy gradients use expected gradients

## Suggested Resources

| Resource | Type | Difficulty |
|----------|------|-----------|
| STAT 110 (Harvard, Joe Blitzstein) | Course | Beginner-Intermediate |
| Pattern Recognition and ML (Bishop) Ch.1-2 | Book | Intermediate |
| Reinforcement Learning (Sutton & Barto) Ch.1-3 | Book | Intermediate |

## Suggested Exercises

- Simulate a Markov chain and compute its stationary distribution
- Estimate `E[X]` via Monte Carlo sampling; verify convergence with LLN
- Implement importance sampling for off-policy evaluation
