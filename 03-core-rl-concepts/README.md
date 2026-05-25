# Core RL Concepts

> Map section: [RL_LEARNING_MAP.md - Section 3](../RL_LEARNING_MAP.md#3-core-rl-concepts)

## What This Module Covers

The fundamental building blocks of reinforcement learning: the agent-environment loop, states, actions, rewards, policies, value functions, Bellman equations, and the exploration-exploitation tradeoff.

## Key Topics

- The Agent-Environment Loop
- State, Action, Reward definitions
- Policy (deterministic and stochastic)
- Value Functions (V and Q)
- Advantage Function
- Bellman Equations (Expectation and Optimality)
- Exploration vs Exploitation strategies
- Model-Free vs Model-Based RL

## Core Equations

**Value Function:** `V^pi(s) = E_pi[G_t | S_t = s]`
**Q-Function:** `Q^pi(s,a) = E_pi[G_t | S_t = s, A_t = a]`
**Advantage:** `A^pi(s,a) = Q^pi(s,a) - V^pi(s)`

## Suggested Resources

- Sutton & Barto Ch.1-4 (Book, Beginner-Intermediate)
- David Silver's RL Course Lectures 1-3 (Video, Intermediate)
- HuggingFace Deep RL Course Unit 0-1 (Online, Beginner)
	- https://huggingface.co/learn/deep-rl-course/en/unit4/introduction

## Suggested Exercises

- Draw the agent-environment loop for 3 different real-world problems
- Implement a simple bandit environment and compare epsilon-greedy vs UCB
- Manually compute V and Q for a 2-state MDP
- Implement epsilon-greedy with decay schedule
