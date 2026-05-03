# Calculus & Optimization for RL

> Map section: [RL_LEARNING_MAP.md - Section 2.3](../RL_LEARNING_MAP.md#23-calculus--optimization)

## What This Module Covers

Calculus and optimization as the engine of learning in RL. Gradients drive policy improvement; optimization theory guarantees (or limits) convergence.

## Key Topics

- Partial derivatives and gradients
- Chain rule and backpropagation
- Jacobian and Hessian matrices
- Gradient descent and SGD variants (Adam, RMSprop)
- Convex vs non-convex optimization
- Lagrange multipliers and constrained optimization
- KKT conditions

## How It's Used in RL

- Policy gradient methods compute `nabla J(theta)` to optimize policies
- Backpropagation through neural networks uses the chain rule
- Adam optimizer is the default for most deep RL algorithms
- PPO uses constrained optimization (clipping) inspired by trust regions
- KL divergence penalty in RLHF is a Lagrangian constraint

## Suggested Resources

| Resource | Type | Difficulty |
|----------|------|-----------|
| 3Blue1Brown: Neural Networks | Video | Beginner |
| Khan Academy: Multivariable Calculus | Course | Beginner |
| Convex Optimization (Boyd & Vandenberghe) | Book/Course | Advanced |
| Numerical Optimization (Nocedal & Wright) | Book | Advanced |

## Suggested Exercises

- Implement gradient descent from scratch in NumPy
- Implement Adam optimizer and compare with vanilla SGD
- Derive the policy gradient theorem step by step
