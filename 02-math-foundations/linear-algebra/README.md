# Linear Algebra for RL

> Map section: [RL_LEARNING_MAP.md - Section 2.1](../../RL_LEARNING_MAP.md#21-linear-algebra)

## What This Module Covers

Linear algebra fundamentals as they apply to reinforcement learning: vectors for state/action representation, matrices for transition dynamics, eigenvalues for convergence analysis.

## Key Topics

- Vectors and vector spaces
- Matrix operations (multiplication, inverse, transpose)
- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- Inner products and norms
- Positive definite matrices

## How It's Used in RL

- State and action representations are vectors
- Transition dynamics use matrix operations
- Value function approximation relies on matrix algebra
- Policy parameters are weight matrices in neural networks
- Eigenvalues appear in convergence proofs and stability analysis

## Suggested Resources

| Resource | Type | Difficulty |
|----------|------|-----------|
| 3Blue1Brown: Essence of Linear Algebra | Video | Beginner |
| MIT 18.06 (Gilbert Strang) | Course | Intermediate |
| Mathematics for Machine Learning (Deisenroth et al.) | Book | Intermediate |
| Linear Algebra Done Right (Axler) | Book | Advanced |

## Suggested Exercises

- Implement vector/matrix operations in NumPy
- Compute eigenvalues of a transition matrix and verify convergence
- Apply SVD for dimensionality reduction on a small RL problem
