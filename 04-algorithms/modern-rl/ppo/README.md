# PPO (Proximal Policy Optimization)

> Map section: [RL_LEARNING_MAP.md - Section 4.3](../RL_LEARNING_MAP.md#ppo-proximal-policy-optimization)

## What This Module Covers

PPO is the most widely used RL algorithm in production. It powers RLHF for ChatGPT, Claude, and Gemini. The clipped objective prevents destructively large policy updates.

## Key Topics

- Clipped surrogate objective
- Trust-region motivation
- Advantage estimation (GAE - Generalized Advantage Estimation)
- Value function loss
- Entropy bonus for exploration
- PPO in RLHF pipeline

## Key Equation

```
L^CLIP(theta) = E[min(ratio(theta) * A, clip(ratio(theta), 1-eps, 1+eps) * A)]
where ratio(theta) = pi_theta(a|s) / pi_theta_old(a|s)
```

## Suggested Resources

- Schulman et al., 2017: "Proximal Policy Optimization Algorithms" (Paper)
- CleanRL PPO: ~300 lines single file (GitHub, Beginner)
- Stable Baselines3 PPO (Framework, Intermediate)
- Spinning Up PPO with mathematical exposition (Intermediate)

## Suggested Exercises

- Implement PPO from scratch on CartPole
- Implement PPO on LunarLander
- Compare PPO with and without GAE
- Visualize policy ratio clipping behavior
