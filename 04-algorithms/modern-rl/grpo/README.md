# GRPO (Group Relative Policy Optimization)

> Map section: [RL_LEARNING_MAP.md - Section 4.3](../RL_LEARNING_MAP.md#grpo-group-relative-policy-optimization)

## What This Module Covers

GRPO is a memory-efficient PPO variant that eliminates the critic network by normalizing rewards within groups of sampled responses. Used to train DeepSeekMath and DeepSeek-R1.

## Key Topics

- Group-based reward normalization
- Elimination of value/critic network
- Memory efficiency for large models
- Application to reasoning tasks (math, code)
- Comparison with PPO and DPO

## Key Equation

```
r_i_normalized = (r_i - mean(r_group)) / std(r_group)
```
Then apply PPO-style clipping on these normalized advantages.

**Key advantage:** No separate value/critic network -> saves ~50% GPU memory.

## Suggested Resources

- Shao et al., 2024: "DeepSeekMath" (Paper, introduces GRPO)
- DeepSeek-V2 Technical Report (GRPO at scale)
- HuggingFace TRL GRPOTrainer (Framework)

## Suggested Exercises

- Train a small LM with GRPO on math reasoning using TRL
- Compare GRPO vs DPO on same preference data
- Profile GPU memory: PPO vs GRPO
