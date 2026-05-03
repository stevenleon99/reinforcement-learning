# Modern RL & LLM Alignment

> Map section: [RL_LEARNING_MAP.md - Section 4.3](../RL_LEARNING_MAP.md#43-modern-rl-algorithm-comparison-table)

## What This Module Covers

The cutting-edge algorithms driving LLM alignment and modern RL research: PPO, DPO, GRPO, and the RLHF pipeline.

## Submodules

| Algorithm | Folder | Year | Key Idea |
|-----------|--------|------|----------|
| PPO | [./ppo/](./ppo/) | 2017 | Clipped trust-region policy updates |
| DPO | [./dpo/](./dpo/) | 2023 | Direct preference optimization (no reward model) |
| GRPO | [./grpo/](./grpo/) | 2024 | Group-relative rewards, no critic needed |

## Suggested Resources

- Schulman et al., 2017 (PPO paper)
- Rafailov et al., 2023 (DPO paper)
- Shao et al., 2024 / DeepSeekMath (GRPO paper)
- HuggingFace TRL library (implementation)

## Suggested Exercises

- Implement PPO from scratch on LunarLander
- Fine-tune a small LM with DPO using HuggingFace TRL
- Compare DPO vs GRPO on a preference dataset
