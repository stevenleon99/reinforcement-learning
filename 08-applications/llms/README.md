# RL for LLMs

> Map section: [RL_LEARNING_MAP.md - Section 9.3](../RL_LEARNING_MAP.md#93-rl-for-llm-alignment-rlhf-dpo-grpo)

## What This Module Covers

The hottest application of RL in 2023-2025: aligning large language models with human preferences using RLHF, DPO, and GRPO.

## Key Topics

- RLHF pipeline (pretrain -> SFT -> reward model -> PPO)
- DPO: direct preference optimization (no reward model)
- GRPO: memory-efficient PPO variant
- RLAIF: RL from AI feedback
- DeepSeek-R1: GRPO for reasoning

## Alignment Pipeline

1. Pre-train LM (next-token prediction)
2. Supervised Fine-Tuning (SFT)
3. Alignment via RLHF (PPO) or DPO/GRPO

## Suggested Resources

- HuggingFace TRL (the standard library)
- InstructGPT paper (Ouyang 2022)
- DPO paper (Rafailov 2023)
- DeepSeekMath paper (Shao 2024)
- HuggingFace RLHF blog

## Suggested Exercises

- Fine-tune Qwen-0.5B with SFT then DPO using TRL
- Compare DPO vs GRPO on same dataset
- Build a reward model for text preferences
