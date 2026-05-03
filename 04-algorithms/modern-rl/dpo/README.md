# DPO (Direct Preference Optimization)

> Map section: [RL_LEARNING_MAP.md - Section 4.3](../RL_LEARNING_MAP.md#dpo-direct-preference-optimization)

## What This Module Covers

DPO eliminates the reward model from RLHF by directly optimizing a classification loss on preference pairs. Simpler, more stable, and increasingly the default for LLM alignment.

## Key Topics

- RLHF pipeline and reward model limitations
- Closed-form optimal policy derivation
- DPO loss as a classification objective
- Beta parameter (controls deviation from reference)
- Comparison with PPO-based RLHF
- Extensions: IPO, KTO, ORPO

## Key Equation

```
L_DPO = -E[log sigma(beta * (log pi_theta(y_w|x) - log pi_ref(y_w|x))
                        - beta * (log pi_theta(y_l|x) - log pi_ref(y_l|x)))]
```
- y_w = preferred response, y_l = rejected response
- pi_ref = reference (SFT) model

## Suggested Resources

- Rafailov et al., 2023: "Direct Preference Optimization" (Paper)
- HuggingFace TRL DPOTrainer (Framework)
- Original DPO implementation: github.com/eric-mitchell/direct-preference-optimization

## Suggested Exercises

- Fine-tune Qwen-0.5B with DPO using TRL
- Compare DPO vs SFT-only on response quality
- Ablate beta parameter (0.1, 0.5, 1.0)
