Math expression:
![[Pasted image 20260516143954.png]]

**Meaning:**
The epsilon-greedy strategy is a policy that handles the exploration/exploitation trade-off.


**Application:**
- Strategy: At the beginning of the training, **the probability of doing exploration will be huge since ɛ is very high, so most of the time, we’ll explore.** But as the training goes on, and consequently our **Q-table gets better and better in its estimations, we progressively reduce the epsilon value** since we will need less and less exploration and more exploitation.