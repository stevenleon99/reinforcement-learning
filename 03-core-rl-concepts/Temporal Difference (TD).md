**Math Expression:**

```
V(s) ← V(s) + α [r + γV(s') − V(s)]
```

s  = current state
r  = reward received
s' = next state
α  = learning rate
γ  = discount factor
V(s) = current estimate
V(s') = estimated value of next state

**Meaning:**
TD error = reward + discounted next-state value − current-state value
Update after every step using the next state's estimated value.


**Application in RL:**
SARSA  
Q-learning  
Expected SARSA

