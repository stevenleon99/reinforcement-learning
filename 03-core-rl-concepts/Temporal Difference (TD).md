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

```python
# Example: Q-learning update
data = rb.sample(args.batch_size)
with torch.no_grad():
    target_max, _ = target_network(data.next_observations).max(dim=1)
    # if done=0, the episode continues, so we add the discounted next state value
    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
old_val = q_network(data.observations).gather(1, data.actions).squeeze()
loss = F.mse_loss(td_target, old_val)
```
```text
A simple example:

At state t:
You are one step before winning.
Q_network says: Q(state_t, action) = 5

After taking the action:
You get reward = 1
Next state value from target network = 10
gamma = 0.9

TD target = 1 + 0.9 * 10 = 10
TD error = 10 - 5 = 5

So the network learns:

My current estimate 5 was too low.
It should move closer to 10.
```

