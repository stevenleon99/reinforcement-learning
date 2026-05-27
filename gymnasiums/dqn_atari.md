## Reference
- [cleanrl/dqn_atari.py](../cleanrl/cleanrl/dqn_atari.py)

## Description
- Prepare the replayBuffer
  
```Markdown

During the first 80,000 timesteps:

1. The agent observes the current state.
   In Atari DQN, this state = 4 stacked frames.
   - the game world is selected by env_id: str = "Name and version of the game"

2. The agent chooses an action.
   But early on, the action is mostly random because epsilon is high.

3. The action is sent to the Gymnasium Atari environment.

4. The environment returns:
   next_obs, reward, termination/truncation info.

5. The transition is saved into the replay buffer:
   obs, action, reward, next_obs, done.
    - The buffer is a fixed-size circular queue. 
    ```
    self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
    ```

```

- Start training after 80,000 steps when the replay buffer is sufficiently populated.

```Markdown

6. Sample 32 old transitions from replay buffer

7. Compute TD target using target network

8. Compute current Q-value using q_network

9. Calculate loss

10. Backpropagate loss

11. Update q_network parameters

12. Occasionally update target_network

```

> **Target Network and Q Network**
> It is the same network predicts both the current Q-value and the target Q-value, 
> the target keeps moving while the model is learning, making training unstable.