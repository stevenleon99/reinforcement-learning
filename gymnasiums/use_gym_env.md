## How to Interact with Gymnasium Environments
---

> **Loop for interacting with gymnasium environments:**
>  1. Current observation: obs
>  2. Agent chooses action
>  3. Environment executes action
>  4. Environment returns next_obs and reward
>  5. Replay buffer stores:
>     obs, action, reward, next_obs, done
>  6. Later, training samples random batches from the buffer


> **How to interact with gymnasium environments**
> - Atari environment = actually runs the game
> - Replay buffer = records what happened
> - DQN model = learns from recorded experience