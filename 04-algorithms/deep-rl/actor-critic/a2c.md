# Introduction to A2C in Reinforcement Learning

## 1. Math

A2C stands for **Advantage Actor-Critic**. It is a reinforcement learning algorithm that combines two main components:

- **Actor**: decides which action to take.
- **Critic**: evaluates how good the action is.

In reinforcement learning, an agent interacts with an environment. At each time step, the agent observes a state \(s_t\), takes an action \(a_t\), receives a reward \(r_t\), and moves to the next state \(s_{t+1}\).

The goal is to maximize the expected total discounted reward:

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\]

> So, the total reward gets maximized indirectly:
> 1. The actor tries an action.
> 2. The critic judges whether the outcome was better or worse than expected.
> 3. Good actions become more likely.
> 4. Bad actions become less likely.
> 5. Over many episodes, the policy improves.
> 6. A better policy produces higher total reward.

where \(\gamma\) is the discount factor.

In A2C, the critic estimates the value function:

\[
V(s_t)
\]

The value function estimates how good a state is. The advantage function is calculated as:

> Because the optimizer minimizes loss:
>   If advantage is positive, the loss encourages the model to increase the probability of that action.
>   If advantage is negative, the loss encourages the model to decrease the probability of that action.

\[
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
\]

In practice, A2C often estimates advantage using:

\[
A_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

The actor is updated using the policy gradient:

\[
\nabla_\theta J(\theta) = \nabla_\theta \log \pi_\theta(a_t|s_t) A_t
\]

The critic is updated by minimizing the value loss:

\[
L_v = (R_t - V(s_t))^2
\]
can also be written as:
\[
L_v = (r_t + \gamma V(s_{t+1}) - V(s_t))^2
\]
> is exactly the **one-step advantage estimate**, also called the **TD error**.

## 2. Meaning

The key idea of A2C is that the agent should not only learn which actions lead to high rewards, but also understand whether an action is better or worse than expected.

The **actor** learns the policy, which means it learns the probability of choosing each action in a given state. The **critic** learns to estimate the value of each state. The **advantage function** tells the actor whether the selected action was better than the critic expected.

Compared with basic policy gradient methods, A2C is more stable because the critic provides a baseline. This helps reduce the variance of learning and makes training more efficient.

## 3. Example with Simple Code

Below is a simple example showing the basic structure of A2C using PyTorch. This is a simplified version to help understand the logic.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared(state)

        action_logits = self.actor(x)
        state_value = self.critic(x)

        return action_logits, state_value


env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

gamma = 0.99

for episode in range(500):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        action_logits, value = model(state_tensor)
        action_probs = torch.softmax(action_logits, dim=-1)

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        with torch.no_grad():
            _, next_value = model(next_state_tensor)

        target = reward + gamma * next_value * (1 - int(done))
        advantage = target - value

        actor_loss = -action_dist.log_prob(action) * advantage.detach()
        critic_loss = advantage.pow(2)

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    print(f"Episode {episode}, Total Reward: {total_reward}")