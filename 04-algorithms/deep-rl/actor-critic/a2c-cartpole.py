# Example commands:
# python a2c-cartpole.py --record-initial-policy --record-steps 1000 --video-dir videos
# python a2c-cartpole.py --total-steps 100000 --record-initial-policy --record-when-stable --record-current-policy --record-reward-threshold 450 --stable-window 10 --record-steps 1000 --video-dir videos
# python a2c-cartpole.py --total-steps 100000 --center-reward-coef 0.2 --velocity-reward-coef 0.1 --angular-velocity-reward-coef 0.1 --record-initial-policy --record-when-stable --record-current-policy --record-reward-threshold 450 --stable-window 10 --record-steps 1000 --video-dir videos

import argparse
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image


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


def shaped_reward(original_reward, state, args):
    """
    Reward shaping for CartPole.

    CartPole state:
        state[0] = cart position
        state[1] = cart velocity
        state[2] = pole angle
        state[3] = pole angular velocity

    Reward shaping goals:
        1. Keep cart near the center.
        2. Keep cart velocity close to 0.
        3. Keep pole angular velocity close to 0.

    Important:
        This does not directly modify the neural network value function.
        It modifies the reward signal, so the critic learns a value function
        that prefers these states.
    """

    cart_position = state[0]
    cart_velocity = state[1]
    pole_angular_velocity = state[3]

    # CartPole x-position limit is approximately +/- 2.4
    max_position = 2.4

    # Practical scaling constants for velocity terms
    # These values clip the reward scores into [0, 1]
    cart_velocity_scale = 3.0
    pole_angular_velocity_scale = 3.5

    # Highest when cart is at center x = 0
    distance_from_center = abs(cart_position)
    center_score = 1.0 - min(distance_from_center / max_position, 1.0)

    # Highest when cart velocity is 0
    cart_velocity_score = 1.0 - min(
        abs(cart_velocity) / cart_velocity_scale,
        1.0
    )

    # Highest when pole angular velocity is 0
    pole_angular_velocity_score = 1.0 - min(
        abs(pole_angular_velocity) / pole_angular_velocity_scale,
        1.0
    )

    shaped = (
        original_reward
        + args.center_reward_coef * center_score
        + args.velocity_reward_coef * cart_velocity_score
        + args.angular_velocity_reward_coef * pole_angular_velocity_score
    )

    return shaped


def save_frame(frame, frame_dir, episode, step):
    episode_dir = frame_dir / f"episode_{episode:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    image_path = episode_dir / f"frame_{step:05d}.png"
    Image.fromarray(frame).save(image_path)


def make_train_env(args):
    if args.capture_video or args.save_frames:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
    else:
        env = gym.make("CartPole-v1")

    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_steps)

    if args.capture_video:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: episode_id % args.video_every == 0,
            name_prefix="a2c_train"
        )

    return env


def compute_returns(rewards, dones, next_value, gamma):
    returns = []
    R = next_value

    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0.0

        R = reward + gamma * R
        returns.insert(0, R)

    return returns


def record_policy(model, args, device, name_prefix):
    """
    Record one video using the current model policy.

    Uses greedy action selection:
        action = argmax(policy logits)
    """

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.record_steps)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: True,
        name_prefix=name_prefix
    )

    state, _ = env.reset()
    done = False
    total_reward = 0
    total_original_reward = 0
    step = 0

    model.eval()

    while not done and step < args.record_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_logits, _ = model(state_tensor)
            action = torch.argmax(action_logits, dim=-1)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        original_reward = reward

        if args.use_shaped_reward_in_recording:
            reward = shaped_reward(
                original_reward=reward,
                state=next_state,
                args=args
            )

        state = next_state
        total_reward += reward
        total_original_reward += original_reward
        step += 1

    env.close()

    print(
        f"Saved {name_prefix} video. "
        f"Original Reward: {total_original_reward}, "
        f"Reported Reward: {total_reward:.2f}, "
        f"Steps: {step}"
    )

    model.train()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_train_env(args)

    frame_dir = Path(args.frame_dir)
    if args.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Record initial untrained policy before training
    if args.record_initial_policy:
        record_policy(
            model=model,
            args=args,
            device=device,
            name_prefix="initial_policy"
        )

    state, _ = env.reset()

    episode = 0
    episode_reward = 0
    episode_original_reward = 0
    episode_step = 0
    total_steps = 0

    recent_rewards = []
    has_recorded_stable_policy = False

    while total_steps < args.total_steps:
        log_probs = []
        rewards = []
        dones = []
        values = []

        for _ in range(args.rollout_steps):
            if args.save_frames and episode % args.frame_every == 0:
                frame = env.render()
                save_frame(frame, frame_dir, episode, episode_step)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            action_logits, value = model(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)

            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            original_reward = reward

            reward = shaped_reward(
                original_reward=reward,
                state=next_state,
                args=args
            )

            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze())

            state = next_state

            episode_reward += reward
            episode_original_reward += original_reward
            episode_step += 1
            total_steps += 1

            if done:
                print(
                    f"Episode {episode}, "
                    f"Original Reward: {episode_original_reward}, "
                    f"Shaped Reward: {episode_reward:.2f}, "
                    f"Total Steps: {total_steps}"
                )

                # Use original reward for stability check
                recent_rewards.append(episode_original_reward)

                if len(recent_rewards) > args.stable_window:
                    recent_rewards.pop(0)

                if len(recent_rewards) == args.stable_window:
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                    print(f"Recent average original reward: {avg_reward:.2f}")

                    if (
                        args.record_when_stable
                        and not has_recorded_stable_policy
                        and avg_reward >= args.record_reward_threshold
                    ):
                        print(
                            f"Policy is stable. "
                            f"Average original reward {avg_reward:.2f} >= "
                            f"{args.record_reward_threshold}. "
                            f"Recording stable policy video..."
                        )

                        record_policy(
                            model=model,
                            args=args,
                            device=device,
                            name_prefix="stable_policy"
                        )

                        has_recorded_stable_policy = True

                state, _ = env.reset()
                episode += 1
                episode_reward = 0
                episode_original_reward = 0
                episode_step = 0

            if total_steps >= args.total_steps:
                break

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, next_value = model(state_tensor)
            next_value = next_value.item()

        returns = compute_returns(
            rewards=rewards,
            dones=dones,
            next_value=next_value,
            gamma=args.gamma
        )

        returns = torch.FloatTensor(returns).to(device)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + args.value_coef * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Record final trained policy after training
    if args.record_current_policy:
        record_policy(
            model=model,
            args=args,
            device=device,
            name_prefix="final_policy"
        )

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="A2C CartPole with center and velocity reward shaping"
    )

    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--rollout-steps", type=int, default=32)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)

    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument(
        "--center-reward-coef",
        type=float,
        default=0.2,
        help="Extra reward coefficient for staying close to the center"
    )

    parser.add_argument(
        "--velocity-reward-coef",
        type=float,
        default=0.1,
        help="Extra reward coefficient for keeping cart velocity close to 0"
    )

    parser.add_argument(
        "--angular-velocity-reward-coef",
        type=float,
        default=0.1,
        help="Extra reward coefficient for keeping pole angular velocity close to 0"
    )

    parser.add_argument(
        "--capture-video",
        action="store_true",
        help="Record videos during training"
    )

    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Folder where videos will be saved"
    )

    parser.add_argument(
        "--video-every",
        type=int,
        default=50,
        help="Record one training video every N episodes"
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save rendered frames as PNG images"
    )

    parser.add_argument(
        "--frame-dir",
        type=str,
        default="frames",
        help="Folder where frames will be saved"
    )

    parser.add_argument(
        "--frame-every",
        type=int,
        default=50,
        help="Save frames every N episodes"
    )

    parser.add_argument(
        "--record-initial-policy",
        action="store_true",
        help="Before training, record one video using the untrained initial policy"
    )

    parser.add_argument(
        "--record-current-policy",
        action="store_true",
        help="After training, record one video using the final trained policy"
    )

    parser.add_argument(
        "--record-steps",
        type=int,
        default=1000,
        help="Maximum steps for policy video recording"
    )

    parser.add_argument(
        "--record-when-stable",
        action="store_true",
        help="Record policy video once recent average reward reaches threshold"
    )

    parser.add_argument(
        "--record-reward-threshold",
        type=float,
        default=450,
        help="Original reward threshold for stable-policy recording"
    )

    parser.add_argument(
        "--stable-window",
        type=int,
        default=10,
        help="Number of recent episodes used to compute average reward"
    )

    parser.add_argument(
        "--use-shaped-reward-in-recording",
        action="store_true",
        help="Use shaped reward when reporting recorded video reward"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)