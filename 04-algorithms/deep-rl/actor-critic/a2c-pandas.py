import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DDPG, HerReplayBuffer
import panda_gym

# Training environment
env = gym.make("PandaPush-v3")

model = DDPG(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class=HerReplayBuffer,
    verbose=1
)

model.learn(total_timesteps=100000)

# Save model
model.save("ddpg_panda_push")

# Close training env
env.close()


# -----------------------------
# Record video after training
# -----------------------------

video_env = gym.make("PandaPush-v3", render_mode="rgb_array")

video_env = RecordVideo(
    video_env,
    video_folder="./videos",
    name_prefix="ddpg_panda_push_final",
    episode_trigger=lambda episode_id: True
)

obs, info = video_env.reset()

terminated = False
truncated = False

while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = video_env.step(action)

video_env.close()