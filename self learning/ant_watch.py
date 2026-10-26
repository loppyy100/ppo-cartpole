import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Ant-v5", render_mode="human")
model = PPO.load("my_ant_brain")

obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()well it wa