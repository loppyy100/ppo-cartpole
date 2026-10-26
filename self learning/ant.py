import gymnasium as gym
from stable_baselines3 import PPO
env = gym.make("Ant-v5" , render_mode="human")
print(env.observation_space)
print(env.action_space)

obs, inf= env.reset()
for _ in range (1000):
    action = env.action_space.sample()
    obs, reward, truncated, terminated, info = env.step(action)
    if terminated or truncated:
        obs , inf =env.reset()
env.close()