import gymnasium as gym
from stable_baselines3 import PPO
env = gym.make ("Ant-v5")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)
model.save('my_ant_brain')