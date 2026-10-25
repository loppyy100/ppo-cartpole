import gymnasium as gym 
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")
model = PPO.load ("my_first_brain")

obs , info =env.reset()
total = 0 
done = False 
while not done :
     action , _ = model.predict(obs, deterministic=True)
     obs, reward, terminated, truncated, info = env.step(action)
     total += reward 
     done = terminated or truncated 
print("Score:", total)