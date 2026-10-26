import gymnasium as gym


env = gym.make("CartPole-v1")
print(env.observation_space)
print(env.action_space)

obs, info = env.reset()
print(obs)
total = 0
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    done = terminated or truncated
print("Score:", total)
print(obs, reward)