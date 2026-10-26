

import os

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# How long to train. 200k steps takes a few minutes on CPU.
TOTAL_STEPS = int(os.environ.get("RL_DEMO_TIMESTEPS", "200000"))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------
# Callback: remember the score of every finished episode,
# so we can draw the learning curve afterwards.
# ------------------------------------------------------------
class RewardRecorder(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []  
        self.episode_steps = []   

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:  # a game just ended
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_steps.append(self.num_timesteps)
        return True


def make_env():
    return Monitor(gym.make("CartPole-v1"))


def main():
    print("=" * 55)
    print(" STEP 1: Creating the CartPole world...")
    print("=" * 55)
    env = DummyVecEnv([make_env])

    print(" STEP 2: Building a fresh PPO agent (it knows NOTHING yet)")
    model = PPO(
        "MlpPolicy",       
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,        
        verbose=0,
        seed=42,
    )

    print(f" STEP 3: Training for {TOTAL_STEPS:,} steps by trial and error...")
    print("         (+1 reward for every moment the pole stays up,")
    print("          game over when it falls -> it learns to balance)")
    recorder = RewardRecorder()
    model.learn(total_timesteps=TOTAL_STEPS, callback=recorder)
    print("         Training done!")

    model_path = os.path.join(OUT_DIR, "ppo_cartpole")
    model.save(model_path)
    print(f"         Trained brain saved to: {model_path}.zip")

    # --------------------------------------------------------
    # Learning curve: smooth the raw episode scores so the
    # improvement trend is easy to see.
    # --------------------------------------------------------
    print(" STEP 4: Drawing the learning curve...")
    rewards = np.array(recorder.episode_rewards)
    steps = np.array(recorder.episode_steps)
    window = 20
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        smooth_steps = steps[window - 1:]
    else:
        smooth, smooth_steps = rewards, steps

    plt.figure(figsize=(9, 5))
    plt.plot(steps, rewards, color="lightgray", linewidth=0.7, label="Each game")
    plt.plot(smooth_steps, smooth, color="tab:blue", linewidth=2,
             label="Average of last 20 games")
    plt.axhline(475, color="red", linestyle="--", linewidth=1,
                label="Solved (475+)")
    plt.xlabel("Training steps")
    plt.ylabel("Episode reward (max = 500)")
    plt.title("PPO learning to balance CartPole — reward over time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(OUT_DIR, "learning_curve.png")
    plt.savefig(curve_path, dpi=120)
    print(f"         Saved: {curve_path}")

    # --------------------------------------------------------
    # Final test: 20 fresh games, no more learning.
    # --------------------------------------------------------
    print(" STEP 5: Final test — 20 games with the trained agent...")
    test_env = gym.make("CartPole-v1")
    scores = []
    for _ in range(20):
        obs, _ = test_env.reset()
        done = False
        total = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            total += reward
            done = terminated or truncated
        scores.append(total)
    test_env.close()
    print(f"         Average score: {np.mean(scores):.1f} / 500")
    print(f"         Best game:     {max(scores):.0f} / 500")
    print("         (A random agent scores only ~20-30!)")

    # --------------------------------------------------------
    # Record one game of the trained agent as a GIF.
    # --------------------------------------------------------
    print(" STEP 6: Recording the trained agent...")
    rec_env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = rec_env.reset(seed=7)
    frames = []
    done = False
    while not done:
        frames.append(rec_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = rec_env.step(action)
        done = terminated or truncated
    rec_env.close()

    images = [Image.fromarray(f) for f in frames[::2]]  # every 2nd frame
    gif_path = os.path.join(OUT_DIR, "trained_agent.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=33, loop=0)
    print(f"         Saved: {gif_path}")
    print("=" * 55)
    print(" Done: saved learning_curve.png and trained_agent.gif")
    print("=" * 55)


if __name__ == "__main__":
    main()
