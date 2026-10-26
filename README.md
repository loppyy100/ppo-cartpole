# PPO on CartPole

A PPO (Proximal Policy Optimization) agent learns to balance a pole on a cart, using
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and
[Gymnasium](https://gymnasium.farama.org/) `CartPole-v1`. Trained on CPU only.

![trained agent](ppo%20project/trained_agent.gif)

## Result

| | Mean reward (max 500) |
|---|---|
| Random policy | 9 – 32 |
| PPO, 200k steps | **485.4** over 20 deterministic evaluation episodes |

CartPole-v1 counts as solved at an average of 475.

![learning curve](ppo%20project/learning_curve.png)

## What `ppo project/ppo.py` does

1. Builds the environment (wrapped in `Monitor` + `DummyVecEnv`)
2. Trains PPO (`MlpPolicy`, lr 3e-4, n_steps 1024, batch 256, gamma 0.99, seed 42)
3. Records every episode's reward with a custom callback and plots the learning curve
4. Evaluates the trained policy on 20 fresh episodes (deterministic actions)
5. Renders one episode as `trained_agent.gif` and saves the model to `ppo_cartpole.zip`

```bash
pip install stable-baselines3 gymnasium matplotlib pillow
cd "ppo project"
python ppo.py                          # 200k steps
RL_DEMO_TIMESTEPS=50000 python ppo.py  # quicker run
```

## `self learning/`

My step-by-step lessons, typed by hand.

**CartPole**
- `rl.py`: look at the observation/action spaces and play with a random policy (scores 9–32)
- `brain.py`: train PPO for 50k steps (mean episode reward went 23 → 393) and save it
- `watch.py`: load the saved model and watch it play

**Ant (MuJoCo `Ant-v5`)**: 105 observations, 8 continuous motor actions
- `ant.py`: inspect the spaces and watch a random policy flail (and flip over)
- `ant_brain.py`: train PPO (`MlpPolicy`) for 1M steps, saved as `my_ant_brain.zip`
- `ant_watch.py`: load the brain and watch it run

| Ant-v5, PPO | Mean episode reward | Mean episode length (max 1000) |
|---|---|---|
| 300k steps | 127 | 304 |
| 1M steps (~10 min on CPU) | **865** | 541 |

The Ant's reward mostly pays for forward speed, so the trained agent learned to sprint.

## Next

A real quadruped: the Unitree Go1 model (MuJoCo Menagerie) learning to walk, working toward a robot dog climbing stairs.
