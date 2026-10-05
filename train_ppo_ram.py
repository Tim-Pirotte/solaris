from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import ale_py

import subprocess

def train(run):
    train_env = make_vec_env('ALE/Solaris-v5', n_envs=8, seed=42 + run, vec_env_cls=SubprocVecEnv, env_kwargs={'obs_type': 'ram'})
    train_env = VecNormalize(train_env)

    model = PPO('MlpPolicy', train_env, seed=42 + run, device='cpu', tensorboard_log='./runs/logs/')
    model.learn(total_timesteps=10_000_000, progress_bar=True)

    model.save(f'runs/ppo_ram/{run}')
    train_env.save(f'runs/ppo_ram/vec_normalize_{run}.pkl')

if __name__ == '__main__':
    train(5)
    subprocess.run(["shutdown", "-s"])
