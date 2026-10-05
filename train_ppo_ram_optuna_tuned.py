from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
import ale_py

class TimePenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=-0.00016683469954212514):
        super().__init__(env)
        self.penalty = penalty

    def reward(self, reward):
        return reward + self.penalty

def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    
    return func

def train(run):
    train_env = make_vec_env(
        'ALE/Solaris-v5', 
        n_envs=8, 
        seed=42 + run, 
        vec_env_cls=SubprocVecEnv, 
        wrapper_class=TimePenaltyWrapper,
        env_kwargs={'obs_type': 'ram'},
    )

    train_env = VecNormalize(train_env)

    model = PPO(
        'MlpPolicy', 
        train_env, 
        seed=42 + run, 
        device='cpu', 
        tensorboard_log='./runs/logs/',
        n_steps=128,
        batch_size=1024,
        n_epochs=2,
        learning_rate=linear_schedule(3.287019011598836e-05),
        gamma=0.9089875492787637,
        clip_range=linear_schedule(0.1),
        ent_coef=0.007514905998041364,
    )

    model.learn(total_timesteps=10_000_000, progress_bar=True)

    model.save(f'runs/ppo_ram_optuna/{run}')
    train_env.save(f'runs/ppo_ram_optuna/vec_normalize_{run}.pkl')

if __name__ == '__main__':
    train(4)
    train(5)
