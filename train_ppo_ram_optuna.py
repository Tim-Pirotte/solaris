from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from functools import partial
import gymnasium as gym
import torch.nn as nn
import ale_py
import optuna
import torch

N_TRIALS = 50
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 6
N_TIMESTEPS = 500_000
N_EVAL_EPISODES = 5
N_ENVS = 8
TIMEOUT = 28_800
ENV_ID = 'ALE/Solaris-v5'

class TimePenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=-0.01):
        super().__init__(env)
        self.penalty = penalty

    def reward(self, reward):
        return reward + self.penalty

def linear_schedule(initial_value: float):
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    
    return func

class TrialEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes,
        eval_freq,
        deterministic: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )

        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True

                return False

        return continue_training
    
def objective(trial: optuna.Trial) -> float:
    n_steps = 2 ** trial.suggest_int('n_steps', 6, 8)
    batch_size = n_steps * N_ENVS
    n_epochs = trial.suggest_int('n_epochs', 2, 6)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 3e-5, 3e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    activation_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU}[trial.suggest_categorical('activation_fn', ['tanh', 'relu'])]
    
    time_penalty = trial.suggest_float('time_penalty', 0.0001, 1, log=True)

    wrapper_with_penalty = partial(TimePenaltyWrapper, penalty=-time_penalty)

    train_env = make_vec_env(
        ENV_ID, 
        wrapper_class=wrapper_with_penalty,
        env_kwargs={'obs_type': 'ram'}, 
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
    )

    train_env = VecNormalize(train_env)

    model = PPO(
        'MlpPolicy', 
        env=train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs, 
        gamma=gamma, 
        learning_rate=learning_rate, 
        ent_coef=ent_coef,
        policy_kwargs={ 'activation_fn': activation_fn },
        device='cpu',
    )

    eval_env = make_vec_env(ENV_ID, n_envs=1, env_kwargs={'obs_type': 'ram'})
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=max(N_TIMESTEPS // (N_EVALUATIONS * N_ENVS), 1),
    )

    nan_encountered = False

    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        print(e)
        nan_encountered = True
    finally:
        model.env.close()
        eval_env.close()

    if nan_encountered:
        return float('nan')

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(
        study_name='solaris',
        storage='sqlite:///runs/ppo_ram_optuna/optuna.db',
        load_if_exists=True, 
        sampler=sampler, 
        pruner=pruner, 
        direction='maximize',
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)
    
    print('Best trial:')
    print(study.best_trial)
    print('Best value:')
    print(study.best_trial.value)
    print('Parameters:')

    for key, value in study.best_trial.params.items():
        print(f'    {key}: {value}')

    print('User attributes:')

    for key, value in study.best_trial.user_attrs.items():
        print(f'    {key}: {value}')
