import wandb
import os
import numpy as np
import jax
import jax.random as jr
import functools as ft

from time import time
from tqdm import tqdm

from .data import Rollout
from .utils import rollout
from ..env import MultiAgentEnv
from ..algo.base import MultiAgentController
from ..utils.utils import jax_vmap


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: MultiAgentController,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        wandb.login()
        wandb.init(name=params['run_name'], project='gcbf-jax', dir=self.log_dir)

        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        return True

    def train(self):
        # record start time
        start_time = time()

        # preprocess the rollout function
        def rollout_fn_single(params, key):
            return rollout(self.env, ft.partial(self.algo.step, params=params), key)

        def rollout_fn(params, keys):
            return jax.vmap(ft.partial(rollout_fn_single, params))(keys)

        rollout_fn = jax.jit(rollout_fn)

        # preprocess the test function
        def test_fn_single(params, key):
            return rollout(self.env_test, lambda graph, k: (self.algo.act(graph, params), None), key)

        def test_fn(params, keys):
            return jax.vmap(ft.partial(test_fn_single, params))(keys)

        test_fn = jax.jit(test_fn)

        # start training
        test_key = jr.PRNGKey(self.seed)
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        pbar = tqdm(total=self.steps, ncols=80)
        for step in range(0, self.steps + 1):
            # evaluate the algorithm
            if step % self.eval_interval == 0:
                test_rollouts: Rollout = test_fn(self.algo.actor_params, test_keys)
                total_reward = test_rollouts.rewards.sum(axis=-1)
                assert total_reward.shape == (self.n_env_test,)
                reward_min, reward_max = total_reward.min(), total_reward.max()
                reward_mean = np.mean(total_reward)
                reward_final = np.mean(test_rollouts.rewards[:, -1])
                finish_fun = jax_vmap(jax_vmap(self.env_test.finish_mask))
                finish = finish_fun(test_rollouts.graph).max(axis=1).mean()
                cost = test_rollouts.costs.sum(axis=-1).mean()
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1) >= 1e-6)
                eval_info = {
                    "eval/reward": reward_mean,
                    "eval/reward_final": reward_final,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                    "eval/finish": finish,
                    "step": step,
                }
                wandb.log(eval_info, step=self.update_steps)
                time_since_start = time() - start_time
                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, reward: {reward_mean:9.4f}, '
                                f'min/max reward: {reward_min:7.2f}/{reward_max:7.2f}, cost: {cost:8.4f}, '
                                f'unsafe_frac: {unsafe_frac:6.2f}, finish: {finish:6.2f}')
                tqdm.write(eval_verbose)
                if self.save_log and step % self.save_interval == 0:
                    self.algo.save(os.path.join(self.model_dir), step)

            # collect rollouts
            key_x0, self.key = jax.random.split(self.key)
            key_x0 = jax.random.split(key_x0, self.n_env_train)
            rollouts: Rollout = rollout_fn(self.algo.actor_params, key_x0)

            # update the algorithm
            update_info = self.algo.update(rollouts, step)
            wandb.log(update_info, step=self.update_steps)
            self.update_steps += 1

            pbar.update(1)
