import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle

from typing import Optional, Tuple
from flax.training.train_state import TrainState

from gcbfplus.utils.typing import Action, Params, PRNGKey, Array
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import merge01, jax_vmap, tree_merge
from gcbfplus.trainer.data import Rollout
from gcbfplus.trainer.buffer import ReplayBuffer
from gcbfplus.trainer.utils import has_any_nan, compute_norm_and_clip
from gcbfplus.env.base import MultiAgentEnv
from gcbfplus.algo.module.cbf import CBF
from gcbfplus.algo.module.policy import DeterministicPolicy
from .base import MultiAgentController


class GCBF(MultiAgentController):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            gnn_layers: int,
            batch_size: int,
            buffer_size: int,
            lr_actor: float = 3e-5,
            lr_cbf: float = 3e-5,
            alpha: float = 1.0,
            eps: float = 0.02,
            inner_epoch: int = 8,
            loss_action_coef: float = 0.001,
            loss_unsafe_coef: float = 1.,
            loss_safe_coef: float = 1.,
            loss_h_dot_coef: float = 0.2,
            max_grad_norm: float = 2.,
            seed: int = 0,
            online_pol_refine: bool = False,
            **kwargs
    ):
        super(GCBF, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # set hyperparameters
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_cbf = lr_cbf
        self.alpha = alpha
        self.eps = eps
        self.inner_epoch = inner_epoch
        self.loss_action_coef = loss_action_coef
        self.loss_unsafe_coef = loss_unsafe_coef
        self.loss_safe_coef = loss_safe_coef
        self.loss_h_dot_coef = loss_h_dot_coef
        self.gnn_layers = gnn_layers
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.online_pol_refine = online_pol_refine

        # set nominal graph for initialization of the neural networks
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph

        # set up CBF
        self.cbf = CBF(
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_agents=n_agents,
            gnn_layers=gnn_layers
        )
        key = jr.PRNGKey(seed)
        cbf_key, key = jr.split(key)
        cbf_params = self.cbf.net.init(cbf_key, nominal_graph, self.n_agents)
        cbf_optim = optax.adam(learning_rate=lr_cbf)
        self.cbf_optim = optax.apply_if_finite(cbf_optim, 1_000_000)
        self.cbf_train_state = TrainState.create(
            apply_fn=self.cbf.get_cbf,
            params=cbf_params,
            tx=self.cbf_optim
        )

        # set up actor
        self.actor = DeterministicPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        actor_key, key = jr.split(key)
        actor_params = self.actor.net.init(actor_key, nominal_graph, self.n_agents)
        actor_optim = optax.adam(learning_rate=lr_actor)
        self.actor_optim = optax.apply_if_finite(actor_optim, 1_000_000)
        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.sample_action,
            params=actor_params,
            tx=self.actor_optim
        )

        # set up key
        self.key = key
        self.buffer = ReplayBuffer(size=buffer_size)
        self.unsafe_buffer = ReplayBuffer(size=buffer_size // 2)

    @property
    def config(self) -> dict:
        return {
            'batch_size': self.batch_size,
            'lr_actor': self.lr_actor,
            'lr_cbf': self.lr_cbf,
            'alpha': self.alpha,
            'eps': self.eps,
            'inner_epoch': self.inner_epoch,
            'loss_action_coef': self.loss_action_coef,
            'loss_unsafe_coef': self.loss_unsafe_coef,
            'loss_safe_coef': self.loss_safe_coef,
            'loss_h_dot_coef': self.loss_h_dot_coef,
            'gnn_layers': self.gnn_layers,
            'seed': self.seed,
            'max_grad_norm': self.max_grad_norm
        }

    @property
    def actor_params(self) -> Params:
        return self.actor_train_state.params

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if self.online_pol_refine:
            return self.online_policy_refinement(graph, params)
        if params is None:
            params = self.actor_params
        nn_action = 2 * self.actor.get_action(params, graph) + self._env.u_ref(graph)
        return nn_action

    def online_policy_refinement(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_params
        h = self.get_cbf(graph)

        # try u_ref first
        u_ref = self._env.u_ref(graph)
        next_graph_u_ref = self._env.forward_graph(graph, u_ref)
        h_next_u_ref = self.get_cbf(next_graph_u_ref)
        h_dot_u_ref = (h_next_u_ref - h) / self._env.dt
        max_val_h_dot_u_ref = jax.nn.relu(-h_dot_u_ref - self.alpha * h)
        nn_action = 2 * self.actor.get_action(params, graph) + u_ref
        nn_action = jnp.where(max_val_h_dot_u_ref > 0, nn_action, u_ref)

        max_iter = 30
        lr = 0.1

        def do_refinement(inp):
            i_iter, action, prev_h_dot_val = inp

            def h_dot_cond_val(a: Action):
                next_graph = self._env.forward_graph(graph, a)
                h_next = self.get_cbf(next_graph)
                h_dot = (h_next - h) / self._env.dt
                max_val_h_dot = jax.nn.relu(-h_dot - self.alpha * h).mean()
                return max_val_h_dot

            h_dot_val, grad = jax.value_and_grad(h_dot_cond_val)(action)
            action = action - lr * grad
            i_iter += 1
            return i_iter, action, h_dot_val

        def continue_refinement(inp):
            i_iter, action, h_dot_val = inp
            return (h_dot_val > 0) & (i_iter < max_iter)

        _, nn_action, _ = jax.lax.while_loop(
            continue_refinement, do_refinement, init_val=(0, nn_action, 1.0)
        )

        return nn_action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.actor_train_state.apply_fn(params, graph, key)
        return 2 * action + self._env.u_ref(graph), log_pi

    def get_cbf(self, graph: GraphsTuple, params: Optional[Params] = None) -> Array:
        if params is None:
            params = self.cbf_train_state.params
        return self.cbf.get_cbf(params, graph)

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        if self.buffer.length > self.batch_size:
            # sample from memory and unsafe_memory
            memory = self.buffer.sample(rollout.length // 2)
            unsafe_memory = self.unsafe_buffer.sample(rollout.length * rollout.time_horizon)

            # append new data to memory and unsafe_memory
            unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph).max(axis=-1)
            self.unsafe_buffer.append(jtu.tree_map(lambda x: x[unsafe_mask], rollout))
            self.buffer.append(rollout)

            # get update data
            rollout = tree_merge([memory, rollout])
            rollout = jtu.tree_map(lambda x: merge01(x), rollout)
            rollout = tree_merge([unsafe_memory, rollout])
        else:
            self.buffer.append(rollout)
            unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph).max(axis=-1)
            self.unsafe_buffer.append(jtu.tree_map(lambda x: x[unsafe_mask], rollout))
            rollout = jtu.tree_map(lambda x: merge01(x), rollout)

        # inner loop
        update_info = {}
        for i_epoch in range(self.inner_epoch):
            idx = np.arange(rollout.length)
            np.random.shuffle(idx)
            batch_idx = jnp.array(jnp.array_split(idx, idx.shape[0] // self.batch_size))

            cbf_train_state, actor_train_state, update_info = self.update_inner(
                self.cbf_train_state, self.actor_train_state, rollout, batch_idx
            )
            self.cbf_train_state = cbf_train_state
            self.actor_train_state = actor_train_state

        return update_info

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def update_inner(
            self, cbf_train_state: TrainState, actor_train_state: TrainState, rollout: Rollout, batch_idx: Array
    ) -> Tuple[TrainState, TrainState, dict]:

        def update_fn(carry, idx):
            """Update the actor and the CBF network for a single batch given the batch index."""
            cbf, actor = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)

            def get_loss(cbf_params: Params, actor_params: Params) -> Tuple[Array, dict]:
                # get CBF values
                cbf_fn = jax_vmap(ft.partial(self.cbf.get_cbf, cbf_params))
                h = cbf_fn(rollout_batch.graph).squeeze()
                h = merge01(h)

                # unsafe region h(x) < 0
                unsafe_mask = merge01(jax_vmap(self._env.unsafe_mask)(rollout_batch.graph))
                unsafe_data_ratio = jnp.mean(unsafe_mask)
                h_unsafe = jnp.where(unsafe_mask, h, -jnp.ones_like(h) * self.eps * 2)
                max_val_unsafe = jax.nn.relu(h_unsafe + self.eps)
                loss_unsafe = jnp.sum(max_val_unsafe) / (jnp.count_nonzero(unsafe_mask) + 1e-6)
                acc_unsafe_mask = jnp.where(unsafe_mask, h, jnp.ones_like(h))
                acc_unsafe = (jnp.sum(jnp.less(acc_unsafe_mask, 0)) + 1e-6) / (jnp.count_nonzero(unsafe_mask) + 1e-6)

                # safe region h(x) > 0
                safe_mask = merge01(jax_vmap(self._env.safe_mask)(rollout_batch.graph))
                h_safe = jnp.where(safe_mask, h, jnp.ones_like(h) * self.eps * 2)
                max_val_safe = jax.nn.relu(-h_safe + self.eps)
                loss_safe = jnp.sum(max_val_safe) / (jnp.count_nonzero(safe_mask) + 1e-6)
                acc_safe_mask = jnp.where(safe_mask, h, -jnp.ones_like(h))
                acc_safe = (jnp.sum(jnp.greater(acc_safe_mask, 0)) + 1e-6) / (jnp.count_nonzero(safe_mask) + 1e-6)

                # get actions
                action_fn = jax.vmap(ft.partial(self.actor.get_action, actor_params))
                action = action_fn(rollout_batch.graph)

                # get next graph
                forward_fn = jax_vmap(self._env.forward_graph)
                next_graph = forward_fn(rollout_batch.graph, action)
                h_next = merge01(cbf_fn(next_graph))
                h_dot = (h_next - h) / self._env.dt

                # h_dot + alpha * h > 0
                max_val_h_dot = jax.nn.relu(-h_dot - self.alpha * h + self.eps)
                loss_h_dot = jnp.mean(max_val_h_dot)  # + jnp.max(max_val_h_dot)
                acc_h_dot = jnp.mean(jnp.greater(h_dot + self.alpha * h, 0))

                # action loss
                u_ref = jax_vmap(self._env.u_ref)(rollout_batch.graph)
                loss_action = jnp.mean(jnp.square(action - u_ref).sum(axis=-1))

                # total loss
                total_loss = (
                        self.loss_action_coef * loss_action
                        + self.loss_unsafe_coef * loss_unsafe
                        + self.loss_safe_coef * loss_safe
                        + self.loss_h_dot_coef * loss_h_dot
                )

                return total_loss, {'loss/action': loss_action,
                                    'loss/unsafe': loss_unsafe,
                                    'loss/safe': loss_safe,
                                    'loss/h_dot': loss_h_dot,
                                    'loss/total': total_loss,
                                    'acc/unsafe': acc_unsafe,
                                    'acc/safe': acc_safe,
                                    'acc/h_dot': acc_h_dot,
                                    'acc/unsafe_data_ratio': unsafe_data_ratio}

            (loss, loss_info), (grad_cbf, grad_actor) = jax.value_and_grad(
                get_loss, has_aux=True, argnums=(0, 1))(cbf.params, actor.params)
            grad_cbf_has_nan = has_any_nan(grad_cbf).astype(jnp.float32)
            grad_actor_has_nan = has_any_nan(grad_actor).astype(jnp.float32)
            grad_cbf, grad_cbf_norm = compute_norm_and_clip(grad_cbf, self.max_grad_norm)
            grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
            cbf = cbf.apply_gradients(grads=grad_cbf)
            actor = actor.apply_gradients(grads=grad_actor)
            return (cbf, actor), {'grad_norm/cbf': grad_cbf_norm,
                                  'grad_norm/actor': grad_actor_norm,
                                  'grad_has_nan/cbf': grad_cbf_has_nan,
                                  'grad_has_nan/actor': grad_actor_has_nan} | loss_info

        (cbf_train_state, actor_train_state), info = jax.lax.scan(
            update_fn, (cbf_train_state, actor_train_state), batch_idx
        )

        # get training info of the last epoch
        info = jtu.tree_map(lambda x: x[-1], info)

        return cbf_train_state, actor_train_state, info

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.actor_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.cbf_train_state.params, open(os.path.join(model_dir, 'cbf.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.actor_train_state = \
            self.actor_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.cbf_train_state = \
            self.cbf_train_state.replace(params=pickle.load(open(os.path.join(path, 'cbf.pkl'), 'rb')))
