import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import einops as ei

from typing import Optional, Tuple, NamedTuple
from flax.training.train_state import TrainState
from jaxproxqp.jaxproxqp import JaxProxQP

from gcbfplus.utils.typing import Action, Params, PRNGKey, Array, State
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import merge01, jax_vmap, mask2index, tree_merge
from gcbfplus.trainer.data import Rollout
from gcbfplus.trainer.buffer import MaskedReplayBuffer
from gcbfplus.trainer.utils import compute_norm_and_clip, jax2np, tree_copy, empty_grad_tx
from gcbfplus.env.base import MultiAgentEnv
from gcbfplus.algo.module.cbf import CBF
from gcbfplus.algo.module.policy import DeterministicPolicy
from .gcbf import GCBF


class Batch(NamedTuple):
    graph: GraphsTuple
    safe_mask: Array
    unsafe_mask: Array
    u_qp: Action


class GCBFPlus(GCBF):

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
            horizon: int = 32,
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
        self.horizon = horizon

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
        cbf_optim = optax.adamw(learning_rate=lr_cbf, weight_decay=1e-3)
        self.cbf_optim = optax.apply_if_finite(cbf_optim, 1_000_000)
        self.cbf_train_state = TrainState.create(
            apply_fn=self.cbf.get_cbf,
            params=cbf_params,
            tx=self.cbf_optim
        )
        self.cbf_tgt = TrainState.create(apply_fn=self.cbf.get_cbf, params=tree_copy(cbf_params), tx=empty_grad_tx())

        # set up actor
        self.actor = DeterministicPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        actor_key, key = jr.split(key)
        actor_params = self.actor.net.init(actor_key, nominal_graph, self.n_agents)
        actor_optim = optax.adamw(learning_rate=lr_actor, weight_decay=1e-3)
        self.actor_optim = optax.apply_if_finite(actor_optim, 1_000_000)
        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.sample_action,
            params=actor_params,
            tx=self.actor_optim
        )

        # set up key
        self.key = key
        self.buffer = MaskedReplayBuffer(size=buffer_size)
        self.unsafe_buffer = MaskedReplayBuffer(size=buffer_size // 2)
        self.rng = np.random.default_rng(seed=seed + 1)

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
            'max_grad_norm': self.max_grad_norm,
            'horizon': self.horizon
        }

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, unsafe_mask: Array) -> jnp.ndarray:
        # safe if in the horizon, the agent is always safe
        def safe_rollout(single_rollout_mask: Array) -> Array:
            safe_rollout_mask = jnp.ones_like(single_rollout_mask)
            for i in range(single_rollout_mask.shape[0]):
                start = 0 if i < self.horizon else i - self.horizon
                safe_rollout_mask = safe_rollout_mask.at[start: i + 1].set(
                    ((1 - single_rollout_mask[i]) * safe_rollout_mask[start: i + 1]).astype(jnp.bool_))
                # initial state is always safe
                safe_rollout_mask = safe_rollout_mask.at[0].set(1)
            return safe_rollout_mask

        safe = jax_vmap(jax_vmap(safe_rollout, in_axes=1, out_axes=1))(unsafe_mask)
        return safe

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_train_state.params
        action = 2 * self.actor.get_action(params, graph) + self._env.u_ref(graph)
        return action

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        action, log_pi = self.actor_train_state.apply_fn(params, graph, key)
        return 2 * action + self._env.u_ref(graph), log_pi

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=1)
    def update_tgt(self, cbf_tgt: TrainState, cbf: TrainState, tau: float) -> TrainState:
        tgt_params = optax.incremental_update(cbf.params, cbf_tgt.params, tau)
        return cbf_tgt.replace(params=tgt_params)

    @ft.partial(jax.jit, static_argnums=(0,))
    def get_b_u_qp(self, b_graph: GraphsTuple, params) -> Action:
        b_u_qp, bT_relaxation = jax_vmap(ft.partial(self.get_qp_action, cbf_params=params))(b_graph)
        return b_u_qp

    def update_nets(self, rollout: Rollout, safe_mask, unsafe_mask):
        update_info = {}

        # Compute b_u_qp.
        n_chunks = 8
        batch_size = len(rollout.graph.states)
        chunk_size = batch_size // n_chunks

        # t0 = time.time()
        b_u_qp = []
        for ii in range(n_chunks):
            graph = jtu.tree_map(lambda x: x[ii * chunk_size: (ii + 1) * chunk_size], rollout.graph)
            b_u_qp.append(jax2np(self.get_b_u_qp(graph, self.cbf_tgt.params)))
        b_u_qp = tree_merge(b_u_qp)

        batch_orig = Batch(rollout.graph, safe_mask, unsafe_mask, b_u_qp)

        for i_epoch in range(self.inner_epoch):
            idx = self.rng.choice(rollout.length, size=rollout.length, replace=False)
            # (n_mb, mb_size)
            batch_idx = np.stack(np.array_split(idx, idx.shape[0] // self.batch_size), axis=0)
            batch = jtu.tree_map(lambda x: x[batch_idx], batch_orig)

            cbf_train_state, actor_train_state, update_info = self.update_inner(
                self.cbf_train_state, self.actor_train_state, batch
            )
            self.cbf_train_state = cbf_train_state
            self.actor_train_state = actor_train_state

        # Update target.
        self.cbf_tgt = self.update_tgt(self.cbf_tgt, self.cbf_train_state, 0.5)

        return update_info

    def sample_batch(self, rollout: Rollout, safe_mask, unsafe_mask):
        if self.buffer.length > self.batch_size:
            # sample from memory
            memory, safe_mask_memory, unsafe_mask_memory = self.buffer.sample(rollout.length)
            try:
                unsafe_memory, safe_mask_unsafe_memory, unsafe_mask_unsafe_memory = self.unsafe_buffer.sample(
                    rollout.length * rollout.time_horizon)
            except ValueError:
                unsafe_memory = jtu.tree_map(lambda x: merge01(x), memory)
                safe_mask_unsafe_memory = merge01(safe_mask_memory)
                unsafe_mask_unsafe_memory = merge01(unsafe_mask_memory)

            # append new data to memory
            self.buffer.append(rollout, safe_mask, unsafe_mask)
            unsafe_multi_mask = unsafe_mask.max(axis=-1)
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_multi_mask], rollout),
                safe_mask[unsafe_multi_mask],
                unsafe_mask[unsafe_multi_mask]
            )

            # get update data
            # (b, T)
            rollout = tree_merge([memory, rollout])
            safe_mask = tree_merge([safe_mask_memory, safe_mask])
            unsafe_mask = tree_merge([unsafe_mask_memory, unsafe_mask])

            # (b, T) -> (b * T, )
            rollout = jtu.tree_map(lambda x: merge01(x), rollout)
            safe_mask = merge01(safe_mask)
            unsafe_mask = merge01(unsafe_mask)
            rollout_batch = tree_merge([unsafe_memory, rollout])
            safe_mask_batch = tree_merge([safe_mask_unsafe_memory, safe_mask])
            unsafe_mask_batch = tree_merge([unsafe_mask_unsafe_memory, unsafe_mask])
        else:
            self.buffer.append(rollout, safe_mask, unsafe_mask)
            unsafe_multi_mask = unsafe_mask.max(axis=-1)
            self.unsafe_buffer.append(
                jtu.tree_map(lambda x: x[unsafe_multi_mask], rollout),
                safe_mask[unsafe_multi_mask],
                unsafe_mask[unsafe_multi_mask]
            )

            # (b, T) -> (b * T, )
            rollout_batch = jtu.tree_map(lambda x: merge01(x), rollout)
            safe_mask_batch = merge01(safe_mask)
            unsafe_mask_batch = merge01(unsafe_mask)

        return rollout_batch, safe_mask_batch, unsafe_mask_batch

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        # (n_collect, T)
        unsafe_mask = jax_vmap(jax_vmap(self._env.unsafe_mask))(rollout.graph)
        safe_mask = self.safe_mask(unsafe_mask)
        safe_mask, unsafe_mask = jax2np(safe_mask), jax2np(unsafe_mask)

        rollout_np = jax2np(rollout)
        del rollout
        rollout_batch, safe_mask_batch, unsafe_mask_batch = self.sample_batch(rollout_np, safe_mask, unsafe_mask)

        # inner loop
        update_info = self.update_nets(rollout_batch, safe_mask_batch, unsafe_mask_batch)

        return update_info

    def get_qp_action(
            self,
            graph: GraphsTuple,
            relax_penalty: float = 1e3,
            cbf_params=None,
            qp_settings: JaxProxQP.Settings = None,
    ) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> Array:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self._env.add_edge_feats(graph, new_state)
            return self.get_cbf(new_graph, params=cbf_params)

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        h = h_aug(agent_state).squeeze(-1)
        h_x = jax.jacobian(h_aug)(agent_state).squeeze(1)

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
        Lf_h = ei.einsum(h_x, dyn_f, "agent_i agent_j nx, agent_j nx -> agent_i")
        Lg_h = ei.einsum(h_x, dyn_g, "agent_i agent_j nx, agent_j nx nu -> agent_i agent_j nu")
        Lg_h = Lg_h.reshape((self.n_agents, -1))

        u_lb, u_ub = self._env.action_lim()
        u_lb = u_lb[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ub = u_ub[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ref = self._env.u_ref(graph).reshape(-1)

        # construct QP: min x^T H x + g^T x, s.t. Cx <= b
        H = jnp.eye(self._env.action_dim * self.n_agents + self.n_agents, dtype=jnp.float32)
        H = H.at[-self.n_agents:, -self.n_agents:].set(H[-self.n_agents:, -self.n_agents:] * 10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.n_agents)])
        C = -jnp.concatenate([Lg_h, jnp.eye(self.n_agents)], axis=1)
        b = Lf_h + self.alpha * 0.1 * h

        r_lb = jnp.array([0.] * self.n_agents, dtype=jnp.float32)
        r_ub = jnp.array([jnp.inf] * self.n_agents, dtype=jnp.float32)
        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)

        qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        if qp_settings is None:
            qp_settings = JaxProxQP.Settings.default()
        qp_settings.dua_gap_thresh_abs = None
        solver = JaxProxQP(qp, qp_settings)
        sol = solver.solve()

        assert sol.x.shape == (self.action_dim * self.n_agents + self.n_agents,)
        u_opt, r = sol.x[:self.action_dim * self.n_agents], sol.x[-self.n_agents:]
        u_opt = u_opt.reshape(self.n_agents, -1)

        return u_opt, r

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2))
    def update_inner(
            self, cbf_train_state: TrainState, actor_train_state: TrainState, batch: Batch
    ) -> tuple[TrainState, TrainState, dict]:
        def update_fn(carry, minibatch: Batch):
            cbf, actor = carry
            # (batch_size, n_agents) -> (minibatch_size * n_agents, )
            safe_mask_batch = merge01(minibatch.safe_mask)
            unsafe_mask_batch = merge01(minibatch.unsafe_mask)

            def get_loss(cbf_params: Params, actor_params: Params) -> Tuple[Array, dict]:
                # get CBF values
                cbf_fn = jax_vmap(ft.partial(self.cbf.get_cbf, cbf_params))
                cbf_fn_no_grad = jax_vmap(ft.partial(self.cbf.get_cbf, jax.lax.stop_gradient(cbf_params)))
                # (minibatch_size, n_agents)
                h = cbf_fn(minibatch.graph).squeeze(-1)
                # (minibatch_size * n_agents,)
                h = merge01(h)

                # unsafe region h(x) < 0
                unsafe_data_ratio = jnp.mean(unsafe_mask_batch)
                h_unsafe = jnp.where(unsafe_mask_batch, h, -jnp.ones_like(h) * self.eps * 2)
                max_val_unsafe = jax.nn.relu(h_unsafe + self.eps)
                loss_unsafe = jnp.sum(max_val_unsafe) / (jnp.count_nonzero(unsafe_mask_batch) + 1e-6)
                acc_unsafe_mask = jnp.where(unsafe_mask_batch, h, jnp.ones_like(h))
                acc_unsafe = (jnp.sum(jnp.less(acc_unsafe_mask, 0)) + 1e-6) / (jnp.count_nonzero(unsafe_mask_batch) + 1e-6)

                # safe region h(x) > 0
                h_safe = jnp.where(safe_mask_batch, h, jnp.ones_like(h) * self.eps * 2)
                max_val_safe = jax.nn.relu(-h_safe + self.eps)
                loss_safe = jnp.sum(max_val_safe) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)
                acc_safe_mask = jnp.where(safe_mask_batch, h, -jnp.ones_like(h))
                acc_safe = (jnp.sum(jnp.greater(acc_safe_mask, 0)) + 1e-6) / (jnp.count_nonzero(safe_mask_batch) + 1e-6)

                # get neural network actions
                action_fn = jax.vmap(ft.partial(self.act, params=actor_params))
                action = action_fn(minibatch.graph)

                # get next graph
                forward_fn = jax_vmap(self._env.forward_graph)
                next_graph = forward_fn(minibatch.graph, action)
                h_next = merge01(cbf_fn(next_graph).squeeze(-1))
                h_dot = (h_next - h) / self._env.dt

                # stop gradient and get next graph
                h_no_grad = jax.lax.stop_gradient(h)
                h_next_no_grad = merge01(cbf_fn_no_grad(next_graph).squeeze(-1))
                h_dot_no_grad = (h_next_no_grad - h_no_grad) / self._env.dt

                # h_dot + alpha * h > 0 (backpropagate to action, and backpropagate to h when labeled)
                labeled_mask = jnp.logical_or(unsafe_mask_batch, safe_mask_batch)
                max_val_h_dot = jax.nn.relu(-h_dot - self.alpha * h + self.eps)
                max_val_h_dot_no_grad = jax.nn.relu(-h_dot_no_grad - self.alpha * h + self.eps)
                max_val_h_dot = jnp.where(labeled_mask, max_val_h_dot, max_val_h_dot_no_grad)
                loss_h_dot = jnp.mean(max_val_h_dot)
                acc_h_dot = jnp.mean(jnp.greater(h_dot + self.alpha * h, 0))

                # action loss
                assert action.shape == minibatch.u_qp.shape
                loss_action = jnp.mean(jnp.square(action - minibatch.u_qp).sum(axis=-1))

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
            grad_cbf, grad_cbf_norm = compute_norm_and_clip(grad_cbf, self.max_grad_norm)
            grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
            cbf = cbf.apply_gradients(grads=grad_cbf)
            actor = actor.apply_gradients(grads=grad_actor)
            grad_info = {'grad_norm/cbf': grad_cbf_norm, 'grad_norm/actor': grad_actor_norm}
            return (cbf, actor), grad_info | loss_info

        train_state = (cbf_train_state, actor_train_state)
        (cbf_train_state, actor_train_state), info = lax.scan(update_fn, train_state, batch)

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info)
        return cbf_train_state, actor_train_state, info
