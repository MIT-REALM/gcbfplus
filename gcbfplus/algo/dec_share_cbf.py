import functools as ft
import einops as ei
import jax
import jax.numpy as jnp

from jaxproxqp.jaxproxqp import JaxProxQP
from typing import Optional, Tuple

from gcbfplus.env.base import MultiAgentEnv
from gcbfplus.trainer.data import Rollout
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.typing import Action, Array, Params, PRNGKey, State
from gcbfplus.utils.utils import mask2index, jax_vmap
from .base import MultiAgentController
from .utils import get_pwise_cbf_fn


class DecShareCBF(MultiAgentController):
    """Same as DecShareCBF, but takes the k closest agents into account."""

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            alpha: float = 1.0,
            **kwargs
    ):
        super().__init__(env=env, node_dim=node_dim, edge_dim=edge_dim, action_dim=action_dim, n_agents=n_agents)

        if hasattr(env, "enable_stop"):
            env.enable_stop = False

        self.cbf_alpha = alpha
        self.k = 3
        self.cbf = get_pwise_cbf_fn(env, self.k)

    @property
    def config(self) -> dict:
        return {
            "alpha": self.cbf_alpha,
        }

    @property
    def actor_params(self) -> Params:
        raise NotImplementedError

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        raise NotImplementedError

    def get_cbf(self, graph: GraphsTuple) -> tuple[Array, Array]:
        ak_h0, ak_isobs = self.cbf(graph)
        return ak_h0, ak_isobs

    def update(self, rollout: Rollout, step: int) -> dict:
        raise NotImplementedError

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        return self.get_qp_action(graph)[0]

    def get_qp_action(self, graph: GraphsTuple, relax_penalty: float = 1e3) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> tuple[Array, Array]:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = graph._replace(edges=new_state[graph.receivers] - new_state[graph.senders], states=new_state)
            ak_h_, ak_isobs_ = self.get_cbf(new_graph)
            assert ak_h_.shape == (self.n_agents, self.k)
            assert ak_isobs_.shape == (self.n_agents, self.k)
            return ak_h_, ak_isobs_

        def h(new_agent_state: State) -> Array:
            return h_aug(new_agent_state)[0]

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        # (n_agents, k)
        ak_h, ak_isobs = h_aug(agent_state)
        # (n_agents, k | n_agents, nx)
        ak_hx = jax.jacfwd(h)(agent_state)

        a_dyn_f, a_dyn_g = self._env.control_affine_dyn(agent_state)
        ak_Lf_h = ei.einsum(ak_hx, a_dyn_f, "agent_i k agent_j nx, agent_j nx -> agent_i k")
        aka_Lg_h: Array = ei.einsum(ak_hx, a_dyn_g, "agent_i k agent_j nx, agent_j nx nu -> agent_i k agent_j nu")

        def index_fn(idx: int):
            k_Lg_h = aka_Lg_h[idx, :, idx]
            assert k_Lg_h.shape == (self.k, self.action_dim)
            return k_Lg_h

        ak_Lg_h_self = jax_vmap(index_fn)(jnp.arange(self.n_agents))

        au_ref = self._env.u_ref(graph)
        assert au_ref.shape == (self.n_agents, self.action_dim)

        # (n_agents, ). 1 if agent-obs, 0.5 if agent-agent.
        ak_resp = jnp.where(ak_isobs, 1.0, 0.5)

        # construct QP
        au_opt, ar = jax_vmap(ft.partial(self._solve_qp_single, relax_penalty=relax_penalty))(
            ak_h, ak_Lf_h, ak_Lg_h_self, au_ref, ak_resp
        )
        return au_opt, ar

    def _solve_qp_single(self, k_h, k_Lf_h, k_Lg_h, u_ref, k_responsibility: float, relax_penalty: float = 1e3):
        n_qp_x = self._env.action_dim + self.k

        assert k_h.shape == (self.k,)
        assert k_Lf_h.shape == (self.k,)
        assert k_Lg_h.shape == (self.k, self._env.action_dim)

        u_lb, u_ub = self._env.action_lim()
        assert u_lb.shape == u_ub.shape == (self.action_dim,)

        ###########

        H = jnp.eye(n_qp_x, dtype=jnp.float32)
        H = H.at[-self.k :, -self.k :].set(10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.k)], axis=0)
        assert g.shape == (n_qp_x,)

        k_C = -jnp.concatenate([k_Lg_h, jnp.eye(self.k)], axis=1)
        assert k_C.shape == (self.k, n_qp_x)

        # Responsibility is one if this is agent-obs, half if this is agent-agent.
        k_b = k_responsibility * (k_Lf_h + self.cbf_alpha * k_h)
        assert k_b.shape == (self.k,)

        r_lb = jnp.full(self.k, 0.0, dtype=jnp.float32)
        r_ub = jnp.full(self.k, jnp.inf, dtype=jnp.float32)

        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)
        assert l_box.shape == u_box.shape == (n_qp_x,)

        qp = JaxProxQP.QPModel.create(H, g, k_C, k_b, l_box, u_box)
        settings = JaxProxQP.Settings.default()
        settings.max_iter = 100
        settings.dua_gap_thresh_abs = None
        solver = JaxProxQP(qp, settings)
        sol = solver.solve()

        assert sol.x.shape == (n_qp_x,)
        u_opt, r = sol.x[: self.action_dim], sol.x[-self.k :]

        return u_opt, r

    def save(self, save_dir: str, step: int):
        raise NotImplementedError

    def load(self, load_dir: str, step: int):
        raise NotImplementedError
