import jax.numpy as jnp
import jax
import einops as ei

from typing import Optional, Tuple
from jaxproxqp.jaxproxqp import JaxProxQP

from gcbfplus.utils.typing import Action, Params, PRNGKey, Array, State
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import mask2index
from gcbfplus.trainer.data import Rollout
from gcbfplus.env.base import MultiAgentEnv
from .utils import get_pwise_cbf_fn
from .base import MultiAgentController


class CentralizedCBF(MultiAgentController):

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
        super(CentralizedCBF, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        self.alpha = alpha
        self.k = 3
        self.cbf = get_pwise_cbf_fn(env, self.k)

    @property
    def config(self) -> dict:
        return {
            'alpha': self.alpha,
        }

    @property
    def actor_params(self) -> Params:
        raise NotImplementedError

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        raise NotImplementedError

    def get_cbf(self, graph: GraphsTuple) -> Array:
        return self.cbf(graph)[0]

    def update(self, rollout: Rollout, step: int) -> dict:
        raise NotImplementedError

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        return self.get_qp_action(graph)[0]

    def get_qp_action(self, graph: GraphsTuple, relax_penalty: float = 1e3) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> Array:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = graph._replace(edges=new_state[graph.receivers] - new_state[graph.senders], states=new_state)
            val = self.get_cbf(new_graph)
            assert val.shape == (self.n_agents, self.k)
            return val

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        h = h_aug(agent_state)  # (n_agents, k)
        h_x = jax.jacfwd(h_aug)(agent_state)  # (n_agents, k | n_agents, nx)
        h = h.reshape(-1)  # (n_agents * k,)

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
        Lf_h = ei.einsum(h_x, dyn_f, "agent_i k agent_j nx, agent_j nx -> agent_i k")
        Lg_h = ei.einsum(h_x, dyn_g, "agent_i k agent_j nx, agent_j nx nu -> agent_i k agent_j nu")
        Lf_h = Lf_h.reshape(-1)  # (n_agents * k,)
        Lg_h = Lg_h.reshape((self.n_agents * self.k, -1))  # (n_agents * k, n_agents * nu)

        u_lb, u_ub = self._env.action_lim()
        u_lb = u_lb[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ub = u_ub[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ref = self._env.u_ref(graph).reshape(-1)

        # construct QP
        H = jnp.eye(self._env.action_dim * self.n_agents + self.n_agents * self.k, dtype=jnp.float32)
        H = H.at[-self.n_agents * self.k:, -self.n_agents * self.k:].set(
            H[-self.n_agents * self.k:, -self.n_agents * self.k:] * 10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.n_agents * self.k)])
        C = -jnp.concatenate([Lg_h, jnp.eye(self.n_agents * self.k)], axis=1)
        b = Lf_h + self.alpha * h  # (n_agents * k,)

        r_lb = jnp.array([0.] * self.n_agents * self.k, dtype=jnp.float32)
        r_ub = jnp.array([jnp.inf] * self.n_agents * self.k, dtype=jnp.float32)

        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)

        qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        settings = JaxProxQP.Settings.default()
        settings.max_iter = 100
        settings.dua_gap_thresh_abs = None
        solver = JaxProxQP(qp, settings)
        sol = solver.solve()

        assert sol.x.shape == (self.action_dim * self.n_agents + self.n_agents * self.k,)
        u_opt, r = sol.x[:self.action_dim * self.n_agents], sol.x[-self.n_agents * self.k:]
        u_opt = u_opt.reshape(self.n_agents, -1)

        return u_opt, r

    def save(self, save_dir: str, step: int):
        raise NotImplementedError

    def load(self, load_dir: str, step: int):
        raise NotImplementedError
