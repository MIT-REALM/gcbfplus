import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos3d, Reward, State
from ..utils.utils import merge01
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Sphere
from .plot import render_video
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class LinearDrone(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "drone_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.15, 0.3],
        "n_obs": 4
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None
    ):
        super(LinearDrone, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)

        self._A = np.zeros((self.state_dim, self.state_dim))
        self._A[0, 3] = 1.
        self._A[1, 4] = 1.
        self._A[2, 5] = 1.
        self._A[3, 3] = -1.1
        self._A[4, 4] = -1.1
        self._A[5, 5] = -6.
        A_discrete = scipy.linalg.expm(self._A * self._dt)

        self._B = np.zeros((self.state_dim, self.action_dim))
        self._B[3, 0] = 10.
        self._B[4, 1] = 10.
        self._B[5, 2] = 10.  # 6.0

        self._Q = np.diag([5e1, 5e1, 5e1, 1., 1., 1.])
        self._R = np.eye(self.action_dim)
        self._K = lqr(A_discrete, self._B, self._Q, self._R)
        self.create_obstacles = jax.vmap(Sphere.create)
        self.n_rays = 16  # consider top k rays

    @property
    def state_dim(self) -> int:
        return 6  # x, y, z, vx, vy, vz

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 6  # x_rel, y_rel, z_rel, vx_rel, vy_rel, vz_rel

    @property
    def action_dim(self) -> int:
        return 3  # ax, ay, az

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 3), minval=0, maxval=self.area_size)

        r_key, key = jr.split(key, 2)
        obs_radius = jr.uniform(r_key, (n_rng_obs,),
                                minval=self._params["obs_len_range"][0] / 2,
                                maxval=self._params["obs_len_range"][1] / 2)
        obstacles = self.create_obstacles(obs_pos, obs_radius)

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 3, obstacles, self.num_agents, 4 * self.params["drone_radius"], self.max_travel)

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, self.state_dim - 3))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, self.state_dim - 3))], axis=1)

        env_states = self.EnvState(states, goals, obstacles)

        return self.get_graph(env_states)

    def clip_action(self, action: Action) -> Action:
        lower_limit, upper_limit = self.action_lim()
        return jnp.clip(action, lower_limit, upper_limit)

    def agent_step_euler(self, agent_state: Array, action: Array) -> Array:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_state.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_state, action)
        n_state_agent_new = agent_state + x_dot * self.dt
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        return jnp.matmul(agent_states, self._A.T) + jnp.matmul(action, self._B.T)

    def step(
        self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        next_agent_states = self.agent_step_euler(agent_states, action)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goal_states, obstacles)

        info = {}
        if get_eval_info:
            # collision between agents and obstacles
            agent_pos = agent_states[:, :3]
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["drone_radius"])

        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states[:, :3]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["drone_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["drone_radius"])
        cost += collision.mean()

        return cost

    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=3,
            n_agent=self.num_agents,
            n_rays=self.n_rays,
            r=self.params["drone_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: Pos3d) -> list[EdgeBlock]:
        n_hits = self.num_agents * self.n_rays

        # agent - agent connection
        agent_pos = state.agent[:, :3]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :3].set(agent_goal_feats[:, :3] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self.n_rays, (i + 1) * self.n_rays)
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :3]
            lidar_feats = state.agent[i, :] - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self.n_rays))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.matmul(state, self._A.T)
        g = self._B
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_feats = state[graph.receivers] - state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :3].set(edge_feats[:, :3] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self.n_rays * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(LinearDrone.GOAL)
        node_type = node_type.at[-n_hits:].set(LinearDrone.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self.params['n_rays'],
                sense_range=self._params["comm_radius"],
                max_returns=self.n_rays,
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :3]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros_like(lidar_data)], axis=-1)
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        low_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -0.5, -0.5, -0.5])
        up_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf, 0.5, 0.5, 0.5])
        return low_lim, up_lim

    def action_lim(self) -> Tuple[Action, Action]:
        low_lim = jnp.array([-1., -1., -1.])
        up_lim = jnp.array([1., 1., 1.])
        return low_lim, up_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        error = goal - agent
        error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]
        agent_vel = graph.type_states(type_idx=0, n_type=self.num_agents)[:, 3:]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["drone_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["drone_radius"] * 2.5)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"] * 1.5)

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["drone_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["drone_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["drone_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]
        goal_pos = graph.env_states.goal[:, :3]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["drone_radius"] * 2
        return reach
