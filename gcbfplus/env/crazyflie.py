import functools as ft
import pathlib
import control as ct
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from typing import NamedTuple, Optional, Tuple

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos3d, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .linear_drone import LinearDrone
from .obstacle import Obstacle, Sphere
from .plot import render_video
from .utils import RK4_step, get_lidar, inside_obstacles, get_node_goal_rng


def get_rotmat(phi, theta, psi):
    c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
    c_th, s_th = jnp.cos(theta), jnp.sin(theta)
    c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
    R_W_cf = jnp.array(
        [
            [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
            [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
            [-s_th, c_th * s_phi, c_th * c_phi],
        ]
    )
    return R_W_cf


class CrazyFlie(MultiAgentEnv):
    """Pass it velocities, which are then used as targets in LQR."""

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
        "comm_radius": 1.0,
        "n_rays": 16,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 0,
        "m": 0.0299,
        "Ixx": 1.395 * 10 ** (-5),
        "Iyy": 1.395 * 10 ** (-5),
        "Izz": 2.173 * 10 ** (-5),
        "CT": 3.1582 * 10 ** (-10),
        "CD": 7.9379 * 10 ** (-12),
        "d": 0.03973,
    }

    # State indices
    X, Y, Z, PSI, THETA, PHI, U, V, W, R, Q, P = range(12)

    # Control indices
    F_1, F_2, F_3, F_4 = range(4)

    # LL State indices
    L_PHI, L_THETA, L_PSI, L_P, L_Q, L_R, L_VX, L_VY, L_VZ = range(9)
    L_U_VX, L_U_VY, L_U_VZ, L_U_R = range(4)

    def __init__(
        self,
        num_agents: int,
        area_size: float,
        max_step: int = 256,
        max_travel: float = None,
        dt: float = 0.03,
        params: dict = None,
    ):
        super().__init__(num_agents, area_size, max_step, max_travel, dt, params)

        self.create_obstacles = jax.vmap(Sphere.create)
        self.n_rays = 16
        self.n_rays = min(self.n_rays, self._params["n_rays"] ** 2 // 2 + 2)
        self.normalize_by_CT = True

        self.vel_targets_scale = np.array([2.0, 2.0, 0.5, 0.1])

        # Compute LL LQR gains. The linearization results in double-integrator like dynamics.
        self._K_ll = self._compute_K_ll()
        self._K_nom = self._compute_K_nom()

    @property
    def state_dim(self) -> int:
        return 12  # x, y, z, psi, theta, phi, u, v, w, r, q, p

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 12   # relative position, relative velocity, relative z direction, relative P Q R (world frame)

    @property
    def action_dim(self) -> int:
        # world frame velocities + yaw rate
        return 4

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

        next_agent_states = self.agent_step_rk4(agent_states, action)

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

    def edge_state(self, state: State) -> State:

        def edge_state_fn_(x: State) -> State:
            pos_W = x[:3]  # world frame position
            u = x[CrazyFlie.U]
            v = x[CrazyFlie.V]
            w = x[CrazyFlie.W]

            R_W_cf = self.get_rotation_mat(x)
            v_Wcf_cf = jnp.array([u, v, w])
            v_Wcf_W = R_W_cf @ v_Wcf_cf  # world frame velocity

            z_W = R_W_cf[:, 2]  # world frame z-axis

            r = x[CrazyFlie.R]
            q = x[CrazyFlie.Q]
            p = x[CrazyFlie.P]
            omega = jnp.array([p, q, r])
            omega_W = R_W_cf @ omega  # world frame angular velocity

            return jnp.concatenate([pos_W, v_Wcf_W, z_W, omega_W], axis=-1)

        return jax.vmap(edge_state_fn_)(state)

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_state = self.edge_state(state)
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :3].set(edge_feats[:, :3] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def edge_blocks(self, state: EnvState, lidar_data: Pos3d) -> list[EdgeBlock]:
        n_hits = self.num_agents * self.n_rays

        # agent - agent connection
        agent_pos = state.agent[:, :3]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_edge_state = self.edge_state(state.agent)
        state_diff = agent_edge_state[:, None, :] - agent_edge_state[None, :, :]
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        goal_edge_state = self.edge_state(state.goal)
        agent_goal_feats = agent_edge_state[:, None, :] - goal_edge_state[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :3] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :3].set(agent_goal_feats[:, :3] * coef)
        agent_goal_edges = EdgeBlock(agent_goal_feats, agent_goal_mask, id_agent, id_goal)

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        lidar_edge_state = self.edge_state(lidar_data)
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self.n_rays, (i + 1) * self.n_rays)
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :3]
            lidar_feats = agent_edge_state[i, :] - lidar_edge_state[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self.n_rays))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def _single_agent_f(self, x: Array):
        assert x.shape == (self.state_dim,)
        Ixx, Iyy, Izz = self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
        I = jnp.array([Ixx, Iyy, Izz])

        # roll, pitch, yaw
        phi, theta, psi = x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI]

        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        t_th = jnp.tan(theta)

        u, v, w = x[CrazyFlie.U], x[CrazyFlie.V], x[CrazyFlie.W]
        uvw = jnp.array([u, v, w])

        p, q, r = x[CrazyFlie.P], x[CrazyFlie.Q], x[CrazyFlie.R]
        pqr = jnp.array([p, q, r])

        R_W_cf = self.get_rotation_mat(x)
        v_Wcf_cf = jnp.array([u, v, w])
        v_Wcf_W = R_W_cf @ v_Wcf_cf
        assert v_Wcf_W.shape == (3,)

        # Euler angle dynamics.
        mat = jnp.array(
            [
                [0, s_phi / c_th, c_phi / c_th],
                [0, c_phi, -s_phi],
                [1, s_phi * t_th, c_phi * t_th],
            ]
        )
        deuler_ypr = mat @ pqr

        # Body frame linear acceleration.
        acc_cf_g = -R_W_cf[2, :] * 9.81
        acc_cf = -jnp.cross(pqr, uvw) + acc_cf_g

        # Body frame angular acceleration.
        pqr_dot = -jnp.cross(pqr, I * pqr) / I
        rpq_dot = pqr_dot[::-1]
        assert pqr_dot.shape == (3,)

        x_dot = jnp.concatenate([v_Wcf_W, deuler_ypr, acc_cf, rpq_dot], axis=0)
        assert x_dot.shape == (self.state_dim,)

        return x_dot

    def _single_agent_gu(self, x: Array, control: Array) -> Array:
        assert x.shape == (self.state_dim,)

        params = self._params
        m, Ixx, Iyy, Izz = params["m"], params["Ixx"], params["Iyy"], params["Izz"]
        CT, CD, d = params["CT"], params["CD"], params["d"]

        """
               3   ↑P  0
                  ╲ ╱   
               R ← ╳ 
                  ╱ ╲   
               2       1
        """

        if self.normalize_by_CT:
            CT, CD = 1, CD / CT

        # Try and avoid catastrophic cancellation by doing the sums early
        w_term = jnp.sum(control)
        p_term = jnp.sum(control * jnp.array([-1.0, -1.0, 1.0, 1.0]))
        q_term = jnp.sum(control * jnp.array([-1.0, 1.0, 1.0, -1.0]))
        r_term = jnp.sum(control * jnp.array([-1.0, 1.0, -1.0, 1.0]))

        w_dot = CT * w_term / m
        p_dot = CT * np.sqrt(2) * d * p_term / Ixx
        q_dot = CT * np.sqrt(2) * d * q_term / Ixx
        r_dot = CD * r_term / Izz

        gu = jnp.zeros(self.state_dim)
        gu = gu.at[CrazyFlie.W].set(w_dot)
        gu = gu.at[CrazyFlie.P].set(p_dot)
        gu = gu.at[CrazyFlie.Q].set(q_dot)
        gu = gu.at[CrazyFlie.R].set(r_dot)

        return gu

    def thrust_from_motor(self):
        """
        Return the matrix

               F1  F2  F3  F4
            W
            P
            Q
            R

        """
        params = self._params
        m, Ixx, Iyy, Izz = params["m"], params["Ixx"], params["Iyy"], params["Izz"]
        CT, CD, d = params["CT"], params["CD"], params["d"]
        if self.normalize_by_CT:
            CT, CD = 1, CD / CT
        dw_du = CT * np.full(4, 1 / m)
        dp_du = CT * np.sqrt(2) * d * jnp.array([-1.0, -1.0, 1.0, 1.0]) / Ixx
        dq_du = CT * np.sqrt(2) * d * jnp.array([-1.0, 1.0, 1.0, -1.0]) / Iyy
        dr_du = CD * np.array([-1.0, 1.0, -1.0, 1.0]) / Izz

        mat = np.stack([dw_du, dp_du, dq_du, dr_du], axis=0)
        return mat

    def _agent_xdot_single_agent(self, state: AgentState, control: Action) -> AgentState:
        f = self._single_agent_f(state)
        gu = self._single_agent_gu(state, control)
        assert f.shape == (self.state_dim,) and gu.shape == (self.state_dim,)
        assert control.shape == (self.action_dim,)
        xdot = f + gu
        assert xdot.shape == (self.state_dim,)
        return xdot

    def _xdot_ll(self, x: AgentState, u: Action) -> AgentState:
        assert x.shape == (9,)
        assert u.shape == (4,)
        m, Ixx, Iyy, Izz = self._params["m"], self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
        I = jnp.array([Ixx, Iyy, Izz])

        CT, CD, d = self._params["CT"], self._params["CD"], self._params["d"]
        if self.normalize_by_CT:
            CT, CD = 1, CD / CT

        phi, theta, psi = x[CrazyFlie.L_PHI], x[CrazyFlie.L_THETA], x[CrazyFlie.L_PSI]

        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th, s_th = jnp.cos(theta), jnp.sin(theta)
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        t_th = jnp.tan(theta)

        vx, vy, vz = x[CrazyFlie.L_VX], x[CrazyFlie.L_VY], x[CrazyFlie.L_VZ]
        v_Wcf_W = jnp.array([vx, vy, vz])

        p, q, r = x[CrazyFlie.L_P], x[CrazyFlie.L_Q], x[CrazyFlie.L_R]
        pqr = jnp.array([p, q, r])

        # Euler angle dynamics.
        mat = jnp.array(
            [
                [1, s_phi * t_th, c_phi * t_th],
                [0, c_phi, -s_phi],
                [0, s_phi / c_th, c_phi / c_th],
            ]
        )
        deuler_rpy = mat @ pqr

        R_W_cf = get_rotmat(x[CrazyFlie.L_PHI], x[CrazyFlie.L_THETA], x[CrazyFlie.L_PSI])

        # World frame linear acceleration.
        acc_W = jnp.array([0.0, 0.0, -9.81])

        # Body frame angular acceleration.
        pqr_dot = -jnp.cross(pqr, I * pqr) / I
        assert pqr_dot.shape == (3,)

        ############################################
        # Control part.
        dw_du = CT * jnp.full(4, 1 / m)
        dp_du = CT * np.sqrt(2) * d * jnp.array([-1.0, -1.0, 1.0, 1.0]) / Ixx
        dq_du = CT * np.sqrt(2) * d * jnp.array([-1.0, 1.0, 1.0, -1.0]) / Iyy
        dr_du = CD * jnp.array([-1.0, 1.0, -1.0, 1.0]) / Izz
        assert dw_du.shape == dp_du.shape == dq_du.shape == dr_du.shape == (self.action_dim,)

        wdot = dw_du @ u
        pdot = dp_du @ u
        qdot = dq_du @ u
        rdot = dr_du @ u

        pqr_dot_control = jnp.array([pdot, qdot, rdot])
        acc_cf_control = jnp.array([0.0, 0.0, wdot])
        acc_W_control = R_W_cf @ acc_cf_control

        ############################################
        x_dot = jnp.concatenate([deuler_rpy, pqr_dot + pqr_dot_control, acc_W + acc_W_control], axis=0)
        assert x_dot.shape == (9,)

        return x_dot

    def _compute_K_ll(self):
        def xdot(x, u):
            return self._xdot_ll(x, u + self.u_eq)

        x_zero = np.zeros(9)
        u_zero = np.zeros(4)

        # Sanity check: u_eq at equilibrium should result in equilibrium
        xdot_zero = np.array(xdot(x_zero, u_zero))
        np.testing.assert_allclose(xdot_zero, 0, atol=5e-5)

        A_ll, B_ll = jax.jacobian(xdot, argnums=(0, 1))(x_zero, u_zero)

        # Remove psi from A and B.
        A_ll = np.delete(np.delete(A_ll, CrazyFlie.L_PSI, axis=0), CrazyFlie.L_PSI, axis=1)
        assert A_ll.shape == (8, 8)

        B_ll = np.delete(B_ll, CrazyFlie.L_PSI, axis=0)
        assert B_ll.shape == (8, 4)

        #                [  ϕ,   θ,   p,   q,   r,  vx,  vy, vz ]
        Q = 1.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 20.0])
        Q = np.diag(Q)

        #                       [  w    p    q    r ]
        R_thrust = 0.01 * np.array([5.0, 1.0, 1.0, 1.0])
        T_fr_M = self.thrust_from_motor()
        R_motor = T_fr_M.T @ np.diag(R_thrust) @ T_fr_M

        K, S, E = ct.lqr(A_ll, B_ll, Q, R_motor)
        assert K.shape == (4, 8)

        # Add back psi to K.
        K = np.insert(K, CrazyFlie.L_PSI, 0, axis=1)
        assert K.shape == (4, 9)

        return K

    def _compute_K_nom(self):
        x_zero, u_zero = np.zeros(self.state_dim), np.zeros(self.action_dim)
        A_hl, B_hl = jax.jacobian(self._agent_xdot_single_agent_hl, argnums=(0, 1))(x_zero, u_zero)

        #                 [   x,   y,   z |  ψ,   θ,   ϕ |  u,   v,   w, | r,   q,   p ]
        Q = 2 * np.array([50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        #                [ vx,  vy,  vz,  r ]
        R = 4 * np.array([1.0, 1.0, 1.0, 1.0])

        K, S, E = ct.lqr(A_hl, B_hl, np.diag(Q), np.diag(R))
        return K

    def u_ref_inner_single(self, state: AgentState, goal: AgentState) -> Action:
        error = jnp.array(state - goal)
        # Clip the position errors.
        dist = jnp.linalg.norm(error[:3])
        dist_safe = jnp.maximum(dist, 1e-4)
        clip_coef = jnp.where(dist > self.comm_radius, self.comm_radius / dist_safe, 1.0)
        error = error.at[:3].multiply(clip_coef)
        u_hl = -self._K_nom @ error
        return self.clip_action(u_hl)

    @property
    def u_eq(self):
        """Return the equilibrium state.
        We want total thrust equals m*g.
        """
        u_eq = jnp.zeros((self.action_dim,))
        u_eq = u_eq.at[CrazyFlie.F_1].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_2].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_3].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_4].set(self._params["m"] * 9.81 / 4)

        if not self.normalize_by_CT:
            u_eq = u_eq / self._params["CT"]

        return u_eq

    def _get_ll_state(self, state: AgentState) -> AgentState:
        u, v, w = state[CrazyFlie.U], state[CrazyFlie.V], state[CrazyFlie.W]
        uvw = jnp.array([u, v, w])

        # 1: World frame velocities.
        phi, theta, psi = state[CrazyFlie.PHI], state[CrazyFlie.THETA], state[CrazyFlie.PSI]
        R_W_cf = self.get_rotation_mat(state)
        v_Wcf_cf = uvw
        v_Wcf_W = R_W_cf @ v_Wcf_cf
        assert v_Wcf_W.shape == (3,)
        vx, vy, vz = v_Wcf_W

        # 2: Yaw rate.
        p, q, r = state[CrazyFlie.P], state[CrazyFlie.Q], state[CrazyFlie.R]

        #                         WORLD FRAME
        # [ ϕ, θ, ψ, | p, q, r, | vx, vy, vz ]
        ll_state = jnp.array([phi, theta, psi, p, q, r, vx, vy, vz])
        assert ll_state.shape == (9,)
        return ll_state

    def _vel_targets_to_ll_state(self, vel_targets: Action):
        vx, vy, vz, r = vel_targets
        #                         WORLD FRAME
        # [ ϕ, θ, ψ, | p, q, r, | vx, vy, vz ]
        ll_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, r, vx, vy, vz])
        assert ll_state.shape == (9,)
        return ll_state

    def _get_ll_controls(self, state: AgentState, vel_targets: Action) -> Action:
        assert vel_targets.shape == (4,)
        ll_state = self._get_ll_state(state)
        ll_state_des = self._vel_targets_to_ll_state(vel_targets)

        control = -self._K_ll @ (ll_state - ll_state_des)
        control = self.u_eq + control
        return control

    def _agent_xdot_single_agent_hl(self, state: AgentState, vel_targets_scaled: Action) -> AgentState:
        assert state.shape == (self.state_dim,)
        assert vel_targets_scaled.shape == (self.action_dim,)

        # Clip the vel_targets_scaled to respect action limits.
        vel_targets_scaled = self.clip_action(vel_targets_scaled)
        vel_targets = vel_targets_scaled * self.vel_targets_scale

        # Get the control from the high level vel_targets.
        control = self._get_ll_controls(state, vel_targets)
        return self._agent_xdot_single_agent(state, control)

    def agent_xdot(self, agent_states: AgentState, vel_targets: Action) -> AgentState:
        assert vel_targets.ndim == agent_states.ndim
        if vel_targets.ndim == 1:
            return self._agent_xdot_single_agent_hl(agent_states, vel_targets)
        return jax_vmap(self._agent_xdot_single_agent_hl)(agent_states, vel_targets)

    def agent_step_rk4(self, agent_state: Array, vel_targets: Array) -> Array:
        assert agent_state.shape == (self.num_agents, self.state_dim)
        assert vel_targets.shape == (self.num_agents, self.action_dim)
        n_state_agent_new = RK4_step(self.agent_xdot, agent_state, vel_targets, self.dt)
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def rk4_single(self, agent_state: Array, vel_targets: Array):
        assert agent_state.shape == (self.state_dim,)
        assert vel_targets.shape == (self.action_dim,)
        state_agent_new = RK4_step(self._agent_xdot_single_agent_hl, agent_state, vel_targets, self.dt)
        assert state_agent_new.shape == (self.state_dim,)

        control = self._get_ll_controls(agent_state, vel_targets)
        return self.clip_state(state_agent_new), control

    def control_affine_dyn_single(self, state: AgentState) -> tuple[Array, Array]:
        u_zero = jnp.zeros(4)
        f = self.agent_xdot(state, u_zero)
        g = jax.jacobian(self.agent_xdot, argnums=1)(state, u_zero)

        return f, g

    def control_affine_dyn(self, state: State) -> tuple[Array, Array]:
        assert state.ndim == 2
        a_f, a_g = jax_vmap(self.control_affine_dyn_single)(state)
        return a_f, a_g

    @staticmethod
    def get_rotation_mat(x: State) -> State:
        return get_rotmat(x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI])

    def action_lim(self) -> Tuple[Action, Action]:
        low_lim = jnp.array([-1.0, -1.0, -1.0, -1.0])
        up_lim = jnp.array([1.0, 1.0, 1.0, 1.0])
        return low_lim, up_lim

    def render(self, graph: GraphsTuple) -> plt.Figure:
        pass

    def get_graph(self, state: EnvState) -> GraphsTuple:
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
        lidar_data = jnp.concatenate(
            [lidar_data, jnp.zeros((lidar_data.shape[0], self.state_dim - lidar_data.shape[1]))], axis=-1
        )
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
        # x, y, z, psi, theta, phi, u, v, w, r, q, p
        low_lim = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf,
                             -jnp.inf, -jnp.pi / 4, -jnp.pi / 4,
                             -0.3, -0.3, -0.3,
                             -10, -10, -10])
        up_lim = jnp.array([jnp.inf, jnp.inf, jnp.inf,
                            jnp.inf, jnp.pi / 4, jnp.pi / 4,
                            0.3, 0.3, 0.3,
                            10, 10, 10])
        return low_lim, up_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        return jax_vmap(self.u_ref_inner_single)(agent, goal)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_rk4(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :3]

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

    @property
    def comm_radius(self):
        return self._params["comm_radius"]

    @property
    def x_labels(self):
        x_labels = ["x", "y", "z", r"$\psi$", r"$\theta$", r"$\phi$", "u", "v", "w", "r", "q", "p"]
        assert len(x_labels) == self.state_dim
        return x_labels

    @property
    def uhl_labels(self):
        uhl_labels = [r"$vx_{tgt}$", r"$vy_{tgt}$", r"$vz_{tgt}$", r"$r_{tgt}$"]
        assert len(uhl_labels) == self.action_dim
        return uhl_labels
