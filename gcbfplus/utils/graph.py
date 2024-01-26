from typing import Generic, NamedTuple, TypeVar, get_type_hints

import einops as ei
import jax.numpy as jnp
import jax.tree_util as jtu
from jax._src.tree_util import GetAttrKey

from ..utils.typing import Any, Array, Bool, Float, Int
from .utils import merge01

_State = TypeVar("_State")
_EnvState = TypeVar("_EnvState")


class EdgeBlock(NamedTuple):
    edge_feats: Float[Array, "n_recv n_send n_edge_feat"]
    edge_mask: Bool[Array, "n_recv n_send"]
    ids_recv: Int[Array, "n_recv"]
    ids_send: Int[Array, "n_send"]

    @property
    def n_recv(self):
        assert self.edge_feats.shape[0] == self.edge_mask.shape[0] == len(self.ids_recv)
        return len(self.ids_recv)

    @property
    def n_send(self):
        assert self.edge_feats.shape[1] == self.edge_mask.shape[1] == len(self.ids_send)
        return len(self.ids_send)

    @property
    def n_edges(self):
        return self.n_recv * self.n_send

    def make_edges(self, pad_id: int, edge_mask: Bool[Array, "n_recv n_send"] = None):
        id_recv_rep = ei.repeat(self.ids_recv, "n_recv -> n_recv n_send", n_send=self.n_send)
        id_send_rep = ei.repeat(self.ids_send, "n_send -> n_recv n_send", n_recv=self.n_recv)
        edge_mask = self.edge_mask if edge_mask is None else edge_mask
        e_recvs = merge01(jnp.where(edge_mask, id_recv_rep, pad_id))
        e_sends = merge01(jnp.where(edge_mask, id_send_rep, pad_id))
        e_edge_feats = merge01(self.edge_feats)
        assert e_recvs.shape == e_sends.shape == e_edge_feats.shape[:1] == (self.n_edges,)

        return e_edge_feats, e_recvs, e_sends


@jtu.register_pytree_with_keys_class
class GraphsTuple(tuple, Generic[_State, _EnvState]):
    n_node: Int[Array, "n_graph"]  # number of nodes in each subgraph
    n_edge: Int[Array, "n_graph"]  # number of edges in each subgraph

    nodes: Float[Array, "sum_n_node ..."]  # node features
    edges: Float[Array, "sum_n_edge ..."]  # edge features
    states: _State  # node state features
    receivers: Int[Array, "sum_n_edge"]
    senders: Int[Array, "sum_n_edge"]
    node_type: Int[Array, "sum_n_node"]  # by default, 0 is agent, -1 is padding
    env_states: _EnvState  # environment state features
    connectivity: Int[Array, "sum_n_node sum_n_node"] = None  # desired connectivity matrix

    def __new__(
        cls,
        n_node,
        n_edge,
        nodes,
        edges,
        states: _State,
        receivers,
        senders,
        node_type,
        env_states: _EnvState,
        connectivity=None,
    ):
        tup = (n_node, n_edge, nodes, edges, states, receivers, senders, node_type, env_states, connectivity)
        self = tuple.__new__(cls, tup)
        self.n_node = n_node
        self.n_edge = n_edge
        self.nodes = nodes
        self.edges = edges
        self.states = states
        self.receivers = receivers
        self.senders = senders
        self.node_type = node_type
        self.env_states = env_states
        self.connectivity = connectivity
        return self

    def tree_flatten_with_keys(self):
        flat_contents = [(GetAttrKey(k), getattr(self, k)) for k in get_type_hints(GraphsTuple).keys()]
        aux_data = None
        return flat_contents, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def is_single(self) -> bool:
        return self.n_node.ndim == 0

    @property
    def n_graphs(self) -> int:
        if self.n_node.ndim == 0:
            return 1
        assert len(self.n_node) == len(self.n_edge)
        return len(self.n_node)

    @property
    def batch_shape(self):
        return self.n_node.shape

    def type_nodes(self, type_idx: int, n_type: int) -> Float[Array, "... n_type n_feats"]:
        assert self.nodes.ndim == 2
        n_feats = self.nodes.shape[1]

        n_is_type = self.node_type == type_idx
        idx = jnp.cumsum(n_is_type) - 1

        sum_n_type = self.n_graphs * n_type
        type_feats = jnp.zeros((sum_n_type, n_feats))
        type_feats = type_feats.at[idx, :].add(n_is_type[:, None] * self.nodes)

        out = type_feats.reshape(self.batch_shape + (n_type, n_feats))
        return out

    def type_states(self, type_idx: int, n_type: int) -> Float[Array, "... n_type n_states"]:
        assert self.states.ndim == 2
        n_states = self.states.shape[1]

        n_is_type = self.node_type == type_idx
        idx = jnp.cumsum(n_is_type) - 1

        sum_n_type = self.n_graphs * n_type
        type_feats = jnp.zeros((sum_n_type, n_states))
        type_feats = type_feats.at[idx, :].add(n_is_type[:, None] * self.states)

        out = type_feats.reshape(self.batch_shape + (n_type, n_states))
        return out

    def __str__(self) -> str:
        node_repr = str(self.nodes)
        edge_repr = str(self.edges)

        return "n_node={}, n_edge={}, \n{}\n---------\n{}\n-------\n{}\n  |  \n{}".format(
            self.n_node, self.n_edge, node_repr, edge_repr, self.senders, self.receivers
        )

    def _replace(
        self,
        n_node=None,
        n_edge=None,
        nodes=None,
        edges=None,
        states: _State = None,
        receivers=None,
        senders=None,
        node_type=None,
        env_states: _EnvState = None,
        connectivity=None,
    ) -> "GraphsTuple":
        return GraphsTuple(
            self.n_node if n_node is None else n_node,
            self.n_edge if n_edge is None else n_edge,
            self.nodes if nodes is None else nodes,
            self.edges if edges is None else edges,
            self.states if states is None else states,
            self.receivers if receivers is None else receivers,
            self.senders if senders is None else senders,
            self.node_type if node_type is None else node_type,
            self.env_states if env_states is None else env_states,
            self.connectivity if connectivity is None else connectivity,
        )

    def without_edge(self):
        return GraphsTuple(
            self.n_node,
            self.n_edge,
            self.nodes,
            None,
            self.states,
            self.receivers,
            self.senders,
            self.node_type,
            self.env_states,
            self.connectivity,
        )


class GetGraph(NamedTuple):
    nodes: Float[Array, "n_nodes n_node_feat"]  # node features
    node_type: Int[Array, "n_nodes"]  # by default, 0 is agent
    edge_blocks: list[EdgeBlock]
    env_states: Any
    states: Float[Array, "n_nodes n_state"]  # node state features
    connectivity: Int[Array, "n_node n_node"] = None  # desired connectivity matrix

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def node_dim(self) -> int:
        return self.nodes.shape[1]

    @property
    def state_dim(self) -> int:
        return self.states.shape[1]

    def to_padded(self) -> GraphsTuple:
        # make a dummy node for creating fake self edges.
        node_feat_dummy = jnp.zeros(self.node_dim)
        node_feats_pad = jnp.concatenate([self.nodes, node_feat_dummy[None]], axis=0)
        node_type_pad = jnp.concatenate([self.node_type, jnp.full(1, -1)], axis=0)
        state_dummy = jnp.ones(self.state_dim) * -1
        state_pad = jnp.concatenate([self.states, state_dummy[None]], axis=0)

        # Construct edge list.
        pad_id = self.n_nodes
        edge_feats_lst, recv_list, send_list = [], [], []
        for edge_block in self.edge_blocks:
            e_edge_feats, e_recvs, e_sends = edge_block.make_edges(pad_id)
            edge_feats_lst.append(e_edge_feats)
            recv_list.append(e_recvs)
            send_list.append(e_sends)
        e_edge_feats = jnp.concatenate(edge_feats_lst, axis=0)
        e_recv, e_send = jnp.concatenate(recv_list), jnp.concatenate(send_list)

        n_nodes, n_edges = self.n_nodes + 1, e_edge_feats.shape[0]
        assert e_recv.shape == e_send.shape == (n_edges,)
        n_nodes = jnp.array(n_nodes, dtype=jnp.int32)
        n_edges = jnp.array(n_edges, dtype=jnp.int32)

        return GraphsTuple(
            n_nodes,
            n_edges,
            node_feats_pad,
            e_edge_feats,
            state_pad,
            e_recv,
            e_send,
            node_type_pad,
            self.env_states,
            self.connectivity,
        )
