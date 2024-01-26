import flax.linen as nn
import functools as ft
import jax.numpy as jnp
import jraph
import jax.tree_util as jtu

from typing import Type, NamedTuple, Callable, Tuple

from ..utils.typing import EdgeAttr, Node, Array
from ..utils.graph import GraphsTuple
from .mlp import MLP, default_nn_init
from .utils import safe_get

save_attn = False


def save_set_attn(v):
    global save_attn
    save_attn = v


class GNNUpdate(NamedTuple):
    message: Callable[[EdgeAttr, Node, Node], Array]
    aggregate: Callable[[Array, Array, int], Array]
    update: Callable[[Node, Array], Array]

    def __call__(self, graph: GraphsTuple) -> GraphsTuple:
        assert graph.n_node.shape == tuple()
        node_feats_send = jtu.tree_map(lambda n: safe_get(n, graph.senders), graph.nodes)
        node_feats_recv = jtu.tree_map(lambda n: safe_get(n, graph.receivers), graph.nodes)

        # message passing
        edges = self.message(graph.edges, node_feats_send, node_feats_recv)

        # aggregate messages
        aggr_msg = jtu.tree_map(lambda edge: self.aggregate(edge, graph.receivers, graph.nodes.shape[0]), edges)

        # update nodes
        new_node_feats = self.update(graph.nodes, aggr_msg)

        return graph._replace(nodes=new_node_feats)


class GNNLayer(nn.Module):
    msg_net_cls: Type[nn.Module]
    aggr_net_cls: Type[nn.Module]
    update_net_cls: Type[nn.Module]
    msg_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, graph: GraphsTuple) -> GraphsTuple:
        def message(edge_feats: EdgeAttr, sender_feats: Node, receiver_feats: Node) -> Array:
            feats = jnp.concatenate([edge_feats, sender_feats, receiver_feats], axis=-1)
            feats = self.msg_net_cls()(feats)
            feats = nn.Dense(self.msg_dim, kernel_init=default_nn_init())(feats)
            return feats

        def update(node_feats: Node, msgs: Array) -> Array:
            feats = jnp.concatenate([node_feats, msgs], axis=-1)
            feats = self.update_net_cls()(feats)
            feats = nn.Dense(self.out_dim, kernel_init=default_nn_init())(feats)
            return feats

        def aggregate(msgs: Array, recv_idx: Array, num_segments: int) -> Array:
            gate_feats = self.aggr_net_cls()(msgs)
            gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
            attn = jraph.segment_softmax(gate_feats, segment_ids=recv_idx, num_segments=num_segments)
            assert attn.shape[0] == msgs.shape[0]

            aggr_msg = jraph.segment_sum(attn[:, None] * msgs, segment_ids=recv_idx, num_segments=num_segments)
            return aggr_msg

        update_fn = GNNUpdate(message, aggregate, update)
        return update_fn(graph)


class GNN(nn.Module):
    msg_dim: int
    hid_size_msg: Tuple[int, ...]
    hid_size_aggr: Tuple[int, ...]
    hid_size_update: Tuple[int, ...]
    out_dim: int
    n_layers: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, node_type: int = None, n_type: int = None) -> Array:
        for i in range(self.n_layers):
            out_dim = self.out_dim if i == self.n_layers - 1 else self.msg_dim
            msg_net = ft.partial(MLP, hid_sizes=self.hid_size_msg, act=nn.relu, act_final=False, name="msg")
            attn_net = ft.partial(MLP, hid_sizes=self.hid_size_aggr, act=nn.relu, act_final=False, name="attn")
            update_net = ft.partial(MLP, hid_sizes=self.hid_size_update, act=nn.relu, act_final=False, name="update")
            gnn_layer = GNNLayer(
                msg_net_cls=msg_net,
                aggr_net_cls=attn_net,
                update_net_cls=update_net,
                msg_dim=self.msg_dim,
                out_dim=out_dim,
            )
            graph = gnn_layer(graph)
        if node_type is None:
            return graph.nodes
        else:
            return graph.type_nodes(node_type, n_type)
