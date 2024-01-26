import functools as ft
import flax.linen as nn
import jax
import jax.numpy as jnp

from typing import Type

from ...nn.mlp import MLP
from ...nn.gnn import GNN
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params
from ...utils.graph import GraphsTuple


class StateFn(nn.Module):
    gnn_cls: Type[GNN]
    aggr_cls: Type[nn.Module]
    head_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Array:
        # get node features
        x = self.gnn_cls()(obs, node_type=0, n_type=n_agents)

        # aggregate information using attention
        gate_feats = self.aggr_cls()(x)
        gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
        attn = jax.nn.softmax(gate_feats, axis=-1)
        x = jnp.sum(attn[:, None] * x, axis=0)

        # pass through head class
        x = self.head_cls()(x)
        x = nn.Dense(1, kernel_init=default_nn_init())(x)

        return x


class ValueNet:

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, gnn_layers: int = 1):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.value_gnn = ft.partial(
            GNN,
            msg_dim=64,
            hid_size_msg=(128, 128),
            hid_size_aggr=(128, 128),
            hid_size_update=(128, 128),
            out_dim=64,
            n_layers=gnn_layers
        )
        self.value_attn = ft.partial(
            MLP,
            hid_sizes=(128, 128),
            act=nn.relu,
            act_final=False,
            name='ValueAttn'
        )
        self.value_head = ft.partial(
            MLP,
            hid_sizes=(128, 128),
            act=nn.relu,
            act_final=False,
            name='ValueHead'
        )
        # self.net = StateFn(, _nu=1)
        self.net = StateFn(
            gnn_cls=self.value_gnn,
            aggr_cls=self.value_attn,
            head_cls=self.value_head
        )

    def get_value(self, params: Params, obs: GraphsTuple) -> Array:
        values = self.net.apply(params, obs, self.n_agents)
        return values.squeeze()
