import flax.linen as nn
import jax.numpy as jnp

from typing import Any, Callable, Literal, Sequence, Iterable, Generator, TypeVar
from jaxtyping import Array, Bool, Float, Int, Shaped


ActFn = Callable[[Array], Array]
PRGNKey = Float[Array, '2']
AnyFloat = Float[Array, '*']
Shape = tuple[int, ...]
InitFn = Callable[[PRGNKey, Shape, Any], Any]
HidSizes = Sequence[int]


_Elem = TypeVar("_Elem")


default_nn_init = nn.initializers.xavier_uniform


def scaled_init(initializer: nn.initializers.Initializer, scale: float) -> nn.initializers.Initializer:
    def scaled_init_inner(*args, **kwargs) -> AnyFloat:
        return scale * initializer(*args, **kwargs)

    return scaled_init_inner


ActLiteral = Literal["relu", "tanh", "elu", "swish", "silu", "gelu", "softplus"]


def get_act_from_str(act_str: ActLiteral) -> ActFn:
    act_dict: dict[Literal, ActFn] = dict(
        relu=nn.relu, tanh=nn.tanh, elu=nn.elu, swish=nn.swish, silu=nn.silu, gelu=nn.gelu, softplus=nn.softplus
    )
    return act_dict[act_str]


def signal_last_enumerate(it: Iterable[_Elem]) -> Generator[tuple[bool, int, _Elem], None, None]:
    iterable = iter(it)
    count = 0
    ret_var = next(iterable)
    for val in iterable:
        yield False, count, ret_var
        count += 1
        ret_var = val
    yield True, count, ret_var


def safe_get(arr, idx):
    return arr.at[idx].get(mode='fill', fill_value=jnp.nan)
