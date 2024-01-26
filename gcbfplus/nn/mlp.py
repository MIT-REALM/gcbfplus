import flax.linen as nn

from .utils import default_nn_init, scaled_init, AnyFloat, HidSizes, ActFn, signal_last_enumerate


class MLP(nn.Module):
    hid_sizes: HidSizes
    act: ActFn = nn.relu
    act_final: bool = True
    use_layernorm: bool = False
    scale_final: float | None = None
    dropout_rate: float | None = None

    @nn.compact
    def __call__(self, x: AnyFloat, apply_dropout: bool = False) -> AnyFloat:
        nn_init = default_nn_init
        for is_last_layer, ii, hid_size in signal_last_enumerate(self.hid_sizes):
            if is_last_layer and self.scale_final is not None:
                x = nn.Dense(hid_size, kernel_init=scaled_init(nn_init(), self.scale_final))(x)
            else:
                x = nn.Dense(hid_size, kernel_init=nn_init())(x)

            no_activation = is_last_layer and not self.act_final
            if not no_activation:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate, deterministic=not apply_dropout)(x)
                if self.use_layernorm:
                    x = nn.LayerNorm()(x)
                x = self.act(x)
        return x
