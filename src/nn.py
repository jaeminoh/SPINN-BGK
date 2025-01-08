import jax.numpy as np
import jax.random as jr


def Siren(layers: list[int], w0: float) -> tuple[callable]:
    """
    MLP with sine activation function.
    Specialized initialization scheme is necessary.
    Reference: Sitzmann et al. SIREN.
    """

    def siren_init(key, d_in, d_out, is_first=False):
        if is_first:
            scale = 1 / d_in
            W = scale * jr.uniform(key, (d_in, d_out), minval=-1, maxval=1)
            scale = np.sqrt(1 / d_in)
            b = scale * jr.uniform(key, (d_out,), minval=-1, maxval=1)
            return W, b
        else:
            scale = np.sqrt(6 / d_in) / w0
            W = scale * jr.uniform(key, (d_in, d_out), minval=-1, maxval=1)
            scale = np.sqrt(1 / d_in)
            b = scale * jr.uniform(key, (d_out,), minval=-1, maxval=1)
            return W, b

    def init(rng_key):
        _, *keys = jr.split(rng_key, len(layers))
        params = [siren_init(keys[0], layers[0], layers[1], is_first=True)] + list(
            map(siren_init, keys[1:], layers[1:-1], layers[2:])
        )
        return params

    def activation(x):
        return np.sin(w0 * x)

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply
