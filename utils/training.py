import jax
import jax.random as jr


def unif_sampler(key, domain, minval, maxval) -> jax.Array:
    update = jr.uniform(key, (domain.shape), minval=minval, maxval=maxval)
    return update
