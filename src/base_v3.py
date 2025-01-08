from functools import partial

import jax
from jax import random, grad
import jax.numpy as np
import optax
from tqdm import trange

from utils.transform import maxwellian3d
from utils.training import unif_sampler


class base_v3:
    def __init__(
        self,
        T: float = 0.1,
        X: float = 0.5,
        V: float = 10,
        Kn: float = None,
    ):
        super().__init__()
        self.T = np.array([0, T])
        self.X = np.array([-X, X])
        self.V = np.array([-V, V])
        self.nu = 1 / Kn
        self.maxwellian = maxwellian3d

    def rho_u_temp(self, params, t, x, y, z):
        moments = self.moments(params, t, x, y, z)
        m, p, E = moments[0, ...], moments[1:4, ...], moments[4:, ...]
        rho = np.maximum(m, 1e-3)
        u = p / rho
        temp = (E.sum(0) / rho - (u**2).sum(0)) / 3
        temp = np.maximum(temp, 1e-3)
        return rho, u, temp

    def sampling(self, key, domain):
        keys = random.split(key, self.dim)
        update = [unif_sampler(k, d, *b) for k, d, b in zip(keys, domain, self.bd)]
        return update

    @partial(jax.jit, static_argnums=(0, 1))
    def step(self, optimizer, params, state, key, domain, *args):
        # update params
        g, _ = grad(self.loss, has_aux=True)(params, domain, *args)
        updates, state = optimizer.update(g, state, params)
        params = optax.apply_updates(params, updates)
        # sample domain
        key, subkey = random.split(key)
        domain = self.sampling(subkey, domain)
        return params, state, key, domain

    @partial(jax.jit, static_argnums=(0,))
    def logger(self, params):
        loss, (loss_r, loss_ic, loss_bc) = self.loss(params, self.domain_te)
        return loss, (loss_r, loss_ic, loss_bc)

    def train(self, optimizer, domain, params, key, nIter):
        min_loss = np.inf
        domain = [*domain]
        state = optimizer.init(params)
        pde_log, ic_log, bc_log = [], [], []
        for it in (pbar := trange(1, nIter + 1)):
            params, state, key, domain = self.step(
                optimizer, params, state, key, domain
            )
            if it % 100 == 0:
                loss, (loss_r, loss_ic, loss_bc) = self.logger(params)
                pde_log.append(loss_r), ic_log.append(loss_ic), bc_log.append(loss_bc)
                if loss < min_loss:
                    min_loss = loss
                    opt_params = params
                    pbar.set_postfix({"loss": f"{loss:.3e}"})
                if np.sum(np.isnan(loss)) > 0:
                    break
        return opt_params, pde_log, ic_log, bc_log
