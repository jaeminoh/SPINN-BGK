import jax
import jax.numpy as jnp

from utils.transform import maxwellian3d, to_column, gaussian
from src.base_v3 import base_v3


class x3v3(base_v3):
    dim: int = 7

    def __init__(
        self,
        T: float = 0.1,
        X: float = 0.5,
        V: float = 10,
        Kn: float = None,
    ):
        super().__init__(T, X, V, Kn)
        self.alpha = 1e-0
        self.eps = 1e-3
        self.bd = ((1 + 1e-7) * self.T,) + (self.X,) * 3 + ((1 + 1e-7) * self.V,) * 3
        self.domain_te = (
            (jnp.linspace(*self.T, 12),)
            + (jnp.linspace(*self.X, 12),) * 3
            + (jnp.linspace(*self.V, 12),) * 3
        )

    def _f_eq(self, params, t, x, y, z):
        weights, params, *_ = params
        outputs = [self.apply(p, i) for p, i in zip(params, to_column([t, x, y, z]))]
        _f_eq = jnp.einsum("az,bz,cz,dz,ez->abcde", weights, *outputs)
        rho, u, temp = _f_eq[0, ...], _f_eq[1:4, ...], _f_eq[4, ...]
        rho, temp = jnp.exp(-rho), jnp.exp(-temp)
        return rho, u, temp

    def _f_neq(self, params, t, x, y, z, vx, vy, vz):
        *_, params, ut = params
        _f_tx = [self.apply(p, i) for p, i in zip(params[:-3], to_column([t, x, y, z]))]
        u, t = ut
        _f_v = [
            self.apply(p, i) * gaussian(u_, t, i)
            for p, u_, i in zip(params[-3:], u, to_column([vx, vy, vz]))
        ]
        _f_neq = _f_tx + _f_v
        return _f_neq

    def f(self, params, t, x, y, z, vx, vy, vz):
        rho, u, temp = self._f_eq(params, t, x, y, z)
        f_eq = self.maxwellian(rho, u, temp, vx, vy, vz)
        _f_neq = self._f_neq(params, t, x, y, z, vx, vy, vz)
        f_neq = jnp.einsum("az,bz,cz,dz,ez,fz,gz->abcdefg", *_f_neq)
        f = f_eq + self.alpha * f_neq
        return f.squeeze()

    def loss(self, params, domain):
        loss, w = self._loss(params, domain)
        loss_r, loss_ic, *loss_bc = jax.tree_map(
            lambda loss, w: jnp.mean(
                (loss / jax.lax.stop_gradient(jnp.abs(w) + self.eps)) ** 2
            ),
            loss,
            w,
        )
        loss_bc = sum(loss_bc)
        loss = loss_r + 1e3 * loss_ic + loss_bc
        return loss, (loss_r, loss_ic, loss_bc)

    def moments_neq(self, params, t, x, y, z):
        symbol = "az,bz,cz,dz,z,z,z->abcd"
        # intermediate tensors
        tensors = self._f_neq(params, t, x, y, z, *[self.v] * 3)
        tensor_tx, tensor_v = tensors[:-3], tensors[-3:]
        int_f = jax.tree_map(lambda fv: jnp.dot(self.w, fv), tensor_v)
        int_vf = jax.tree_map(lambda fv: jnp.dot(self.wv, fv), tensor_v)
        int_v2f = jax.tree_map(lambda fv: jnp.dot(self.wvv, fv), tensor_v)
        # moments
        m000 = jnp.einsum(symbol, *tensor_tx, int_f[0], int_f[1], int_f[2])
        m100 = jnp.einsum(symbol, *tensor_tx, int_vf[0], int_f[1], int_f[2])
        m010 = jnp.einsum(symbol, *tensor_tx, int_f[0], int_vf[1], int_f[2])
        m001 = jnp.einsum(symbol, *tensor_tx, int_f[0], int_f[1], int_vf[2])
        m200 = jnp.einsum(symbol, *tensor_tx, int_v2f[0], int_f[1], int_f[2])
        m020 = jnp.einsum(symbol, *tensor_tx, int_f[0], int_v2f[1], int_f[2])
        m002 = jnp.einsum(symbol, *tensor_tx, int_f[0], int_f[1], int_v2f[2])
        moments = jnp.stack([m000, m100, m010, m001, m200, m020, m002])
        return moments

    def moments(self, params, t, x, y, z):
        r"""
        $m, p, E = \iiint f (1, v, |v|^2) dv$.
        m_{3,0,0} = \rho (u_x^3 + 3u_xT).
        """
        # moments
        rho, (ux, uy, uz), temp = self._f_eq(params, t, x, y, z)
        m000 = jnp.ones_like(rho)
        m100 = ux
        m010 = uy
        m001 = uz
        m200 = temp + ux**2
        m020 = temp + uy**2
        m002 = temp + uz**2
        moments = rho * jnp.stack([m000, m100, m010, m001, m200, m020, m002])
        moments += self.alpha * self.moments_neq(params, t, x, y, z)
        return moments


class initial_condition:
    def __init__(self, X, V):
        self.X = jnp.array([-X, X])
        self.V = jnp.array([-V, V])
        self.maxwellian = maxwellian3d

    def rho_u_temp(self, x, y, z):
        return self.rho(x, y, z), self.u(x, y, z), self.temp(x, y, z)

    def f0(self, x, y, z, vx, vy, vz):  # (Nx^2, Nv^3)
        rho, u, temp = self.rho_u_temp(x, y, z)
        return self.maxwellian(rho, u, temp, vx, vy, vz)


class smooth(initial_condition):
    def __init__(self, X: float = 0.5, V: float = 10.0):
        super().__init__(X, V)

    def rho(self, x, y, z):
        x, y = x[:, None, None], y[:, None]  # x: (Nx,1,1); y: (Ny,1)
        pi2 = 2 * jnp.pi 
        return 1 + 0.5 * jnp.sin(pi2 * x) * jnp.sin(pi2 * y) * jnp.sin(pi2 * z)

    def u(self, x, y, z):
        zeros = jnp.zeros((x.size, y.size, z.size))
        u = jnp.stack([zeros] * 3)
        return u

    def temp(self, x, y, z):
        return jnp.ones((x.size, y.size, z.size))