import jax
import jax.numpy as np
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import numpy as onp

from src.nn import Siren
from src.x3v3 import x3v3, smooth
from utils.transform import trapezoidal_rule


class spinn(x3v3):
    def __init__(
        self,
        T=0.1,
        X=0.5,
        V=10.0,
        Nv=256,
        width=128,
        depth=3,
        rank=256,
        w0=10.0,
        ic=smooth,
        Kn=None,
    ):
        super().__init__(T, X, V, Kn)
        layers = [1] + [width for _ in range(depth - 1)] + [rank]
        self.init, self.apply = Siren(layers, w0)
        self.rank = rank
        self.ic = ic(X, V)
        self.v, self.w = trapezoidal_rule(Nv, -V, V)
        self.wv = self.w * self.v
        self.wvv = self.wv * self.v

    def _loss(self, params, domain):
        t, x, y, z, vx, vy, vz = domain

        def t_mapsto_f(t):
            return self.f(params, t, x, y, z, vx, vy, vz)

        def x_mapsto_f(x):
            return self.f(params, t, x, y, z, vx, vy, vz)

        def y_mapsto_f(y):
            return self.f(params, t, x, y, z, vx, vy, vz)

        def z_mapsto_f(z):
            return self.f(params, t, x, y, z, vx, vy, vz)

        # pde
        f, f_t = jax.jvp(t_mapsto_f, (t,), (np.ones(t.shape),))
        f_x = jax.jvp(x_mapsto_f, (x,), (np.ones(x.shape),))[1]
        f_y = jax.jvp(y_mapsto_f, (y,), (np.ones(y.shape),))[1]
        f_z = jax.jvp(z_mapsto_f, (z,), (np.ones(z.shape),))[1]
        maxwellian = self.maxwellian(*self.rho_u_temp(params, t, x, y, z), vx, vy, vz)
        pde = (
            f_t
            + vx[:, None, None] * f_x
            + vy[:, None] * f_y
            + vz * f_z
            - self.nu * (maxwellian - f)
        )
        # initial condition
        f0 = self.ic.f0(x, y, z, vx, vy, vz)
        ic = self.f(params, np.array([0.0]), x, y, z, vx, vy, vz) - f0
        # boundary condition
        fx = self.f(params, t, self.X, y, z, vx, vy, vz)
        fy = self.f(params, t, x, self.X, z, vx, vy, vz)
        fz = self.f(params, t, x, y, self.X, vx, vy, vz)
        bc_x = fx[:, 1] - fx[:, 0]
        bc_y = fy[:, :, 1] - fy[:, :, 0]
        bc_z = fz[:, :, :, 1] - fz[:, :, :, 0]
        return (pde, ic, bc_x, bc_y, bc_z), (
            f,
            f0,
            np.abs(fx).mean(1),
            np.abs(fy).mean(2),
            np.abs(fz).mean(3),
        )


def main(seed: int = 0, Kn: float = None, rank: int = 128):
    nIter = 1 * 10**5
    w0 = 10
    lr = optax.cosine_decay_schedule(1e-4 / w0, nIter)
    opt = optax.lion(lr, weight_decay=0)
    T, X, V = 0.1, 0.5, 6.0
    N = [12] * 7
    seed = jr.key(seed)

    assert Kn is not None, "set Kn!"
    print(f"3d smooth, Kn={Kn}, rank={rank}")
    model = spinn(T=T, X=X, V=V, w0=w0, Kn=Kn, rank=rank)
    domain = [np.linspace(*bd, n) for bd, n in zip(model.bd, N)]

    train_key, init_key = jr.split(seed)
    init_key = jr.split(init_key, model.dim + 5)
    init_params = (
        jr.uniform(init_key[0], (5, model.rank), minval=-1, maxval=1)
        * np.sqrt(6 / model.rank),
        [model.init(k) for k in init_key[1:5]],
        [model.init(k) for k in init_key[5:]],
        [np.array([0.0, 0.0, 0.0]), np.array(1.0)],
    )

    opt_params, *logs = model.train(opt, domain, init_params, train_key, nIter)

    print(opt_params[-1])

    # save network parameters
    onp.save(
        f"data/x3v3/smooth/rank{rank}_params_Kn{Kn}.npy",
        onp.asarray(opt_params, dtype="object"),
    )

    # loss trajectory
    _, ax0 = plt.subplots(figsize=(4, 4))
    ax0.semilogy(logs[0], label=f"PDE Loss:{logs[0][-1]:.3e}")
    ax0.semilogy(logs[1], label=f"IC Loss:{logs[1][-1]:.3e}")
    ax0.semilogy(logs[2], label=f"BC Loss:{logs[2][-1]:.3e}")
    ax0.set_xlabel("100 iterations")
    ax0.set_title("Test Mean Squared Loss")
    ax0.legend()
    plt.tight_layout()
    plt.savefig(f"figures/x3v3/smooth/loss_Kn{Kn}.png", format="png")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
