import numpy as np
from scipy.io import loadmat

from smooth_3d import spinn
from src.x3v3 import smooth
from utils.transform import Lp_norm


def measuring(problem: str = "smooth", Kn: float = 1.0):
    d = 3

    if problem == "smooth":
        Nx = 80
        rank = 128
        ic = smooth(0.5, 10.0)

    print(f"{d}d {problem} problem, Kn: {Kn}, rank: {rank}, Nx: {Nx}")

    # t=0.1
    data = loadmat(f"data/x3v3/{problem}/Kn{Kn}_Nx{Nx}.mat")
    keys = ["Plot_X", "rho", "u1", "u2", "u3", "T"]
    x, *true = [data[key].squeeze() for key in keys]
    x = x[:, 0, 0]
    h = x[1] - x[0]
    # ic
    rho0, (ux0, uy0, uz0), T0 = ic.rho_u_temp(x, x, x)
    true0 = [rho0, ux0, uy0, uz0, T0]
    # spinn
    params = np.load(
        f"data/x3v3/{problem}/rank{rank}_params_Kn{Kn}.npy", allow_pickle=True
    )
    model = spinn(rank=rank, Kn=Kn)
    t = np.array([0.0, 0.1])
    rho, (ux, uy, uz), temp = model.rho_u_temp(params, t, x, x, x)
    pred = [rho, ux, uy, uz, temp]

    # error
    plus_one = [False, True, True, True, False]
    numerator = list(map(lambda x, y: Lp_norm(x[0] - y, h=h, d=d), pred, true0))
    denominator = list(
        map(lambda x, y: Lp_norm(x, h=h, d=d, plus_one=y), true0, plus_one)
    )
    err0 = list(map(lambda x, y: f"{x / y:.2e}", numerator, denominator))
    print(f"t=0.0 error: {err0}")

    numerator = list(map(lambda x, y: Lp_norm(x[1] - y, h=h, d=d), pred, true))
    denominator = list(
        map(lambda x, y: Lp_norm(x, h=h, d=d, plus_one=y), true, plus_one)
    )
    err = list(map(lambda x, y: f"{x / y:.2e}", numerator, denominator))
    print(f"t=0.1 error: {err}")

    print(
        f"conservation check. rho0: {rho0.sum() * h**3}, numeric: {true[0].sum() * h**3}, spinn0: {rho[0].sum() * h**3}, spinn1: {rho[1].sum() * h**3}"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(measuring)
