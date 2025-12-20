from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SynthConfig:
    system: str
    n_traj: int
    time: int
    dt: float
    seed: int


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _toy2d_step(x: np.ndarray, _t: float) -> np.ndarray:
    rot = np.stack([-x[..., 1], x[..., 0]], axis=-1)
    drift = -0.1 * x + 0.25 * np.tanh(x)
    return rot + drift


def _vdp_step(mu: float) -> Callable[[np.ndarray, float], np.ndarray]:
    def f(x: np.ndarray, _t: float) -> np.ndarray:
        q = x[..., 0]
        p = x[..., 1]
        dq = p
        dp = mu * (1.0 - q**2) * p - q
        return np.stack([dq, dp], axis=-1)

    return f


def _lorenz_step(sigma: float, rho: float, beta: float) -> Callable[[np.ndarray, float], np.ndarray]:
    def f(x: np.ndarray, _t: float) -> np.ndarray:
        X = x[..., 0]
        Y = x[..., 1]
        Z = x[..., 2]
        dX = sigma * (Y - X)
        dY = X * (rho - Z) - Y
        dZ = X * Y - beta * Z
        return np.stack([dX, dY, dZ], axis=-1)

    return f


def _duffing_step(delta: float, alpha: float, beta: float, gamma: float, omega: float) -> Callable[[np.ndarray, float], np.ndarray]:
    def f(x: np.ndarray, t: float) -> np.ndarray:
        q = x[..., 0]
        p = x[..., 1]
        dq = p
        dp = -delta * p - alpha * q - beta * (q**3) + gamma * np.cos(omega * t)
        return np.stack([dq, dp], axis=-1)

    return f


def system_dt(system: str) -> float:
    s = system.lower()
    if s == "lorenz":
        return 0.01
    return 0.02


def _system_fn(system: str) -> Tuple[int, Callable[[np.ndarray, float], np.ndarray], Dict]:
    s = system.lower()
    if s == "toy2d":
        return 2, _toy2d_step, {"system": "toy2d"}
    if s == "vdp":
        mu = 1.0
        return 2, _vdp_step(mu), {"system": "vdp", "mu": mu}
    if s == "lorenz":
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        return 3, _lorenz_step(sigma, rho, beta), {"system": "lorenz", "sigma": sigma, "rho": rho, "beta": beta}
    if s == "duffing":
        delta, alpha, beta, gamma, omega = 0.2, -1.0, 1.0, 0.3, 1.2
        return 2, _duffing_step(delta, alpha, beta, gamma, omega), {
            "system": "duffing",
            "delta": delta,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "omega": omega,
        }
    raise ValueError(f"unknown system: {system}")


def generate(cfg: SynthConfig) -> Tuple[np.ndarray, Dict]:
    dim, step, sys_meta = _system_fn(cfg.system)
    rng = _rng(cfg.seed)

    x = 0.5 * rng.normal(size=(cfg.n_traj, dim)).astype(np.float32)
    traj = np.zeros((cfg.n_traj, cfg.time, dim), dtype=np.float32)
    traj[:, 0, :] = x

    for t in range(cfg.time - 1):
        tt = float(t) * float(cfg.dt)
        dx = step(x, tt).astype(np.float32)
        x = x + float(cfg.dt) * dx
        traj[:, t + 1, :] = x

    meta = {
        "synth": {
            "system": cfg.system,
            "n_traj": int(cfg.n_traj),
            "time": int(cfg.time),
            "dim": int(dim),
            "dt": float(cfg.dt),
            "seed": int(cfg.seed),
            "system_params": sys_meta,
        }
    }
    return traj, meta


def generate_system_data(
    system: str,
    *,
    n_traj: int = 256,
    time: int = 201,
    dt: float | None = None,
    seed: int = 0,
    train_frac: float = 0.8,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float, Dict]:
    """
    Convenience wrapper producing (x_t, x_t1) pairs suitable for training.

    Returns:
      (train_x_t, train_x_t1), (test_x_t, test_x_t1), dt, meta
    """

    if dt is None:
        dt = system_dt(system)
    cfg = SynthConfig(system=str(system), n_traj=int(n_traj), time=int(time), dt=float(dt), seed=int(seed))
    traj, meta = generate(cfg)

    x_t = traj[:, :-1, :].reshape(-1, traj.shape[-1])
    x_t1 = traj[:, 1:, :].reshape(-1, traj.shape[-1])

    n = x_t.shape[0]
    n_train = int(float(train_frac) * float(n))
    train = (x_t[:n_train], x_t1[:n_train])
    test = (x_t[n_train:], x_t1[n_train:])
    return train, test, float(dt), meta
