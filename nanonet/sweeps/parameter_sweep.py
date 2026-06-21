"""
nanonet.sweeps.parameter_sweep
-------------------------------
Parameter sweep driver.  Cases vary the main network parameters:

    L       — domain size (fixed in standard cases)
    N       — number of junctions
    fv      — void fraction (Case R)
    mu_a    — mean activation voltage
    std_a   — std of activation voltage distribution (sigma)

Cases
-----
Case 1 : vary std_a (σ) at fixed mu_a and N
Case 2 : vary mu_a at fixed std_a and N (multiple σ widths)
Case 3 : vary N × mu_a grid at fixed std_a
Case 4 : vary N at fixed mu_a and std_a
Case R : vary void fraction fv at fixed N, mu_a, std_a

Each case returns a list of result dicts (one per (parameters, seed) combo)
with scalars and per-voltage arrays.  Use the write_* helpers in nanonet.io
to export these to CSV.

Standalone runner
-----------------
    python -m nanonet.sweeps.parameter_sweep
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence


# ------------------------------------------------------------------
# Configuration dataclass
# ------------------------------------------------------------------

@dataclass
class SweepConfig:
    """All parameters controlling a parameter sweep.

    Physics parameters
    ------------------
    L : float or (float, float)
        Domain size.
    N : int
        Number of junctions (default case).
    fv : float
        Default void fraction (used in Case R sweep).
    mu_a : float
        Default mean activation voltage [V].
    std_a : float
        Default std of activation voltage distribution [V].
    va_min, va_max : float
        Hard clip bounds on sampled activation voltages.
    edge_k : float
        Edge-resistance constant.
    node_r_scale : float
        Node-resistance constant.
    r_floor : float
        Node-resistance floor voltage.
    connection_radius : float
        Edge-formation radius.
    source_frac : float
        Fraction of domain width forming the source electrode strip (left side).
        Default 0.15 (left 15 %).
    drain_frac : float
        Fraction of domain width forming the drain electrode strip (right side).
        Default 0.15 (right 15 %).

    Sweep parameters
    ----------------
    V_start, V_max, V_step : float
        Voltage sweep range.
    seeds : list of int
        Random seeds.
    max_workers : int
        Multiprocessing worker count.
    results_root : str
        Output root directory.

    Sweep-specific parameter grids
    -------------------------------
    sigma_values, fixed_mean          : run_sweep_vary_std  — vary σ at fixed μ
    mean_values, sigma_values_mean    : run_sweep_vary_mean — vary μ at each σ
    N_values, fixed_mean_N, fixed_std_N : run_sweep_vary_N — vary N at fixed μ, σ
    void_fractions, N_voids,
      mu_a_voids, std_a_voids         : run_sweep_vary_voids — vary void fraction
    void_radius                       : void circle radius for the void sweep
    """

    # Domain / physics defaults
    L: float = 1.0
    N: int = 500
    fv: float = 0.0
    mu_a: float = 6.0
    std_a: float = 3.0
    va_min: float = 0.0
    va_max: float = 20.0
    edge_k: float = 2.0e10
    node_r_scale: float = 5.0e8
    r_floor: float = 0.5
    connection_radius: float = 0.15
    source_frac: float = 0.15
    drain_frac: float = 0.15
    strict_N: bool = False

    # Voltage sweep
    V_start: float = 0.0
    V_max: float = 16.0
    V_step: float = 0.5

    # Execution
    seeds: list = field(default_factory=lambda: [41, 51, 61, 71, 81])
    max_workers: int = 2
    results_root: str = "IV_results"

    # run_sweep_vary_std: vary σ at fixed μ and N
    sigma_values: list = field(default_factory=lambda: [1.0, 3.0, 5.0, 7.0])
    fixed_mean: float = 8.0

    # run_sweep_vary_mean: vary μ at each σ
    mean_values: list = field(default_factory=lambda: [4.0, 6.0, 8.0, 10.0])
    sigma_values_mean: list = field(default_factory=lambda: [1.0, 3.0, 5.0, 7.0])

    # run_sweep_vary_N: vary N at fixed μ and σ
    N_values: list = field(default_factory=lambda: [200, 400, 600, 800])
    fixed_mean_N: float = 6.0
    fixed_std_N: float = 3.0

    # run_sweep_vary_voids: vary void fraction at fixed N, μ, σ
    void_fractions: list = field(default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    N_voids: int = 500
    mu_a_voids: float = 6.0
    std_a_voids: float = 3.0
    void_radius: float = 0.08

    def seed_for(self, i: int) -> int:
        return self.seeds[i % len(self.seeds)]


# ------------------------------------------------------------------
# Internal: build one network and run one IV+sweep
# ------------------------------------------------------------------

def _build_and_run(
    N: int,
    mu_a: float,
    std_a: float,
    seed: int,
    cfg: SweepConfig,
    fv: float = 0.0,
) -> dict:
    from nanonet.core.network import NanoparticleNetwork
    from nanonet.analysis.sweep import sweep as _sweep

    net = NanoparticleNetwork(
        L=cfg.L, N=N, fv=fv, mu_a=mu_a, std_a=std_a,
        connection_radius=cfg.connection_radius,
        edge_k=cfg.edge_k, node_r_scale=cfg.node_r_scale, r_floor=cfg.r_floor,
        source_frac=cfg.source_frac, drain_frac=cfg.drain_frac,
        va_min=cfg.va_min, va_max=cfg.va_max,
        void_radius=cfg.void_radius,
        strict_N=cfg.strict_N,
    )
    net.build(seed=seed)

    iv = net.iv_curve(V_start=cfg.V_start, V_max=cfg.V_max, V_step=cfg.V_step)
    evo = _sweep(
        net, cfg.V_start, cfg.V_max, cfg.V_step,
        effective_resistance=False,
        algebraic_connectivity=False,
    )

    node_va = np.array([net.G.nodes[nd]["Vth"] for nd in net.G.nodes()], float)
    peak_idx = int(np.argmax(iv["currents"]))
    perc_V = iv["threshold_voltage"]
    trans_V = _transition_voltage_from_evo(evo)
    fit = _fit_power_law(
        np.asarray(iv["voltages"], float),
        np.asarray(iv["currents"], float),
        v_step=cfg.V_step,
        v_transition=trans_V,
    )

    return {
        "N": int(net.n_junctions),
        "N_requested": int(N),
        "mean_va_target": float(mu_a),
        "sigma_va_target": float(std_a),
        "seed": int(seed),
        "void_fraction": float(fv),
        "voltages": iv["voltages"],
        "currents": iv["currents"],
        "conductances": iv["conductances"],
        "node_va": node_va,
        "n_nodes_actual": int(net.n_junctions),
        "n_edges": int(net.G.number_of_edges()),
        "n_sources": int(len(net.source_nodes)),
        "n_drains": int(len(net.drain_nodes)),
        "sampled_mean_va": float(np.mean(node_va)),
        "sampled_std_va": float(np.std(node_va)),
        "peak_current_A": float(iv["currents"][peak_idx]),
        "peak_voltage_V": float(iv["voltages"][peak_idx]),
        "percolation_voltage_V": float(perc_V) if perc_V is not None else float("nan"),
        "transition_voltage_V": float(trans_V) if trans_V is not None else float("nan"),
        "max_conductance_S": float(np.max(iv["conductances"])),
        "fit_V_T_V": float(fit["V_T"]) if fit["success"] else float("nan"),
        "fit_zeta": float(fit["zeta"]) if fit["success"] else float("nan"),
        "fit_R2": float(fit["R2"]) if fit["success"] else float("nan"),
        "fit_success": bool(fit["success"]),
        "fit_reason": fit.get("reason", ""),
        "evolution_rows": evo["rows"],
        "evolution_fields": evo["fieldnames"],
    }


def _worker(args):
    N, mu_a, std_a, seed, cfg, fv = args
    return _build_and_run(N, mu_a, std_a, seed, cfg, fv)


def _pool_workers(cfg: SweepConfig, n_tasks: int) -> int:
    return max(1, min(cfg.max_workers, n_tasks))


# ------------------------------------------------------------------
# Power-law fit (I = A * (V - V_T)^zeta)
# ------------------------------------------------------------------

def _first_positive_span(v, i, span=1.0):
    v = np.asarray(v); i = np.asarray(i)
    pos = i > 0
    start = None
    for idx, is_pos in enumerate(pos):
        if is_pos:
            if start is None:
                start = idx
            if v[idx] - v[start] >= span:
                return start
        else:
            start = None
    return None


def _fit_power_law(v_arr, i_arr, v_step=0.5, v_transition=None) -> dict:
    """Fit I = A (V - V_T)^zeta over Phase II only.

    Phase I  : zero-current region (V < percolation onset).
    Phase II : nonlinear region from percolation onset up to v_transition.
    Phase III: quasi-linear region beyond v_transition (excluded from fit).

    v_transition is the voltage at which 90% of nodes have activated.
    If not supplied, the entire conducting range is used (legacy behaviour).
    """
    nan = float("nan")
    perc_idx = _first_positive_span(v_arr, i_arr, span=max(1.0, v_step))
    if perc_idx is None:
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                    reason="no_conduction")
    V_T_guess = float(v_arr[perc_idx])
    # Upper bound: Phase II ends at the 90%-activation transition voltage.
    # If unavailable, fall back to the full sweep range.
    if v_transition is not None and np.isfinite(v_transition) and v_transition > V_T_guess:
        v_upper = float(v_transition)
    else:
        v_upper = float(v_arr[-1])
    mask = (v_arr >= V_T_guess) & (v_arr <= v_upper) & (i_arr > 0)
    V_fit = np.asarray(v_arr[mask], float)
    I_fit = np.asarray(i_arr[mask], float)
    if len(V_fit) < 4:
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                    reason=f"too_few_points({len(V_fit)})")

    def loglog_slope(V_T):
        dV = V_fit - V_T
        good = dV > 0
        if good.sum() < 4:
            return None
        x = np.log(dV[good]); y = np.log(I_fit[good])
        A_mat = np.vstack([x, np.ones_like(x)]).T
        (zeta, logA), *_ = np.linalg.lstsq(A_mat, y, rcond=None)
        y_pred = zeta * x + logA
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
        return float(zeta), float(logA), float(R2)

    v_min = V_fit.min()
    lo = max(0.0, V_T_guess - 2.0 * max(v_step, 0.5))
    hi = v_min - 1e-6
    if hi <= lo:
        hi = lo + 1e-6
    best = None
    for V_T_try in np.linspace(lo, hi, 40):
        out = loglog_slope(V_T_try)
        if out is None:
            continue
        zeta, logA, R2 = out
        if best is None or (np.isfinite(R2) and R2 > best[3]):
            best = (V_T_try, zeta, logA, R2)

    if best is None:
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                    reason="loglog_failed")
    V_T_fit, zeta_fit, logA_fit, R2 = best
    A_fit = float(np.exp(logA_fit))
    if not np.isfinite(zeta_fit) or not np.isfinite(R2):
        return dict(success=False, V_T=nan, zeta=nan, A=nan, R2=nan, reason="nonfinite")
    if R2 < 0.80:
        return dict(success=False, V_T=float(V_T_fit), zeta=float(zeta_fit),
                    A=A_fit, R2=float(R2), reason=f"poor_fit_R2={R2:.3f}")
    if zeta_fit <= 0.1 or zeta_fit >= 10.0:
        return dict(success=False, V_T=float(V_T_fit), zeta=float(zeta_fit),
                    A=A_fit, R2=float(R2), reason="zeta_out_of_range")
    return dict(success=True, V_T=float(V_T_fit), zeta=float(zeta_fit),
                A=A_fit, R2=float(R2), reason="ok")


def _transition_voltage_from_evo(evo: dict) -> float | None:
    rows = evo.get("rows") or []
    if not rows:
        return None
    n_total = evo.get("n_total_nodes") or max(
        (r.get("activated_nodes", 0) or 0) for r in rows
    )
    if not n_total:
        return None
    target = 0.90 * float(n_total)
    V = np.array([r.get("V") for r in rows], float)
    an = np.array([r.get("activated_nodes", np.nan) for r in rows], float)
    order = np.argsort(V)
    V, an = V[order], an[order]
    reached = np.where(an >= target)[0]
    return float(V[reached[0]]) if len(reached) else None


# ------------------------------------------------------------------
# Aggregate helpers
# ------------------------------------------------------------------

_ARRAY_KEYS = {
    "voltages", "currents", "conductances", "node_va",
    "evolution_rows", "evolution_fields",
}


def _scalar_row(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in _ARRAY_KEYS}


def make_fit_table(results: list[dict], case_name: str) -> pd.DataFrame:
    """Build a per-run summary DataFrame of fitted transport parameters."""
    rows = []
    for r in results:
        rows.append({
            "case": case_name,
            "N": r["N"],
            "mean_Va_target_V": r["mean_va_target"],
            "sigma_Va_target_V": r["sigma_va_target"],
            "seed": r["seed"],
            "void_fraction": r.get("void_fraction", 0.0),
            "sampled_mean_Va_V": r["sampled_mean_va"],
            "sampled_sigma_Va_V": r["sampled_std_va"],
            "percolation_voltage_V": r["percolation_voltage_V"],
            "transition_voltage_V": r.get("transition_voltage_V", float("nan")),
            "fit_V_T_V": r["fit_V_T_V"],
            "fit_zeta": r["fit_zeta"],
            "fit_R2": r["fit_R2"],
            "fit_success": r["fit_success"],
            "peak_current_A": r["peak_current_A"],
            "max_conductance_S": r["max_conductance_S"],
        })
    return pd.DataFrame(rows)


def aggregate_fit_table(results: list[dict], group_key: str, group_col: str) -> pd.DataFrame:
    """Aggregate V_T and zeta by a swept parameter (mean ± std over seeds)."""
    rows = []
    for val in sorted(set(r[group_key] for r in results)):
        grp = [r for r in results if r[group_key] == val]
        vt = np.array([r["fit_V_T_V"] for r in grp
                       if r.get("fit_success") and np.isfinite(r["fit_V_T_V"])], float)
        zt = np.array([r["fit_zeta"] for r in grp
                       if r.get("fit_success") and np.isfinite(r["fit_zeta"])], float)
        rows.append({
            group_col: val,
            "n_seeds": int(max(len(vt), len(zt))),
            "VT_mean": float(np.mean(vt)) if len(vt) else float("nan"),
            "VT_std": float(np.std(vt, ddof=1)) if len(vt) > 1 else 0.0,
            "zeta_mean": float(np.mean(zt)) if len(zt) else float("nan"),
            "zeta_std": float(np.std(zt, ddof=1)) if len(zt) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Sweep runners (return list of result dicts)
# ------------------------------------------------------------------

def run_sweep_vary_std(cfg: SweepConfig | None = None) -> list[dict]:
    """Vary std_a (σ) at fixed mu_a and N.

    Sweeps over cfg.sigma_values with cfg.fixed_mean held constant.
    All seeds in cfg.seeds are run at every σ value (for error bars).
    Returns a flat list sorted by (sigma_va_target, seed).
    """
    if cfg is None:
        cfg = SweepConfig()
    tasks = [
        (cfg.N, cfg.fixed_mean, sigma, seed, cfg, 0.0)
        for sigma in cfg.sigma_values
        for seed in cfg.seeds
    ]
    with Pool(processes=_pool_workers(cfg, len(tasks)), maxtasksperchild=4) as pool:
        results = list(pool.imap_unordered(_worker, tasks, chunksize=1))
    for r in results:
        r["case"] = "vary_std"
    results.sort(key=lambda r: (r["sigma_va_target"], r["seed"]))
    return results


def run_sweep_vary_mean(cfg: SweepConfig | None = None) -> dict[float, list[dict]]:
    """Vary mu_a at each sigma in cfg.sigma_values_mean.

    Returns a dict keyed by sigma value; each entry is a flat list of
    results sorted by (mean_va_target, seed).
    """
    if cfg is None:
        cfg = SweepConfig()
    results_by_sigma: dict = {}
    for sigma in cfg.sigma_values_mean:
        tasks = [
            (cfg.N, mean, sigma, seed, cfg, 0.0)
            for mean in cfg.mean_values
            for seed in cfg.seeds
        ]
        with Pool(processes=_pool_workers(cfg, len(tasks)), maxtasksperchild=4) as pool:
            results = list(pool.imap_unordered(_worker, tasks, chunksize=1))
        for r in results:
            r["case"] = f"vary_mean_sigma{sigma:g}"
        results.sort(key=lambda r: (r["mean_va_target"], r["seed"]))
        results_by_sigma[float(sigma)] = results
    return results_by_sigma


def run_sweep_vary_N(cfg: SweepConfig | None = None) -> list[dict]:
    """Vary N at fixed mu_a and std_a.

    Sweeps over cfg.N_values with cfg.fixed_mean_N and cfg.fixed_std_N
    held constant. All seeds in cfg.seeds are run at every N (for error
    bars). Returns a flat list sorted by (N, seed).
    """
    if cfg is None:
        cfg = SweepConfig()
    tasks = [
        (N, cfg.fixed_mean_N, cfg.fixed_std_N, seed, cfg, 0.0)
        for N in cfg.N_values
        for seed in cfg.seeds
    ]
    with Pool(processes=_pool_workers(cfg, len(tasks)), maxtasksperchild=4) as pool:
        results = list(pool.imap_unordered(_worker, tasks, chunksize=1))
    for r in results:
        r["case"] = "vary_N"
    results.sort(key=lambda r: (r["N"], r["seed"]))
    return results


def run_sweep_vary_voids(cfg: SweepConfig | None = None) -> list[dict]:
    """Vary void fraction fv at fixed N, mu_a, std_a.

    Sweeps over cfg.void_fractions using cfg.N_voids, cfg.mu_a_voids,
    and cfg.std_a_voids. All seeds in cfg.seeds are run at every fv
    (for error bars). Returns a flat list sorted by (void_fraction, seed).
    """
    if cfg is None:
        cfg = SweepConfig()
    tasks = [
        (cfg.N_voids, cfg.mu_a_voids, cfg.std_a_voids, seed, cfg, fv)
        for fv in cfg.void_fractions
        for seed in cfg.seeds
    ]
    with Pool(processes=_pool_workers(cfg, len(tasks)), maxtasksperchild=4) as pool:
        results = list(pool.imap_unordered(_worker, tasks, chunksize=1))
    for r in results:
        r["case"] = "vary_voids"
    results.sort(key=lambda r: (r["void_fraction"], r["seed"]))
    return results


def run_all_cases(cfg: SweepConfig | None = None) -> dict:
    """Run all sweeps and return a results bundle.

    Returns
    -------
    dict with keys: vary_std, vary_mean, vary_N, vary_voids
    """
    if cfg is None:
        cfg = SweepConfig()
    print("Running sweep: vary std_a (σ) ...")
    r_std = run_sweep_vary_std(cfg)
    print("Running sweep: vary mu_a (μ) ...")
    r_mean = run_sweep_vary_mean(cfg)
    print("Running sweep: vary N ...")
    r_N = run_sweep_vary_N(cfg)
    print("Running sweep: vary void fraction ...")
    r_voids = run_sweep_vary_voids(cfg)
    return {
        "vary_std": r_std,
        "vary_mean": r_mean,
        "vary_N": r_N,
        "vary_voids": r_voids,
    }


# ------------------------------------------------------------------
# Standalone CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    from datetime import datetime

    cfg = SweepConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(cfg.results_root) / f"sweep_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = run_all_cases(cfg)

    for sweep_name, results in bundle.items():
        if isinstance(results, dict):
            # vary_mean returns {sigma: [results]}
            for sigma, res_list in results.items():
                df = make_fit_table(res_list, f"{sweep_name}_sigma{sigma:g}")
                df.to_csv(outdir / f"{sweep_name}_sigma{sigma:g}_fit_table.csv", index=False)
        else:
            df = make_fit_table(results, sweep_name)
            df.to_csv(outdir / f"{sweep_name}_fit_table.csv", index=False)

    print(f"Results saved to {outdir}")
