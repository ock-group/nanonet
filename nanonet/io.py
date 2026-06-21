"""
nanonet.io
----------
I/O helpers for sweep results: CSV export, pickle, and YAML config loading.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Network persistence
# ------------------------------------------------------------------

def save_network(net, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(net, f)


def load_network(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# Sweep result export
# ------------------------------------------------------------------

_ARRAY_KEYS = {
    "voltages", "currents", "conductances", "node_va",
    "evolution_rows", "evolution_fields",
}


def scalar_row(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in _ARRAY_KEYS}


def write_summary_csv(results: list[dict], path: str | Path) -> str:
    path = str(path)
    rows = [scalar_row(r) for r in results]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_iv_data_csv(results: list[dict], path: str | Path,
                      trim_at_transition: bool = True) -> str:
    """Export per-voltage I-V rows for each result in the list.

    One row per (result, voltage).  Columns: label, voltage_V, current_A, current_nA.
    """
    rows = []
    for r in results:
        lbl = (f"N={r['N']}_mean{r['mean_va_target']:.0f}"
               f"_sigma{r['sigma_va_target']:.0f}_seed{r['seed']}")
        v_end = r.get("transition_voltage_V", float("nan"))
        for v, i in zip(r["voltages"], r["currents"]):
            if trim_at_transition and np.isfinite(v_end) and float(v) > float(v_end) + 1e-9:
                continue
            rows.append({"label": lbl, "voltage_V": float(v),
                         "current_A": float(i), "current_nA": float(i) * 1e9})
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def write_evolution_csvs(results: list[dict], outdir: str | Path, case_name: str) -> None:
    """Write one per-voltage evolution CSV per result."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for r in results:
        rows = r.get("evolution_rows")
        if not rows:
            continue
        fields = r.get("evolution_fields") or list(rows[0].keys())
        tag = (f"{case_name}_N{r['N']}_mean{r['mean_va_target']:.0f}"
               f"_sigma{r['sigma_va_target']:.0f}_seed{r['seed']}")
        fv = r.get("void_fraction")
        if fv:
            tag += f"_fv{fv:.2f}"
        pd.DataFrame(rows, columns=fields).to_csv(
            outdir / f"evolution_{tag}.csv", index=False
        )


def save_results_pickle(bundle: Any, path: str | Path) -> str:
    path = str(path)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


def load_results_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# YAML config loading
# ------------------------------------------------------------------

def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML config and return a SweepConfig instance."""
    import yaml
    from nanonet.sweeps.parameter_sweep import SweepConfig

    with open(path) as f:
        raw = yaml.safe_load(f)

    netcfg = raw.get("network", {})
    thcfg = raw.get("threshold_distribution", {})
    elec = raw.get("electrical", {})
    vsweep = raw.get("voltage_sweep", {})

    cfg = SweepConfig()
    if "domain_size" in netcfg:
        ds = netcfg["domain_size"]
        cfg.L = (float(ds[0]), float(ds[1])) if hasattr(ds, "__len__") else float(ds)
    if "n_junctions" in netcfg:
        cfg.N = int(netcfg["n_junctions"])
    if "connection_radius" in netcfg:
        cfg.connection_radius = float(netcfg["connection_radius"])
    if "left_thresh" in netcfg:
        cfg.left_thresh = float(netcfg["left_thresh"])
    if "right_thresh" in netcfg:
        cfg.right_thresh = float(netcfg["right_thresh"])

    if "mean" in thcfg:
        cfg.mu_a = float(thcfg["mean"])
    if "std" in thcfg:
        cfg.std_a = float(thcfg["std"])
    if "min" in thcfg:
        cfg.va_min = float(thcfg["min"])
    if "max" in thcfg:
        cfg.va_max = float(thcfg["max"])

    if "edge_k" in elec:
        cfg.edge_k = float(elec["edge_k"])
    if "node_r_scale" in elec:
        cfg.node_r_scale = float(elec["node_r_scale"])
    if "r_floor" in elec:
        cfg.r_floor = float(elec["r_floor"])

    if "V_start" in vsweep:
        cfg.V_start = float(vsweep["V_start"])
    if "V_max" in vsweep:
        cfg.V_max = float(vsweep["V_max"])
    if "V_step" in vsweep:
        cfg.V_step = float(vsweep["V_step"])
    if "fit_window" in vsweep:
        cfg.fit_window = float(vsweep["fit_window"])

    if "seeds" in raw:
        cfg.seeds = [int(s) for s in raw["seeds"]]
    if "output" in raw and "root_folder" in raw["output"]:
        cfg.results_root = str(raw["output"]["root_folder"])

    return cfg
