"""
nanonet.analysis.sweep
-----------------------
Per-voltage network-evolution sweep: activation metrics, Kirchhoff solve,
current-distribution statistics, and connectivity structure.
"""

from __future__ import annotations

import csv
import numpy as np
import networkx as nx


def conductance_matrix(net, activated_nodes: set, R_MIN: float = 1.0):
    """Build the edge-only Laplacian (conductance) matrix over activated nodes.

    Returns (G_dense_array, ordered_node_list).
    """
    active_list = sorted(activated_nodes)
    local_idx = {v: k for k, v in enumerate(active_list)}
    N = len(active_list)
    G = np.zeros((N, N))
    for i, j in net.G.edges():
        if i in local_idx and j in local_idx:
            g = 1.0 / max(net.G[i][j]["R_edge"], R_MIN)
            ki, kj = local_idx[i], local_idx[j]
            G[ki, ki] += g; G[kj, kj] += g
            G[ki, kj] -= g; G[kj, ki] -= g
    return G, active_list


def _edge_currents_at(net, activated_nodes: set, V_applied: float):
    """Return (total_current, node_potentials, edge_currents) for one voltage."""
    total_I, phi = net._solve_kirchhoff(activated_nodes, V_applied)
    ec = net._reconstruct_edge_currents(activated_nodes, V_applied, phi)
    return total_I, phi, ec


def sweep(
    net,
    V_start: float,
    V_max: float,
    V_step: float,
    *,
    current_frac: float = 0.01,
    G_voltages=None,
    G_out_prefix: str | None = None,
    effective_resistance: bool = True,
    algebraic_connectivity: bool = True,
) -> dict:
    """Run a full voltage sweep and collect per-voltage metrics.

    Parameters
    ----------
    net : NanoparticleNetwork
    V_start, V_max, V_step : float
        Voltage sweep parameters [V].
    current_frac : float
        Backbone edge threshold: edges carrying >= current_frac * I_max.
    G_voltages : list of float, optional
        Voltages at which to export the conductance matrix to disk.
    G_out_prefix : str, optional
        File-path prefix for conductance-matrix output files.
    effective_resistance : bool
        Compute source-to-drain effective resistance (one extra linear solve).
    algebraic_connectivity : bool
        Compute Fiedler value and spectral gap ratio (expensive: dense eig).

    Returns
    -------
    dict with keys:
        rows              : list of per-voltage metric dicts
        fieldnames        : column names (for CSV export)
        edge_currents_by_V: {voltage: {(i,j): current}}
        percolation_V     : float or None
        n_total_nodes     : int
        n_total_edges     : int
    """
    if not net.source_nodes or not net.drain_nodes:
        raise ValueError("Call identify_sources_drains() (or build()) first.")

    G_voltages_set = set(round(float(v), 10) for v in (G_voltages or []))
    rows: list = []
    edge_currents_by_V: dict = {}
    percolation_V = None
    n_total_nodes = net.G.number_of_nodes()
    n_total_edges = net.G.number_of_edges()

    V = V_start
    while V <= V_max + 1e-10:
        Vr = round(V, 10)
        activated_nodes = {n for n in net.G.nodes() if net.G.nodes[n]["Vth"] <= Vr}
        activated_edges = [(i, j) for i, j in net.G.edges()
                           if i in activated_nodes and j in activated_nodes]

        G_act = net.G.subgraph(activated_nodes)
        src_comp: set = set()
        for s in net.source_nodes:
            if s in G_act:
                src_comp |= nx.node_connected_component(G_act, s)
        drn_comp: set = set()
        for d in net.drain_nodes:
            if d in G_act:
                drn_comp |= nx.node_connected_component(G_act, d)
        connected = len(src_comp & drn_comp) > 0
        if connected and percolation_V is None:
            percolation_V = Vr

        # Component structure
        comp_sizes = sorted(
            (len(c) for c in nx.connected_components(G_act)), reverse=True
        )
        num_components = len(comp_sizes)
        largest_cc_nodes = comp_sizes[0] if comp_sizes else 0
        second_cc_nodes = comp_sizes[1] if len(comp_sizes) > 1 else 0
        largest_cc_fraction = (
            largest_cc_nodes / len(activated_nodes) if activated_nodes else 0.0
        )
        mean_finite_cc = (
            float(np.mean(comp_sizes[1:])) if num_components > 1 else 0.0
        )

        total_current, phi, ec = _edge_currents_at(net, activated_nodes, Vr)
        edge_currents_by_V[Vr] = ec

        if ec:
            mags = np.array([abs(c) for c in ec.values()])
            max_mag = mags.max()
            backbone_edges = int(np.sum(mags >= current_frac * max_mag))
            s = mags.sum()
            participation = float((s * s) / np.sum(mags * mags)) if s > 0 else 0.0

            src_set = set(net.source_nodes)
            I_cc = 0.0
            for (i, j), c in ec.items():
                if i in src_set and j not in src_set:
                    I_cc += c
                elif j in src_set and i not in src_set:
                    I_cc -= c
            I_cc = abs(I_cc)

            mean_mag = float(mags.mean())
            max_to_mean = float(max_mag / mean_mag) if mean_mag > 0 else 0.0
            cv_current = float(mags.std() / mean_mag) if mean_mag > 0 else 0.0
            sorted_mags = np.sort(mags)
            n_e = len(sorted_mags)
            cum = np.cumsum(sorted_mags)
            gini_current = (
                float(
                    (2.0 * np.sum(np.arange(1, n_e + 1) * sorted_mags)
                     - (n_e + 1) * cum[-1])
                    / (n_e * cum[-1])
                )
                if cum[-1] > 0 else 0.0
            )
            k = max(1, int(np.ceil(0.10 * n_e)))
            top10_fraction = float(sorted_mags[-k:].sum() / cum[-1]) if cum[-1] > 0 else 0.0
        else:
            backbone_edges = participation = I_cc = 0
            max_to_mean = cv_current = gini_current = top10_fraction = 0.0

        if Vr in G_voltages_set and G_out_prefix and activated_nodes:
            Gmat, active_list = conductance_matrix(net, activated_nodes)
            np.save(f"{G_out_prefix}_V{Vr:g}.npy", Gmat)
            with open(f"{G_out_prefix}_V{Vr:g}_nodeindex.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["matrix_index", "node_id", "is_source", "is_drain"])
                for k_idx, nid in enumerate(active_list):
                    w.writerow([k_idx, nid, int(nid in net.source_nodes),
                                int(nid in net.drain_nodes)])
            with open(f"{G_out_prefix}_V{Vr:g}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row_index", "col_index", "row_node", "col_node", "conductance_S"])
                for ii in range(Gmat.shape[0]):
                    for jj in range(Gmat.shape[0]):
                        if Gmat[ii, jj] != 0.0:
                            w.writerow([ii, jj, active_list[ii], active_list[jj], Gmat[ii, jj]])

        conductance = total_current / Vr if Vr > 1e-12 else 0.0

        R_eff = alg_conn = gap_ratio = np.nan
        if activated_nodes and (effective_resistance or algebraic_connectivity):
            from nanonet.analysis.spectral import (
                effective_resistance as _eff_res,
                spectral_metrics as _spec,
            )
            if effective_resistance:
                R_eff = _eff_res(net, activated_nodes)
            if algebraic_connectivity:
                alg_conn, gap_ratio, _ = _spec(net, activated_nodes)

        rows.append({
            "V": Vr,
            "activated_nodes": len(activated_nodes),
            "activated_edges": len(activated_edges),
            "conducting_nodes": len(phi),
            "conducting_edges": len(ec),
            "source_drain_connected": int(connected),
            "total_current_A": total_current,
            "total_current_chargeconserving_A": I_cc,
            "conductance_S": conductance,
            "backbone_edges": backbone_edges,
            "participation_ratio": participation,
            "num_components": num_components,
            "largest_cc_nodes": largest_cc_nodes,
            "second_cc_nodes": second_cc_nodes,
            "largest_cc_fraction": largest_cc_fraction,
            "mean_finite_cc": mean_finite_cc,
            "current_cv": cv_current,
            "current_gini": gini_current,
            "current_max_to_mean": max_to_mean,
            "current_top10_fraction": top10_fraction,
            "effective_resistance_ohm": R_eff,
            "algebraic_connectivity": alg_conn,
            "spectral_gap_ratio": gap_ratio,
        })
        V = round(V + V_step, 10)

    fieldnames = [
        "V", "activated_nodes", "activated_edges", "conducting_nodes",
        "conducting_edges", "source_drain_connected", "total_current_A",
        "total_current_chargeconserving_A", "conductance_S",
        "backbone_edges", "participation_ratio",
        "num_components", "largest_cc_nodes", "second_cc_nodes",
        "largest_cc_fraction", "mean_finite_cc",
        "current_cv", "current_gini", "current_max_to_mean",
        "current_top10_fraction",
        "effective_resistance_ohm", "algebraic_connectivity", "spectral_gap_ratio",
    ]
    return {
        "rows": rows,
        "fieldnames": fieldnames,
        "edge_currents_by_V": edge_currents_by_V,
        "percolation_V": percolation_V,
        "n_total_nodes": n_total_nodes,
        "n_total_edges": n_total_edges,
    }


def write_csv(result: dict, path: str) -> str:
    """Write full per-voltage metrics table to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=result["fieldnames"])
        w.writeheader()
        for row in result["rows"]:
            w.writerow(row)
    return path


def write_iv_csv(result: dict, path: str) -> str:
    """Write focused I-V table (V, current, conductance) to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V", "current_A", "current_chargeconserving_A", "conductance_S"])
        for r in result["rows"]:
            w.writerow([r["V"], r["total_current_A"],
                        r["total_current_chargeconserving_A"], r["conductance_S"]])
    return path


def write_edge_currents_csv(
    net,
    result: dict,
    path: str,
    conducting_only: bool = True,
    voltages=None,
) -> str:
    """Write per-edge signed currents with node positions to CSV."""
    pos = net.positions
    ecbv = result["edge_currents_by_V"]
    Vset = (set(round(float(v), 10) for v in voltages)
            if voltages is not None else None)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V", "node_i", "node_j", "x_i", "y_i", "x_j", "y_j",
                    "current_A", "abs_current_A"])
        for Vr in sorted(ecbv.keys()):
            if Vset is not None and Vr not in Vset:
                continue
            for (i, j), c in ecbv[Vr].items():
                w.writerow([Vr, i, j,
                            f"{pos[i][0]:.6f}", f"{pos[i][1]:.6f}",
                            f"{pos[j][0]:.6f}", f"{pos[j][1]:.6f}",
                            f"{c:.8e}", f"{abs(c):.8e}"])
    return path
