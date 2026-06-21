"""
nanonet.analysis.spectral
--------------------------
Spectral metrics from the full node+edge system Laplacian:
  * effective_resistance : source-to-drain effective resistance [Ω]
  * spectral_metrics     : algebraic connectivity, spectral gap ratio
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

EIG_REL_TOL = 1e-9
R_MIN = 1.0


def _build_full_system(net, activated_nodes: set) -> dict | None:
    """Assemble the full node+edge Laplacian (no BCs) for the bridging subset."""
    working = set(activated_nodes)
    G_sub0 = net.G.subgraph(working)
    asrc = [n for n in net.source_nodes if n in working]
    adrn = [n for n in net.drain_nodes if n in working]
    if not asrc or not adrn:
        return None

    reach_s: set = set()
    for s in asrc:
        if s in G_sub0:
            reach_s |= nx.node_connected_component(G_sub0, s)
    reach_d: set = set()
    for d in adrn:
        if d in G_sub0:
            reach_d |= nx.node_connected_component(G_sub0, d)
    work = reach_s & reach_d
    if not work:
        return None

    G_sub = net.G.subgraph(work)
    asrc = [n for n in asrc if n in work]
    adrn = [n for n in adrn if n in work]
    all_el = set(asrc) | set(adrn)
    internal = [n for n in sorted(work) if n not in all_el]
    electrode = [n for n in sorted(work) if n in all_el]
    N_int = len(internal)
    SZ = 2 * N_int + len(electrode)
    int_idx = {n: i for i, n in enumerate(internal)}
    el_idx = {n: i for i, n in enumerate(electrode)}

    def row_in(n):
        return 2 * int_idx[n] if n in int_idx else 2 * N_int + el_idx[n]

    def row_out(n):
        return 2 * int_idx[n] + 1 if n in int_idx else 2 * N_int + el_idx[n]

    depth = {n: 999 for n in work}
    q = list(asrc)
    for s in asrc:
        depth[s] = 0
    h = 0
    while h < len(q):
        c = q[h]; h += 1
        for nb in G_sub.neighbors(c):
            if nb in depth and depth[nb] == 999:
                depth[nb] = depth[c] + 1
                q.append(nb)

    M = lil_matrix((SZ, SZ))
    for n in internal:
        g = 1.0 / max(net.G.nodes[n].get("R_node", 0.0), R_MIN)
        ri, ro = row_in(n), row_out(n)
        M[ri, ri] += g; M[ro, ro] += g
        M[ri, ro] -= g; M[ro, ri] -= g
    for i, j in G_sub.edges():
        ge = 1.0 / max(net.G[i][j]["R_edge"], R_MIN)
        sn, dn = (i, j) if depth[i] <= depth[j] else (j, i)
        rso, rdi = row_out(sn), row_in(dn)
        M[rso, rso] += ge; M[rdi, rdi] += ge
        M[rso, rdi] -= ge; M[rdi, rso] -= ge

    return {
        "M": M, "SZ": SZ, "N_int": N_int,
        "row_in": row_in, "row_out": row_out,
        "asrc": asrc, "adrn": adrn,
        "internal": internal, "electrode": electrode,
        "G_sub": G_sub,
    }


def effective_resistance(net, activated_nodes: set, V_test: float = 1.0) -> float:
    """Source-to-drain effective resistance from the full system matrix.

    Returns np.nan if source and drain are not bridged.
    """
    sysd = _build_full_system(net, activated_nodes)
    if sysd is None:
        return np.nan
    M = sysd["M"].tolil()
    SZ = sysd["SZ"]
    row_in, row_out = sysd["row_in"], sysd["row_out"]
    asrc, adrn = sysd["asrc"], sysd["adrn"]

    b = np.zeros(SZ)
    for s in asrc:
        for r in (row_in(s), row_out(s)):
            M[r, :] = 0; M[r, r] = 1.0; b[r] = V_test
    for d in adrn:
        for r in (row_in(d), row_out(d)):
            M[r, :] = 0; M[r, r] = 1.0; b[r] = 0.0
    try:
        phi = spsolve(M.tocsr(), b)
        if not np.all(np.isfinite(phi)):
            return np.nan
    except Exception:
        return np.nan

    G_sub = sysd["G_sub"]
    I_tot = 0.0
    for d in adrn:
        for nb in G_sub.neighbors(d):
            ge = 1.0 / max(net.G[d][nb]["R_edge"], R_MIN)
            I_tot += ge * (phi[row_out(nb)] - phi[row_in(d)])
    I_tot = abs(I_tot)
    return (V_test / I_tot) if I_tot > 0 else np.nan


def spectral_metrics(net, activated_nodes: set) -> tuple[float, float, int]:
    """Algebraic connectivity and spectral gap ratio of the full Laplacian.

    Returns
    -------
    (algebraic_connectivity, spectral_gap_ratio, n_zero_eigenvalues)
    All np.nan / 0 if no source-drain bridge.
    """
    sysd = _build_full_system(net, activated_nodes)
    if sysd is None:
        return np.nan, np.nan, 0
    M = sysd["M"].toarray()
    ev = np.linalg.eigvalsh(0.5 * (M + M.T))
    ev = np.sort(ev)
    emax = ev[-1]
    if emax <= 0:
        return np.nan, np.nan, 0
    tol = EIG_REL_TOL * emax
    n_zero = int(np.sum(ev < tol))
    nonzero = ev[ev >= tol]
    alg_conn = float(nonzero[0]) if len(nonzero) else np.nan
    gap_ratio = float(alg_conn / emax) if np.isfinite(alg_conn) else np.nan
    return alg_conn, gap_ratio, n_zero
