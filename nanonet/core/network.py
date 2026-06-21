"""
nanonet.core.network
--------------------
NanoparticleNetwork: graph-based model of a nanoparticle necklace film.

Key parameters
--------------
L       : float or (float, float)
    Domain side length (square) or (width, height) in normalised units.
N       : int
    Number of junctions (nodes).
fv      : float
    Void fraction in [0, 1).  0 = no voids (default).  Voids are placed
    as fixed-radius circles; nodes and edges that fall inside a void are
    excluded.
mu_a    : float
    Mean of the activation-voltage (Vth) distribution [V].
std_a   : float
    Standard deviation of the activation-voltage distribution [V].
source_frac : float
    Fraction of domain width that forms the source electrode strip on the
    left side.  Nodes with x ≤ source_frac * width are sources.
    Default 0.15 (left 15 % of domain).
drain_frac : float
    Fraction of domain width that forms the drain electrode strip on the
    right side.  Nodes with x ≥ (1 - drain_frac) * width are drains.
    Default 0.15 (right 15 % of domain).

All other parameters (edge resistance constant, etc.) are keyword arguments
with sensible defaults matching the production config.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class NanoparticleNetwork:
    """Graph-based nanoparticle necklace network model.

    Parameters
    ----------
    L : float or tuple of float
        Domain size.  A single float gives a square domain (L, L).
    N : int
        Number of junctions requested.  A small number may be pruned if
        they have fewer than 2 neighbours.
    fv : float
        Void-area fraction (0 = dense film, no voids).
    mu_a : float
        Mean activation voltage [V].
    std_a : float
        Standard deviation of activation voltage distribution [V].
    connection_radius : float
        Maximum inter-junction distance for an edge to form (domain units).
    edge_k : float
        Edge-resistance constant: R_edge = edge_k * distance.
    node_r_scale : float
        Node-resistance constant: R_node = node_r_scale * max(Vth, r_floor).
        Set to 0 for edge-only model.
    r_floor : float
        Floor voltage used in node-resistance calculation.
    source_frac : float
        Fraction of domain width that is the source electrode strip (left side).
        Nodes with x ≤ source_frac * width become source electrodes.
        Default 0.15.
    drain_frac : float
        Fraction of domain width that is the drain electrode strip (right side).
        Nodes with x ≥ (1 - drain_frac) * width become drain electrodes.
        Default 0.15.
    va_min, va_max : float
        Hard clip bounds on the sampled activation voltages.
    void_radius : float
        Radius of each void circle (domain units).  Used when fv > 0.
    strict_N : bool
        If True, the builder iterates until exactly N degree-≥2 nodes survive
        pruning.  Nodes that would be pruned are replaced with new random nodes
        until the count is exact.  Default False (N is a target, not a
        guarantee).
    """

    def __init__(
        self,
        L: float | tuple[float, float] = 1.0,
        N: int = 500,
        fv: float = 0.0,
        mu_a: float = 4.0,
        std_a: float = 1.0,
        *,
        connection_radius: float = 0.15,
        edge_k: float = 2.0e10,
        node_r_scale: float = 5.0e8,
        r_floor: float = 0.5,
        source_frac: float = 0.15,
        drain_frac: float = 0.15,
        va_min: float = 0.0,
        va_max: float = 20.0,
        void_radius: float = 0.08,
        strict_N: bool = False,
    ):
        if isinstance(L, (int, float)):
            self.domain = (float(L), float(L))
        else:
            self.domain = (float(L[0]), float(L[1]))

        self.N = int(N)
        self.fv = float(fv)
        self.mu_a = float(mu_a)
        self.std_a = float(std_a)
        self.connection_radius = float(connection_radius)
        self.edge_k = float(edge_k)
        self.node_r_scale = float(node_r_scale)
        self.r_floor = float(r_floor)
        self.source_frac = float(source_frac)
        self.drain_frac = float(drain_frac)
        # Derived thresholds used by _identify_electrodes; kept as properties
        # so that changing source_frac / drain_frac after __init__ also works.
        self.va_min = float(va_min)
        self.va_max = float(va_max)
        self.void_radius = float(void_radius)
        self.strict_N = bool(strict_N)

        self.G: nx.Graph = nx.Graph()
        self.positions: np.ndarray | None = None
        self.source_nodes: list[int] = []
        self.drain_nodes: list[int] = []
        self.voids: list[dict] = []

        # legacy single-node aliases
        self.source_node: int | None = None
        self.drain_node: int | None = None
        # expose n_junctions for downstream compat
        self.n_junctions: int = 0

    # ------------------------------------------------------------------
    # Public build API
    # ------------------------------------------------------------------

    def build(self, seed: int | None = None) -> "NanoparticleNetwork":
        """Generate positions, graph, and identify electrodes in one call.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        self  (so you can chain: net = NanoparticleNetwork(...).build(seed=42))
        """
        self._place_voids(seed)
        if self.strict_N:
            self._build_strict_N(seed)
        else:
            self._generate_positions(seed)
            self._build_graph(seed)
            self._prune_low_degree()
        self._identify_electrodes()
        return self

    def _build_strict_N(self, seed: int | None) -> None:
        """Build a graph with exactly self.N degree-≥2 nodes.

        Iteratively places nodes, builds edges, prunes degree-<2 nodes, then
        adds replacement nodes for any that were pruned — repeating until
        exactly N valid junctions remain.
        """
        from scipy.spatial import cKDTree

        target = self.N
        rng_pos = np.random.default_rng(seed)
        rng_vth = np.random.default_rng((seed + 1000) if seed is not None else None)

        # Start with the requested number of positions
        pts = list(self._sample_positions(target, rng_pos))
        max_rounds = 20
        for _ in range(max_rounds):
            pos_arr = np.array(pts, dtype=float)
            # build graph on current positions
            G = nx.Graph()
            for i, p in enumerate(pos_arr):
                vth = self._sample_vth(rng_vth)
                G.add_node(i, pos=p, Vth=vth,
                           R_node=self.node_r_scale * max(vth, self.r_floor),
                           activated=False)
            tree = cKDTree(pos_arr)
            for i, j in tree.query_pairs(r=self.connection_radius, output_type="ndarray"):
                if self.voids and self._segment_crosses_void(pos_arr[i], pos_arr[j]):
                    continue
                dist = float(np.linalg.norm(pos_arr[i] - pos_arr[j]))
                r_edge = self.edge_k * dist
                G.add_edge(int(i), int(j), resistance=r_edge, distance=dist,
                           length=dist, R_edge=r_edge, activated=False)
            # prune
            while True:
                low = [n for n, d in G.degree() if d < 2]
                if not low:
                    break
                G.remove_nodes_from(low)
            survivors = sorted(G.nodes())
            n_survivors = len(survivors)
            if n_survivors == target:
                # Exact match — relabel and store
                self.positions = pos_arr[survivors]
                mapping = {old: new for new, old in enumerate(survivors)}
                self.G = nx.relabel_nodes(G, mapping)
                self.n_junctions = target
                print(f"Network: {self.n_junctions} nodes, {self.G.number_of_edges()} edges")
                return
            elif n_survivors > target:
                # Too many survived — keep only first `target` by node index
                keep = survivors[:target]
                self.positions = pos_arr[keep]
                mapping = {old: new for new, old in enumerate(keep)}
                sub = G.subgraph(keep).copy()
                self.G = nx.relabel_nodes(sub, mapping)
                self.n_junctions = target
                print(f"Network: {self.n_junctions} nodes, {self.G.number_of_edges()} edges")
                return
            else:
                # Too few — keep survivors and add deficit as new positions
                deficit = target - n_survivors
                kept_pos = [pts[s] for s in survivors]
                new_pts = self._sample_positions(deficit, rng_pos)
                pts = kept_pos + list(new_pts)

        # Fallback: use whatever we have
        survivors = sorted(G.nodes())
        self.positions = pos_arr[survivors]
        mapping = {old: new for new, old in enumerate(survivors)}
        self.G = nx.relabel_nodes(G, mapping)
        self.n_junctions = self.G.number_of_nodes()
        print(f"strict_N: reached max rounds, got {self.n_junctions}/{target} nodes")
        print(f"Network: {self.n_junctions} nodes, {self.G.number_of_edges()} edges")

    def _sample_positions(self, n: int, rng: np.random.Generator) -> list:
        """Sample n valid positions (outside voids)."""
        pts = []
        max_trials = n * 20_000
        trials = 0
        while len(pts) < n and trials < max_trials:
            p = (rng.uniform(0.0, self.domain[0]), rng.uniform(0.0, self.domain[1]))
            if not self.voids or not self._point_in_void(p):
                pts.append(p)
            trials += 1
        if len(pts) < n:
            raise RuntimeError(f"Could only place {len(pts)}/{n} nodes outside voids.")
        return pts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _place_voids(self, seed: int | None) -> None:
        if self.fv <= 0.0:
            self.voids = []
            return
        r = self.void_radius
        n_v = int(round(self.fv / (np.pi * r * r)))
        if n_v < 1:
            self.voids = []
            return
        rng = np.random.default_rng(seed)
        buf = 0.02
        lo_x = buf + r
        hi_x = self.domain[0] - buf - r
        lo_y = buf + r
        hi_y = self.domain[1] - buf - r
        if hi_x <= lo_x:
            lo_x = hi_x = self.domain[0] / 2
        if hi_y <= lo_y:
            lo_y = hi_y = self.domain[1] / 2
        self.voids = [
            {"center": (float(rng.uniform(lo_x, hi_x)),
                        float(rng.uniform(lo_y, hi_y))),
             "radius": r}
            for _ in range(n_v)
        ]

    def _point_in_void(self, point) -> bool:
        x, y = point
        return any(
            (x - v["center"][0]) ** 2 + (y - v["center"][1]) ** 2 < v["radius"] ** 2
            for v in self.voids
        )

    def _segment_crosses_void(self, p1, p2) -> bool:
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        d = p2 - p1
        for v in self.voids:
            c = np.asarray(v["center"], float)
            r = v["radius"]
            if np.allclose(d, 0.0):
                hit = np.linalg.norm(p1 - c) <= r
            else:
                t = np.clip(np.dot(c - p1, d) / np.dot(d, d), 0.0, 1.0)
                hit = np.linalg.norm(p1 + t * d - c) < r
            if hit:
                return True
        return False

    def _generate_positions(self, seed: int | None) -> None:
        if not self.voids:
            rng = np.random.default_rng(seed)
            self.positions = rng.random((self.N, 2)) * np.array(self.domain)
            return
        rng = np.random.default_rng(seed)
        pts: list = []
        max_trials = self.N * 20_000
        trials = 0
        while len(pts) < self.N and trials < max_trials:
            p = (rng.uniform(0.0, self.domain[0]), rng.uniform(0.0, self.domain[1]))
            if not self._point_in_void(p):
                pts.append(p)
            trials += 1
        if len(pts) < self.N:
            raise RuntimeError(
                f"Only placed {len(pts)}/{self.N} nodes outside voids. "
                "Reduce fv or increase domain size."
            )
        self.positions = np.array(pts, dtype=float)

    def _sample_vth(self, rng: np.random.Generator) -> float:
        val = rng.normal(self.mu_a, self.std_a)
        return float(np.clip(val, self.va_min, self.va_max))

    def _build_graph(self, seed: int | None) -> None:
        from scipy.spatial import cKDTree

        rng = np.random.default_rng((seed + 1000) if seed is not None else None)
        self.G = nx.Graph()
        N = len(self.positions)
        for i in range(N):
            vth = self._sample_vth(rng)
            self.G.add_node(
                i,
                pos=self.positions[i],
                Vth=vth,
                R_node=self.node_r_scale * max(vth, self.r_floor),
                activated=False,
            )
        tree = cKDTree(self.positions)
        pairs = tree.query_pairs(r=self.connection_radius, output_type="ndarray")
        for i, j in pairs:
            if self.voids and self._segment_crosses_void(self.positions[i], self.positions[j]):
                continue
            dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
            r_edge = self.edge_k * dist
            self.G.add_edge(
                int(i), int(j),
                resistance=r_edge,
                distance=dist,
                length=dist,
                R_edge=r_edge,
                activated=False,
            )

    def _prune_low_degree(self) -> None:
        total_removed = 0
        while True:
            low = [n for n, d in self.G.degree() if d < 2]
            if not low:
                break
            self.G.remove_nodes_from(low)
            total_removed += len(low)
        remaining = sorted(self.G.nodes())
        if remaining:
            self.positions = self.positions[remaining]
            mapping = {old: new for new, old in enumerate(remaining)}
            self.G = nx.relabel_nodes(self.G, mapping)
        self.n_junctions = self.G.number_of_nodes()
        if total_removed:
            print(f"Pruned {total_removed} degree-<2 nodes; {self.n_junctions} remain.")
        print(f"Network: {self.n_junctions} nodes, {self.G.number_of_edges()} edges")

    def _identify_electrodes(self) -> None:
        width = self.domain[0]
        src_x = self.source_frac * width           # nodes left of this → source
        drn_x = (1.0 - self.drain_frac) * width   # nodes right of this → drain
        sources, drains = [], []
        for node in self.G.nodes():
            x = self.positions[node][0]
            if x <= src_x:
                sources.append(node)
            elif x >= drn_x:
                drains.append(node)
        self.source_nodes = sources
        self.drain_nodes = drains
        self.source_node = sources[0] if sources else None
        self.drain_node = drains[0] if drains else None
        print(f"Electrodes: {len(sources)} sources (x ≤ {src_x:.3f}), "
              f"{len(drains)} drains (x ≥ {drn_x:.3f})")

    # ------------------------------------------------------------------
    # Legacy generate_network API (kept for backward compatibility)
    # ------------------------------------------------------------------

    def generate_network(
        self,
        seed=None,
        node_Vth=None,
        edge_k=None,
        node_r_scale=None,
        r_floor=None,
    ):
        """Backward-compatible network generator.

        If node_Vth is a dict with 'mean' and 'std' keys those override
        mu_a / std_a.  The call is otherwise equivalent to build().
        """
        if node_Vth is not None:
            if isinstance(node_Vth, dict):
                self.mu_a = float(node_Vth.get("mean", self.mu_a))
                self.std_a = float(node_Vth.get("std", self.std_a))
                self.va_min = float(node_Vth.get("min", self.va_min))
                self.va_max = float(node_Vth.get("max", self.va_max))
            # discrete list/tuple form: sample uniformly from the given values
            elif hasattr(node_Vth, "__iter__"):
                self._discrete_vth = list(node_Vth)
        if edge_k is not None:
            self.edge_k = float(edge_k)
        if node_r_scale is not None:
            self.node_r_scale = float(node_r_scale)
        if r_floor is not None:
            self.r_floor = float(r_floor)

        # Override _sample_vth if discrete values were supplied
        if hasattr(self, "_discrete_vth"):
            _disc = self._discrete_vth

            def _sample_discrete(rng):
                return float(rng.choice(_disc))

            self._sample_vth = _sample_discrete  # type: ignore[method-assign]

        if seed is not None:
            np.random.seed(seed)
        self._place_voids(seed)
        self._generate_positions(seed)
        self._build_graph(seed)
        self._prune_low_degree()

    def identify_sources_drains(self, source_frac=None, drain_frac=None):
        """Re-identify electrodes, optionally updating the electrode fractions.

        Parameters
        ----------
        source_frac : float, optional
            Fraction of domain width forming the source strip (left side).
            Nodes with x ≤ source_frac * width become sources.
        drain_frac : float, optional
            Fraction of domain width forming the drain strip (right side).
            Nodes with x ≥ (1 - drain_frac) * width become drains.

        Returns
        -------
        (source_nodes, drain_nodes) : tuple of lists
        """
        if source_frac is not None:
            self.source_frac = float(source_frac)
        if drain_frac is not None:
            self.drain_frac = float(drain_frac)
        self._identify_electrodes()
        return self.source_nodes, self.drain_nodes

    # ------------------------------------------------------------------
    # Kirchhoff solver
    # ------------------------------------------------------------------

    def _solve_kirchhoff(self, activated_nodes: set, V_applied: float):
        """Solve nodal analysis for one voltage step.

        Returns (total_current, node_potentials).
        """
        if not activated_nodes:
            return 0.0, {}

        active_sources = [n for n in self.source_nodes if n in activated_nodes]
        active_drains = [n for n in self.drain_nodes if n in activated_nodes]
        if not active_sources or not active_drains:
            return 0.0, {}

        G_sub = self.G.subgraph(activated_nodes)

        reach_s: set = set()
        for s in active_sources:
            if s in G_sub:
                reach_s |= nx.node_connected_component(G_sub, s)
        reach_d: set = set()
        for d in active_drains:
            if d in G_sub:
                reach_d |= nx.node_connected_component(G_sub, d)

        working_nodes = reach_s & reach_d
        if not working_nodes:
            return 0.0, {}

        working_list = sorted(working_nodes)
        N = len(working_list)
        local_idx = {n: k for k, n in enumerate(working_list)}
        R_MIN = 1.0

        has_node_resistance = any(
            self.G.nodes[n].get("R_node", 0.0) > 0.0 for n in working_nodes
        )

        if not has_node_resistance:
            G_mat = lil_matrix((N, N), dtype=float)
            for i, j in G_sub.edges():
                if i not in local_idx or j not in local_idx:
                    continue
                g = 1.0 / max(self.G[i][j]["R_edge"], R_MIN)
                ki, kj = local_idx[i], local_idx[j]
                G_mat[ki, ki] += g; G_mat[kj, kj] += g
                G_mat[ki, kj] -= g; G_mat[kj, ki] -= g
            b = np.zeros(N)
            for s in active_sources:
                if s in local_idx:
                    k = local_idx[s]
                    G_mat[k, :] = 0; G_mat[k, k] = 1.0; b[k] = V_applied
            for d in active_drains:
                if d in local_idx:
                    k = local_idx[d]
                    G_mat[k, :] = 0; G_mat[k, k] = 1.0; b[k] = 0.0
            try:
                phi = spsolve(G_mat.tocsr(), b)
                if not np.all(np.isfinite(phi)):
                    return 0.0, {}
            except Exception:
                return 0.0, {}
            total_current = 0.0
            for d in active_drains:
                if d not in local_idx:
                    continue
                k_d = local_idx[d]
                for nb in G_sub.neighbors(d):
                    if nb not in local_idx:
                        continue
                    g = 1.0 / max(self.G[d][nb]["R_edge"], R_MIN)
                    total_current += g * (phi[local_idx[nb]] - phi[k_d])
            return total_current, {working_list[k]: phi[k] for k in range(N)}

        # Node-resistance model (node splitting)
        all_el = set(active_sources) | set(active_drains)
        internal = [n for n in working_list if n not in all_el]
        electrode = [n for n in working_list if n in all_el]
        N_int = len(internal)
        int_idx = {n: i for i, n in enumerate(internal)}
        el_idx = {n: i for i, n in enumerate(electrode)}
        MAT = 2 * N_int + len(electrode)

        def row_in(n):
            return 2 * int_idx[n] if n in int_idx else 2 * N_int + el_idx[n]

        def row_out(n):
            return 2 * int_idx[n] + 1 if n in int_idx else 2 * N_int + el_idx[n]

        depth = {n: 999 for n in working_nodes}
        q = list(active_sources)
        for s in active_sources:
            depth[s] = 0
        h = 0
        while h < len(q):
            cur = q[h]; h += 1
            for nb in G_sub.neighbors(cur):
                if nb in depth and depth[nb] == 999:
                    depth[nb] = depth[cur] + 1
                    q.append(nb)

        G_mat = lil_matrix((MAT, MAT), dtype=float)
        for n in internal:
            g_n = 1.0 / max(self.G.nodes[n].get("R_node", 0.0), R_MIN)
            ri, ro = row_in(n), row_out(n)
            G_mat[ri, ri] += g_n; G_mat[ro, ro] += g_n
            G_mat[ri, ro] -= g_n; G_mat[ro, ri] -= g_n
        for i, j in G_sub.edges():
            if i not in local_idx or j not in local_idx:
                continue
            g_e = 1.0 / max(self.G[i][j]["R_edge"], R_MIN)
            sn, dn = (i, j) if depth[i] <= depth[j] else (j, i)
            rso, rdi = row_out(sn), row_in(dn)
            G_mat[rso, rso] += g_e; G_mat[rdi, rdi] += g_e
            G_mat[rso, rdi] -= g_e; G_mat[rdi, rso] -= g_e

        b = np.zeros(MAT)
        for s in active_sources:
            if s not in el_idx:
                continue
            for row in (row_in(s), row_out(s)):
                G_mat[row, :] = 0; G_mat[row, row] = 1.0; b[row] = V_applied
        for d in active_drains:
            if d not in el_idx:
                continue
            for row in (row_in(d), row_out(d)):
                G_mat[row, :] = 0; G_mat[row, row] = 1.0; b[row] = 0.0

        try:
            phi = spsolve(G_mat.tocsr(), b)
            if not np.all(np.isfinite(phi)):
                return 0.0, {}
        except Exception:
            return 0.0, {}

        total_current = 0.0
        for d in active_drains:
            if d not in el_idx:
                continue
            drn_row = row_in(d)
            for nb in G_sub.neighbors(d):
                if nb not in local_idx:
                    continue
                g_e = 1.0 / max(self.G[d][nb]["R_edge"], R_MIN)
                total_current += g_e * (phi[row_out(nb)] - phi[drn_row])

        node_potentials: dict = {}
        for n in internal:
            node_potentials[n] = 0.5 * (phi[row_in(n)] + phi[row_out(n)])
        for n in electrode:
            node_potentials[n] = phi[row_in(n)]
        return total_current, node_potentials

    # ------------------------------------------------------------------
    # I-V curve computation
    # ------------------------------------------------------------------

    def iv_curve(
        self,
        V_start: float = 0.0,
        V_max: float = 16.0,
        V_step: float = 0.5,
    ) -> dict:
        """Compute I-V curve via Kirchhoff nodal analysis.

        Parameters
        ----------
        V_start, V_max, V_step : float
            Voltage sweep range and step [V].

        Returns
        -------
        dict with keys:
            voltages          : np.ndarray
            currents          : np.ndarray  [A]
            conductances      : np.ndarray  [S]
            threshold_voltage : float or None — first voltage at which current flows
        """
        if not self.source_nodes or not self.drain_nodes:
            raise RuntimeError("Call build() or identify_sources_drains() first.")

        voltages, currents, conductances = [], [], []
        V = V_start
        while V <= V_max + 1e-10:
            activated = {n for n in self.G.nodes() if self.G.nodes[n]["Vth"] <= V}
            I, _ = self._solve_kirchhoff(activated, V)
            voltages.append(V)
            currents.append(I)
            conductances.append(I / V if V > 1e-12 else 0.0)
            V = round(V + V_step, 10)

        v_arr = np.array(voltages)
        i_arr = np.array(currents)
        g_arr = np.array(conductances)

        perc_idx = next((i for i, c in enumerate(currents) if c > 0), None)
        perc_V = voltages[perc_idx] if perc_idx is not None else None

        return {
            "voltages": v_arr,
            "currents": i_arr,
            "conductances": g_arr,
            "threshold_voltage": perc_V,
        }

    # backward-compat alias
    def calculate_iv_curve_kirchhoff(
        self,
        V_start: float = 0.0,
        V_max: float = 16.0,
        V_step: float = 0.5,
    ) -> dict:
        result = self.iv_curve(V_start=V_start, V_max=V_max, V_step=V_step)
        result.update({"num_paths": np.zeros(len(result["voltages"])), "path_details": []})
        return result

    # ------------------------------------------------------------------
    # Microscopic analysis: per-edge currents
    # ------------------------------------------------------------------

    def edge_currents(
        self, V_applied: float
    ) -> dict:
        """Compute signed edge currents at a given applied voltage.

        Parameters
        ----------
        V_applied : float
            Applied voltage [V].

        Returns
        -------
        dict with keys:
            total_current  : float  [A]
            node_potentials: dict {node_id: potential}
            edge_currents  : dict {(i, j): signed_current [A]}
            activated_nodes: set of node IDs
            conducting_edges: int
        """
        activated = {n for n in self.G.nodes() if self.G.nodes[n]["Vth"] <= V_applied}
        total_I, phi = self._solve_kirchhoff(activated, V_applied)
        ec = self._reconstruct_edge_currents(activated, V_applied, phi)
        return {
            "total_current": total_I,
            "node_potentials": phi,
            "edge_currents": ec,
            "activated_nodes": activated,
            "conducting_edges": len(ec),
        }

    def _reconstruct_edge_currents(
        self, activated_nodes: set, V_applied: float, phi: dict
    ) -> dict:
        R_MIN = 1.0
        if not phi:
            return {}
        working = set(phi.keys())
        G_sub = self.G.subgraph(working)
        has_node_resistance = any(
            self.G.nodes[n].get("R_node", 0.0) > 0.0 for n in working
        )
        edge_currents: dict = {}

        if not has_node_resistance:
            for i, j in G_sub.edges():
                if i in phi and j in phi:
                    R_e = max(self.G[i][j]["R_edge"], R_MIN)
                    I_ij = (phi[i] - phi[j]) / R_e
                    if abs(I_ij) > 1e-30:
                        edge_currents[(i, j)] = I_ij
            return edge_currents

        # Node-resistance mode: re-solve the split system to get terminal potentials
        active_sources = [n for n in self.source_nodes if n in working]
        active_drains = [n for n in self.drain_nodes if n in working]
        all_el = set(active_sources) | set(active_drains)
        internal = [n for n in sorted(working) if n not in all_el]
        electrode = [n for n in sorted(working) if n in all_el]
        N_int = len(internal)
        int_idx = {n: i for i, n in enumerate(internal)}
        el_idx = {n: i for i, n in enumerate(electrode)}
        MAT = 2 * N_int + len(electrode)

        def row_in(n):
            return 2 * int_idx[n] if n in int_idx else 2 * N_int + el_idx[n]

        def row_out(n):
            return 2 * int_idx[n] + 1 if n in int_idx else 2 * N_int + el_idx[n]

        depth = {n: 999 for n in working}
        q = list(active_sources)
        for s in active_sources:
            depth[s] = 0
        h = 0
        while h < len(q):
            cur = q[h]; h += 1
            for nb in G_sub.neighbors(cur):
                if nb in depth and depth[nb] == 999:
                    depth[nb] = depth[cur] + 1
                    q.append(nb)

        G_mat = lil_matrix((MAT, MAT), dtype=float)
        for n in internal:
            g_n = 1.0 / max(self.G.nodes[n].get("R_node", 0.0), R_MIN)
            ri, ro = row_in(n), row_out(n)
            G_mat[ri, ri] += g_n; G_mat[ro, ro] += g_n
            G_mat[ri, ro] -= g_n; G_mat[ro, ri] -= g_n
        edge_orient: dict = {}
        for i, j in G_sub.edges():
            g_e = 1.0 / max(self.G[i][j]["R_edge"], R_MIN)
            sn, dn = (i, j) if depth[i] <= depth[j] else (j, i)
            edge_orient[(i, j)] = (sn, dn)
            rso, rdi = row_out(sn), row_in(dn)
            G_mat[rso, rso] += g_e; G_mat[rdi, rdi] += g_e
            G_mat[rso, rdi] -= g_e; G_mat[rdi, rso] -= g_e

        b = np.zeros(MAT)
        for s in active_sources:
            for row in (row_in(s), row_out(s)):
                G_mat[row, :] = 0; G_mat[row, row] = 1.0; b[row] = V_applied
        for d in active_drains:
            for row in (row_in(d), row_out(d)):
                G_mat[row, :] = 0; G_mat[row, row] = 1.0; b[row] = 0.0

        try:
            psi = spsolve(G_mat.tocsr(), b)
            if not np.all(np.isfinite(psi)):
                return {}
        except Exception:
            return {}

        for i, j in G_sub.edges():
            sn, dn = edge_orient[(i, j)]
            R_e = max(self.G[i][j]["R_edge"], R_MIN)
            I = (psi[row_out(sn)] - psi[row_in(dn)]) / R_e
            I_ij = I if sn == i else -I
            if abs(I_ij) > 1e-30:
                edge_currents[(i, j)] = I_ij
        return edge_currents

    # ------------------------------------------------------------------
    # Network metrics
    # ------------------------------------------------------------------

    def network_summary(self) -> dict:
        """Return a scalar summary of the network topology."""
        G = self.G
        n = G.number_of_nodes()
        e = G.number_of_edges()
        degrees = [d for _, d in G.degree()]
        return {
            "n_nodes": n,
            "n_edges": e,
            "n_sources": len(self.source_nodes),
            "n_drains": len(self.drain_nodes),
            "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            "density": e / (n * (n - 1) / 2) if n > 1 else 0.0,
            "is_connected": nx.is_connected(G) if n > 0 else False,
            "mu_a_actual": float(np.mean([G.nodes[nd]["Vth"] for nd in G.nodes()])) if n else float("nan"),
            "std_a_actual": float(np.std([G.nodes[nd]["Vth"] for nd in G.nodes()])) if n else float("nan"),
        }

    def overall_resistance(self, V_applied: float = 1.0) -> float:
        """Effective source-to-drain resistance at a given applied voltage."""
        activated = {n for n in self.G.nodes() if self.G.nodes[n]["Vth"] <= V_applied}
        I, _ = self._solve_kirchhoff(activated, V_applied)
        return (V_applied / I) if I > 0 else float("inf")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "NanoparticleNetwork":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
