"""
nanonet.visualization.plots
----------------------------
Publication-quality figures for NanoparticleNetwork sweep results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

FONT_TITLE = 17
FONT_AXIS_LABEL = 16
FONT_TICK = 14
FONT_LEGEND = 14
FONT_COLORBAR = 14
LINE_WIDTH = 2.0
MARKER_SIZE = 7
FIG_DPI = 200


def _apply_axes_style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=FONT_TITLE)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(alpha=0.3)


def _trim_rows(rows, v_end):
    if v_end is None or not np.isfinite(float(v_end)):
        return rows
    return [r for r in rows if float(r.get("V", np.nan)) <= float(v_end) + 1e-9]


# ------------------------------------------------------------------
# I-V curve
# ------------------------------------------------------------------

def plot_iv_curve(
    result: dict,
    outpath: str,
    title: str = "Kirchhoff I-V curve",
    show_chargeconserving: bool = True,
) -> str:
    """Single-panel I-V curve from a sweep() result."""
    rows = result["rows"]
    V = np.array([r["V"] for r in rows])
    I = np.array([r["total_current_A"] for r in rows])
    Icc = np.array([r["total_current_chargeconserving_A"] for r in rows])
    pV = result.get("percolation_V")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(V, I * 1e9, "-o", ms=4, lw=LINE_WIDTH, color="tab:blue",
            label="Kirchhoff (solver)")
    if show_chargeconserving:
        ax.plot(V, Icc * 1e9, "--", lw=LINE_WIDTH, color="tab:orange",
                label="charge-conserving")
    if pV is not None:
        ax.axvline(pV, color="k", ls=":", lw=1, alpha=0.6)
    _apply_axes_style(ax, title, "Applied voltage [V]", "Current [nA]")
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_iv_overlay(
    results: list[dict],
    outpath: str,
    mode: str = "sigma",
    title: str = "I-V overlay",
) -> str:
    """Overlay I-V curves coloured by the swept parameter."""
    def sort_key(r):
        return (r.get("sigma_va_target", 0) if mode == "sigma" else
                r.get("mean_va_target", 0) if mode == "mean" else
                r.get("N", 0))
    results = sorted(results, key=sort_key)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c, r in zip(colors, results):
        v = np.asarray(r["voltages"], float)
        i = np.asarray(r["currents"], float)
        v_end = float(r.get("transition_voltage_V", np.nan))
        if np.isfinite(v_end):
            keep = v <= v_end + 1e-9
            v, i = v[keep], i[keep]
        if mode == "sigma":
            lbl = rf"$\sigma$ = {r['sigma_va_target']:.0f} V"
        elif mode == "mean":
            lbl = rf"$\langle V_a\rangle$ = {r['mean_va_target']:.0f} V"
        else:
            lbl = f"N = {r['N']}"
        ax.plot(v, i * 1e9, lw=LINE_WIDTH, color=c, label=lbl)
    _apply_axes_style(ax, title, "Applied voltage [V]", "Current [nA]")
    ax.legend(frameon=False, fontsize=FONT_LEGEND)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ------------------------------------------------------------------
# Network-evolution panels
# ------------------------------------------------------------------

def plot_evolution(
    rows: list[dict],
    fields: list[str],
    title: str,
    outpath: str,
    percolation_V=None,
    v_end=None,
) -> str:
    """Four-panel per-voltage network-evolution figure."""
    rows = _trim_rows(rows, v_end)

    def col(name):
        return np.array([r.get(name, np.nan) for r in rows], dtype=float)

    V = col("V")
    an = col("activated_nodes")
    ae = col("activated_edges")
    ce = col("conducting_edges")
    I = col("total_current_A")
    Icc = col("total_current_chargeconserving_A")
    part = col("participation_ratio")
    bb = col("backbone_edges")

    if percolation_V is None:
        sdc = col("source_drain_connected")
        idx = np.where(sdc >= 1)[0]
        percolation_V = float(V[idx[0]]) if len(idx) else None

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=FONT_TITLE + 1, fontweight="bold")

    a = ax[0, 0]
    a.plot(V, an, "-o", ms=4, color="tab:blue", label="activated nodes")
    _apply_axes_style(a, "Activation growth", "Applied voltage [V]", "Node count [-]")
    a.legend(fontsize=FONT_LEGEND)

    a = ax[0, 1]
    a.plot(V, ae, "-", color="tab:gray", label="activated edges")
    a.plot(V, ce, "-", color="tab:green", label="conducting edges")
    _apply_axes_style(a, "Activated vs conducting edges", "Applied voltage [V]", "Edge count [-]")
    a.legend(fontsize=FONT_LEGEND)

    a = ax[1, 0]
    a.plot(V, I * 1e9, "-o", ms=4, color="tab:red", label="solver current")
    a.plot(V, Icc * 1e9, "--", color="tab:orange", label="charge-conserving")
    _apply_axes_style(a, "I-V curve (Kirchhoff)", "Applied voltage [V]", "Current [nA]")
    a.legend(fontsize=FONT_LEGEND)

    a = ax[1, 1]
    a.plot(V, part, "-", color="tab:purple", label="participation ratio")
    a.plot(V, bb, "-", color="tab:orange", label="backbone edges")
    _apply_axes_style(a, "Current concentration", "Applied voltage [V]", "Effective edge count [-]")
    a.legend(fontsize=FONT_LEGEND)

    if percolation_V is not None:
        for a in ax.ravel():
            a.axvline(percolation_V, color="k", ls="--", lw=1, alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_snapshots(
    net,
    snap_voltages: list[float],
    outpath: str,
    title: str | None = None,
) -> str:
    """Multi-panel snapshots showing conduction spreading with voltage."""
    from nanonet.analysis.sweep import _edge_currents_at

    pos = net.positions
    snap_voltages = sorted(set(float(v) for v in snap_voltages))
    ec_by_V = {}
    for V in snap_voltages:
        activated = {n for n in net.G.nodes() if net.G.nodes[n]["Vth"] <= V}
        _, _, ec = _edge_currents_at(net, activated, V)
        ec_by_V[V] = ec

    all_max = max(
        (max((abs(c) for c in ec.values()), default=0.0) for ec in ec_by_V.values()),
        default=1.0,
    ) or 1.0

    n_panels = len(snap_voltages)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    src_set = set(net.source_nodes)
    drn_set = set(net.drain_nodes)

    for ax, V in zip(axes, snap_voltages):
        ec = ec_by_V[V]
        segs_all = [[(pos[i][0], pos[i][1]), (pos[j][0], pos[j][1])]
                    for i, j in net.G.edges()]
        ax.add_collection(LineCollection(segs_all, colors="lightgray",
                                         linewidths=0.2, alpha=0.25, zorder=1))
        if ec:
            segs, widths, vals = [], [], []
            for (i, j), c in ec.items():
                segs.append([(pos[i][0], pos[i][1]), (pos[j][0], pos[j][1])])
                mag = abs(c)
                widths.append(0.3 + 4.0 * mag / all_max)
                vals.append(mag / all_max)
            lc = LineCollection(segs, array=np.array(vals), cmap="plasma",
                                linewidths=widths, zorder=3)
            lc.set_clim(0, 1)
            ax.add_collection(lc)
        activated = {n for n in net.G.nodes() if net.G.nodes[n]["Vth"] <= V}
        off = [n for n in net.G.nodes() if n not in activated]
        on = list(activated)
        if off:
            ax.scatter(pos[off, 0], pos[off, 1], s=4, c="lightgray", alpha=0.6, zorder=2)
        if on:
            ax.scatter(pos[on, 0], pos[on, 1], s=22, c="steelblue",
                       edgecolors="navy", linewidths=0.3, alpha=0.85, zorder=4)
        if src_set:
            s = np.array(sorted(src_set))
            ax.scatter(pos[s, 0], pos[s, 1], s=28, marker="s", c="green",
                       edgecolors="k", lw=0.4, zorder=5)
        if drn_set:
            d = np.array(sorted(drn_set))
            ax.scatter(pos[d, 0], pos[d, 1], s=28, marker="s", c="red",
                       edgecolors="k", lw=0.4, zorder=5)
        ax.set_title(f"$V$ = {V:g} V", fontsize=FONT_TITLE - 1)
        ax.set_xlim(-0.02, net.domain[0] + 0.02)
        ax.set_ylim(-0.02, net.domain[1] + 0.02)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor="green",
                   markeredgecolor="k", markersize=9, label="source"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="red",
                   markeredgecolor="k", markersize=9, label="drain"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="steelblue",
                   markeredgecolor="navy", markersize=9, label="activated"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="lightgray",
                   markeredgecolor="none", markersize=8, label="inactive"),
        ],
        loc="upper left", fontsize=FONT_LEGEND,
    )
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("|edge current| / max [-]", fontsize=FONT_COLORBAR)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    if title is None:
        title = "Conduction region spreading with voltage"
    fig.suptitle(title, fontsize=FONT_TITLE + 1, fontweight="bold")
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ------------------------------------------------------------------
# Structure / distribution / spectral panels
# ------------------------------------------------------------------

def plot_spectral_evolution(
    rows: list[dict],
    fields: list[str],
    title: str,
    outpath: str,
    percolation_V=None,
    v_end=None,
) -> str:
    """Effective resistance and algebraic connectivity vs voltage."""
    rows = _trim_rows(rows, v_end)

    def col(name):
        return np.array([r.get(name, np.nan) for r in rows], dtype=float)

    V = col("V")
    Reff = col("effective_resistance_ohm")
    ac = col("algebraic_connectivity")
    if percolation_V is None:
        sdc = col("source_drain_connected")
        idx = np.where(sdc >= 1)[0]
        percolation_V = float(V[idx[0]]) if len(idx) else None

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=FONT_TITLE + 1, fontweight="bold")

    a = ax[0]
    mask = np.isfinite(Reff) & (Reff > 0)
    a.semilogy(V[mask], Reff[mask], "-o", ms=4, color="tab:red")
    _apply_axes_style(a, "Effective source-drain resistance",
                      "Applied voltage [V]", r"$R_\mathrm{eff}$ [$\Omega$]")

    a = ax[1]
    mask2 = np.isfinite(ac)
    a.plot(V[mask2], ac[mask2], "-o", ms=4, color="tab:blue")
    _apply_axes_style(a, "Algebraic connectivity (Fiedler value)",
                      "Applied voltage [V]", "Algebraic connectivity [-]")

    if percolation_V is not None:
        for a in ax:
            a.axvline(percolation_V, color="k", ls="--", lw=1, alpha=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return outpath
