# nanonet

[![CI](https://github.com/oissakah/nanonet/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/oissakah/nanonet/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

\`nanonet\` is a Python package and Dash application for graph-based simulation of electron transport in self-assembled nanoparticle necklace networks.

This release synchronizes the reusable package with the corrected V14 model developed in [graph-based-nanoparticle-necklace-network](https://github.com/oissakah/graph-based-nanoparticle-necklace-network).

## Corrected transport model

The network is represented as a spatial random graph. Each junction \`i\` is assigned a microscopic activation voltage \`V_a,i\`, while each graph edge represents a nanoparticle-chain connection.

### Voltage-dependent activation

Each junction \`i\` is electrically active when

\`\`\`math
V_{a,i} \le V .
\`\`\`

where \`V\` is the applied device voltage.

An edge is electrically available only when both endpoint junctions are active. This is a phenomenological global-voltage gating rule; activation is not solved self-consistently from the local voltage drop.

### Resistance model

The geometric resistance of an edge is

\`\`\`math
R_{\mathrm{edge},ij}=k_{\mathrm{edge}}\,d_{ij}.
\`\`\`

The junction resistance is fixed and independent of \`V_a,i\`:

\`\`\`math
R_{\mathrm{node}}=R_{\mathrm{junction}}.
\`\`\`

In the implementation, \`R_junction\` is specified by \`node_resistance_ohm\`.

For each active undirected connection, the solver uses the symmetric total resistance

\`\`\`math
R_{\mathrm{tot},ij}
=
R_{\mathrm{edge},ij}
+
\frac{R_i+R_j}{2},
\qquad
g_{ij}
=
\frac{1}{R_{\mathrm{tot},ij}}.
\`\`\`

Electrode-contact nodes contribute zero junction resistance. This formulation keeps activation timing separate from electrical resistance and avoids orientation-dependent split-node artifacts.

### Kirchhoff solution

Only active connected components that touch both electrode sets are included in the electrical solve. The conductances form a symmetric weighted graph Laplacian \`G\`:

\`\`\`math
G_{ii}=\sum_j g_{ij},
\qquad
G_{ij}=-g_{ij}\quad(i\ne j).
\`\`\`

Source-contact nodes are fixed at \`V\`, drain-contact nodes at \`0\`, and the internal node potentials are obtained from the sparse nodal system.

Every edge current is reconstructed from that same solution:

\`\`\`math
I_{ij}=g_{ij}\left(\phi_i-\phi_j\right).
\`\`\`

Source and drain boundary currents are calculated independently. The reported device current is their symmetric average:

\`\`\`math
I(V)
=
\frac{
\left|I_{\mathrm{source}}\right|
+
\left|I_{\mathrm{drain}}\right|
}{2}.
\`\`\`

The absolute source-drain difference is retained as a current-conservation diagnostic.

### Production defaults

| Parameter | Default |
|---|---:|
| Junction count \`N\` | 500 |
| Connection radius \`r_c\` | 0.15 |
| Domain | 1 × 1 |
| Edge resistance constant | 2.0 × 10¹⁰ |
| Fixed junction resistance | 3.5 × 10⁹ Ω |
| Source / drain strips | 0.15 / 0.15 |
| Voltage sweep | 0–16 V in 0.5 V steps |
| Activation bounds | 0–20 V |
| Seeds | 41, 51, 61, 71, 81 |

See \`optimized_config.yaml\`.

## Threshold and nonlinear fit convention

The transport threshold is not a free fit parameter. It is defined as the first sampled source-drain percolation voltage:

\`\`\`math
V_T \equiv V_{\mathrm{perc}}.
\`\`\`

\`V_perc\` is the first sampled voltage at which the active network spans source to drain and current becomes nonzero. It is not refitted as a free parameter.

Above this threshold, the nonlinear region is described by

\`\`\`math
I=A\left(V-V_T\right)^{\zeta}.
\`\`\`

with \`V_T\` held fixed. Only \`A\` and \`ζ\` are fitted.

The threshold point itself is excluded because

\`\`\`math
\log\!\left(V-V_T\right)
\`\`\`

is undefined at \`V = V_T\`. The fit ends when 90% of the nodes are active; if 90% activation is not reached, the available fit window is used up to the configured maximum width.

The compatibility field \`fit_V_T_V\` is therefore identical to \`percolation_voltage_V\`.

## Current participation ratio

The package reports the current participation ratio

\`\`\`math
N_{\mathrm{eff}}
=
\frac{
\left(\sum_e |I_e|\right)^2
}{
\sum_e I_e^2
}.
\`\`\`

\`N_eff\` estimates the effective number of conducting edges sharing the current.

- Small \`N_eff\`: strongly localized transport through relatively few edges.
- Large \`N_eff\`: current is distributed across a broader conducting backbone.

The sweep table exposes the same quantity as \`participation_ratio\`, \`current_participation_ratio\`, and \`N_eff\` for compatibility.

## Independent pathways

At \`V_perc\`, \`nanonet\` reports the maximum number of **edge-disjoint** source-to-drain transport channels.

This is calculated using a unit-capacity max-flow/min-cut calculation rather than enumerating all simple paths.

## Spectral diagnostics

Spectral quantities come from the exact same active-circuit Laplacian used for transport.

The raw algebraic-connectivity quantity is

\`\`\`math
\lambda_2 \; [\mathrm{S}]
\`\`\`

because the Laplacian is conductance weighted.

The dimensionless spectral-gap ratio is

\`\`\`math
\frac{\lambda_2}{\lambda_{\max}}.
\`\`\`

## Density studies

When \`N\` is varied, the domain and connection radius remain fixed:

\`\`\`math
r_c=0.15.
\`\`\`

for

\`\`\`math
N=200,\;400,\;600,\;800.
\`\`\`

Increasing \`N\` therefore increases network density, the number of nearby neighbors, and the number of edges. The older \`N^(-1/2)\` radius scaling is not used.

Two density runners are available:

- \`run_sweep_vary_N_mean\`: crossed \`N × ⟨V_a⟩\` study at fixed \`σ_a\`
- \`run_sweep_vary_N\`: \`N\`-only study at fixed \`⟨V_a⟩\` and \`σ_a\`

## Void fraction

For random-void simulations, the package stores both:

- \`void_fraction_requested\`
- \`void_fraction_achieved\`

The achieved value is estimated from the union of the void areas, so overlapping voids are counted only once.

## Installation

\`\`\`bash
pip install -e .
\`\`\`

For development and tests:

\`\`\`bash
pip install -e ".[dev]"
pytest -q
\`\`\`

Python 3.9 or newer is required.

## Quick start

\`\`\`python
from nanonet import NanoparticleNetwork, sweep

net = NanoparticleNetwork(
    L=1.0,
    N=500,
    fv=0.0,
    mu_a=7.0,
    std_a=2.0,
    connection_radius=0.15,
    edge_k=2.0e10,
    node_resistance_ohm=3.5e9,
).build(seed=41)

iv = net.iv_curve(V_start=0.0, V_max=16.0, V_step=0.5)
print("V_perc =", iv["percolation_voltage"])

result = sweep(
    net,
    V_start=0.0,
    V_max=16.0,
    V_step=0.5,
)

for row in result["rows"]:
    if row["source_drain_connected"]:
        print("V =", row["V"])
        print("I =", row["total_current_A"])
        print("N_eff =", row["N_eff"])
        print(
            "edge-disjoint pathways =",
            row["edge_disjoint_pathways_at_Vperc"],
        )
        break
\`\`\`

## Parameter sweeps

\`\`\`python
from nanonet import (
    SweepConfig,
    run_sweep_vary_std,
    run_sweep_vary_mean,
    run_sweep_vary_N_mean,
    run_sweep_vary_N,
    run_sweep_vary_voids,
)

cfg = SweepConfig(
    node_resistance_ohm=3.5e9,
    connection_radius=0.15,
    seeds=[41, 51, 61, 71, 81],
)

density_map = run_sweep_vary_N_mean(cfg)
\`\`\`

## Interactive web app

Run:

\`\`\`bash
python app.py
\`\`\`

and open \`http://127.0.0.1:8050\`.

The resistance control is **Junction resistance [Ω]** rather than an activation-dependent resistance scale.

The web app uses the same fixed-\`V_T\` convention and the same conductance-weighted circuit Laplacian as the package solver. Its spectral panel reports \`λ₂ [S]\`.

## Package structure

\`\`\`text
nanonet/
├── core/
│   └── network.py
├── analysis/
│   ├── sweep.py
│   └── spectral.py
├── sweeps/
│   └── parameter_sweep.py
├── visualization/
│   └── plots.py
└── io.py

docs/
└── MODEL_CORRECTIONS.md

tests/
├── test_model_consistency.py
├── test_threshold_convention.py
└── test_N_density_definition.py
\`\`\`

## Model corrections

The complete consistency checklist is in [docs/MODEL_CORRECTIONS.md](docs/MODEL_CORRECTIONS.md).

## Research reference

The synchronized research implementation is associated with:

> Obed Issakah, Srivathsan Badrinarayanan, Ravi F. Saraf, and Janghoon Ock, “Graph-Based Kirchhoff Modeling of Non-Ohmic Electron Transport in Self-Assembled Nanonecklace Networks,” arXiv:2607.03698 (2026), DOI: 10.48550/arXiv.2607.03698.

\`\`\`bibtex
@article{issakah2026nanonecklace,
  title   = {Graph-Based Kirchhoff Modeling of Non-Ohmic Electron Transport in Self-Assembled Nanonecklace Networks},
  author  = {Issakah, Obed and Badrinarayanan, Srivathsan and Saraf, Ravi F. and Ock, Janghoon},
  journal = {arXiv preprint arXiv:2607.03698},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.03698}
}
\`\`\`

## License

MIT. See \`LICENSE\`.
