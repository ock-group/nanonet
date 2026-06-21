# nanonet

A Python package for simulating nanoparticle necklace films as spatial graphs, with an interactive web platform for real-time network exploration.

Nanoparticle necklace films are modelled as random geometric graphs where **nodes** are voltage-activated junctions and **edges** are nanoparticle chain segments. Current transport is solved via Kirchhoff nodal analysis, and the nonlinear I–V characteristics are fitted to a power-law model across three distinct conduction phases.

---

## Model overview

The film is represented as a 2D graph embedded in a normalised domain. Each node carries a threshold voltage $V_a$ drawn from a Gaussian distribution $\mathcal{N}(\mu_a, \sigma_a^2)$. A node activates (conducts) only when the applied voltage exceeds its threshold. Edges form between nodes within a connection radius and carry a distance-proportional resistance.

**Key parameters**

| Parameter | Symbol | Description |
|---|---|---|
| Domain size | $L$ | Side length of the square domain (normalised) |
| Junction count | $N$ | Number of junction nodes |
| Void fraction | $f_v$ | Area fraction occupied by insulating voids |
| Mean activation voltage | $\mu_a$ | Mean of the $V_a$ distribution [V] |
| Std activation voltage | $\sigma_a$ | Standard deviation of the $V_a$ distribution [V] |
| Edge resistance constant | `edge_k` | $R_\text{edge} = \text{edge\_k} \times d$ [Ω/m] |
| Node resistance scale | `node_r_scale` | $R_\text{node} = \text{node\_r\_scale} \times \max(V_a, r_\text{floor})$ [Ω/V] |

**I–V phases**

- **Phase I** — zero current; no percolating path exists
- **Phase II** — nonlinear conduction; fitted to $I = A(V - V_T)^\zeta$
- **Phase III** — quasi-linear; begins when 90 % of nodes have activated

The Phase II fit window is bounded below by the percolation onset voltage and above by the 90 %-activation voltage.

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9. Dependencies are listed in `requirements.txt` and `pyproject.toml`.

---

## Quick start

```python
from nanonet import NanoparticleNetwork, sweep, plot_iv_curve

# Build a network
net = NanoparticleNetwork(L=1.0, N=300, fv=0.0, mu_a=6.0, std_a=3.0).build(seed=42)

# I–V curve
iv = net.iv_curve(V_start=0.0, V_max=16.0, V_step=0.5)
print("Threshold voltage:", iv["threshold_voltage"], "V")

# Microscopic analysis at one voltage
mic = net.edge_currents(V_applied=8.0)
print("Total current:", mic["total_current"], "A")
print("Conducting edges:", mic["conducting_edges"])

# Full sweep with connectivity metrics
result = sweep(net, V_start=0.0, V_max=16.0, V_step=0.5)
```

See `tutorial.ipynb` for a complete walkthrough including parameter sweeps and visualisation.

---

## Web platform

An interactive browser-based platform is included for real-time network exploration.

```bash
python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

**Controls (left sidebar)**

| Section | Parameters |
|---|---|
| Network Geometry | $L$, $N$, $f_v$, connection radius, source/drain fraction |
| Activation Voltage | $\mu_a$, $\sigma_a$ |
| Resistance Model | `edge_k`, `node_r_scale` |
| Voltage Sweep | $V_\text{start}$, $V_\text{max}$, $V_\text{step}$ |
| Reproducibility | Random seed |

**Buttons**

- **Preview Network** — instantly renders the network topology (no sweep)
- **Build & Run I-V Sweep** — builds the network and runs the full voltage sweep

**Output panels**

- **Network graph** — nodes coloured by activation state; edges coloured by current magnitude at the probe voltage
- **I–V curve** — raw data with Phase I/II/III shading and power-law fit overlay
- **Algebraic connectivity λ₂ vs voltage** — Fiedler value growth across the sweep
- **Statistics panel** — phase boundary voltages, fit parameters ($V_T$, $\zeta$, $R^2$), and network summary

The **probe voltage slider** scrubs through the already-computed sweep to inspect any voltage snapshot without rerunning the simulation.

---

## Package structure

```
nanonet/
├── core/
│   └── network.py          — NanoparticleNetwork class
├── analysis/
│   ├── sweep.py            — per-voltage activation and current sweep
│   └── spectral.py         — algebraic connectivity, effective resistance
├── sweeps/
│   └── parameter_sweep.py  — SweepConfig and parameter sweep runners
├── visualization/
│   └── plots.py            — matplotlib plot functions
└── io.py                   — CSV and pickle I/O helpers
```

---

## Parameter sweeps

```python
from nanonet import SweepConfig, run_sweep_vary_std, run_all_cases

cfg = SweepConfig(
    N=500, mu_a=6.0,
    sigma_values=[1.0, 3.0, 5.0, 7.0],
    fixed_mean=6.0,
    seeds=[41, 51, 61, 71, 81],
)
results = run_sweep_vary_std(cfg)
```

Available sweep runners:

| Function | Varies |
|---|---|
| `run_sweep_vary_std` | $\sigma_a$ at fixed $\mu_a$ and $N$ |
| `run_sweep_vary_mean` | $\mu_a$ at each $\sigma_a$ |
| `run_sweep_vary_N` | $N$ at fixed $\mu_a$ and $\sigma_a$ |
| `run_sweep_vary_voids` | void fraction $f_v$ at fixed $N$, $\mu_a$, $\sigma_a$ |
| `run_all_cases` | all of the above |

---

## License

See `LICENSE`.
