<div align="center">

# 🩸 Microneedle Array Optimizer

**Find the needle count and geometry that maximize flow through a circular microneedle array, while respecting insertion physics.**

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-SLSQP%20%7C%20trust--constr-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualizations-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Overview

Microneedle arrays are used for minimally invasive sampling and drug delivery. Their design is a genuine trade-off: more and wider needles raise ideal flow, but they also split the insertion force (the **bed-of-nails effect**), crowd the local tissue supply, and can stretch skin instead of piercing it. Too few needles concentrate load on each one.

**Microneedle Array Optimizer** turns that trade-off into a reproducible **simulation pipeline**. It couples a Hagen-Poiseuille flow model with four biomechanical penalty factors, then runs **constrained numerical optimization** to find the best **needle radius**, **needle length**, and **array radius** for every needle count from 2 to 15. An interactive Streamlit app lets you explore the design space, compare solvers, and test how sensitive the answer is to insertion force.

> ⚠️ **Note:** This is a **parametric engineering model**, not a validated clinical simulator. The defaults (for example, the 0.7 N insertion force) are illustrative and were tuned so the flow-vs-needles curve peaks near n = 6. Replace them with measured values before drawing quantitative conclusions.

---

## 🧮 Model & Optimization Mechanics

### 1. Flow model

Total flow is ideal Hagen-Poiseuille flow summed over `n` needles, scaled by four multipliers (each between 0 and 1):

```
Q = n · (π R⁴ ΔP) / (8 μ L) · f_vascular · f_puncture · f_buckle · f_tent
```

| Factor | Physical meaning | Behavior |
|---|---|---|
| `vascular` | Less tissue per needle lowers the effective driving pressure | Drops as needles crowd a small array |
| `puncture` | Insertion force is shared across needles, so tip stress falls below the skin puncture stress | Drops as **n grows** (too many needles) |
| `buckle` | Euler critical load vs. the load carried by each needle | Matters for thin, long needles under high force (too few needles) |
| `tent` | Large arrays stretch skin instead of piercing it | Drops as array radius grows |

### 2. Optimization loop

1. **Enumerate n.** The needle count is an integer from 2 to 15, so every value is solved exactly instead of using a mixed-integer solver.
2. **Optimize continuous variables** `[R, L, array radius]` for each `n`, using **bounded, normalized** variables (scaled to [0, 1]) to keep solvers well-conditioned.
3. **Enforce spacing** between neighboring needles on the circle:
   `2 · a · sin(π/n) ≥ 2R + min_clearance`
4. **Multi-start** each solve from three initial points to reduce local-optimum risk.
5. **Validate feasibility** after each solve, discard non-finite or spacing-violating results, and keep the best design.
6. **Pick the optimum** as the `n` with the highest flow, then compare solvers.

### 3. Solvers

| Solver | Type | Role |
|---|---|---|
| `SLSQP` | Sequential least-squares quadratic programming | Fast; default workhorse |
| `trust-constr` | Trust-region constrained method | Independent cross-check |

The app reports whether both solvers **agree** on the optimum (same `n`, flow within 0.1%).

> 💡 **Tip:** Flow scales as R⁴/L, so the optimizer pushes radius up and length down until a bound or spacing limit stops it. The app flags this as an **active design limit**. To get an interior optimum, widen the bounds or add physics that penalizes large R or short L.

### 4. Reference result (default parameters)

| Needles (n) | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 10 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| Max flow (nL/s) | 33.2 | 45.5 | 54.9 | 60.9 | **62.8** | 60.6 | 55.0 | 41.0 | 22.4 |

Optimum at defaults: **n = 6**, needle radius **50 µm**, needle length **0.5 mm**, array radius **≈ 1.17 mm**, **≈ 62.8 nL/s** (both solvers agree).

---

## ✨ Key Capabilities

- **Interactive parameter and constraint tuning:** sliders for insertion force, tip puncture stress, radius/length/array-radius bounds, minimum needle gap, and advanced physics (viscosity, pressure drop, modulus, and more), with one-click reset.
- **Multi-solver comparison:** run `SLSQP`, `trust-constr`, or both, with automatic agreement checks.
- **Feasibility and boundary diagnostics:** clear errors for infeasible bounds, per-`n` captions for infeasible counts, and notices when variables sit on their limits.
- **Plotly visualizations:**
  - Flow vs. needle count, with the optimum starred
  - "Why it peaks" chart of each limiting factor vs. `n`
  - **Sensitivity analysis:** optimal `n` vs. insertion force
- **Design-space table** for every `n` and solver, with **CSV export**.
- **Model notes tab** documenting the equations and assumptions.
- **Cached solves** (`st.cache_data`) so unrelated widget changes don't re-run the optimizer.

---

## 🗂️ Repository Structure

| Path | Role |
|---|---|
| `app.py` | Core model: `Params` dataclass, flow factors, constrained optimizers, `n` sweep. Also runs as a standalone CLI report. |
| `frontend.py` | Streamlit interface: controls, solver comparison, charts, sensitivity sweep, CSV download. Imports from `app.py`. |
| `requirements.txt` | Pinned dependencies for a reproducible environment. |
| `assets/` *(optional)* | Add screenshots of the app here and reference them in this README. |

> 💡 **Tip:** No dataset or trained weights are needed. Every result is computed analytically and numerically on demand, which keeps the workflow lightweight and **deployment-ready**.

<!-- Add a screenshot once you have one:
![App screenshot](assets/app-screenshot.png)
-->

---

## 🚀 Setup & Execution

**1. Clone and create an environment**

```bash
git clone <your-repo-url>
cd microneedle-array-optimizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3a. Launch the interactive app**

```bash
streamlit run frontend.py
```

Opens at `http://localhost:8501`.

**3b. Or run the standalone sweep** (prints per-`n` flow and the optimal design for each solver)

```bash
python app.py
```

> ⚠️ **Note:** After editing `app.py` (for example, the `n_range`), restart Streamlit or clear its cache (press `C` in the browser tab). Cached results are not invalidated by changes to imported files.

**Pinned versions** (developed and tested on Python 3.13):

| Package | Version |
|---|---|
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| pandas | 3.1.1 |
| plotly | 7.1.0 |
| streamlit | 1.67.0 |

---

## 🎯 Use Cases & Applications

- **Biomedical device prototyping:** shortlist needle counts and geometries before fabrication.
- **Microfluidic and drug-delivery R&D:** study how force, tissue supply, and spacing limit throughput.
- **Engineering design-space exploration:** run **sensitivity analysis** and parameter sweeps to see how robust an optimum is to uncertain inputs.
- **Interactive technical demos:** teach constrained optimization, solver comparison, and physics-informed penalty modeling with a live **evaluation loop**.

---

## ⚠️ Assumptions & Edge Cases

- **Infeasible bounds:** if the array is too small for the needle radius plus clearance, the app reports "No feasible design" and suggests widening limits.
- **Solver disagreement:** flagged in the UI. Inspect the design table and the factor chart before trusting either result.
- **Edge of the needle range:** at high insertion force the optimum can sit at n = 15, so the true peak may lie beyond the tested range.
- **Calibration:** the force and puncture-stress defaults are placeholders, not measured skin data.

---

## 📄 License & Acknowledgements

Released under the **MIT License** (placeholder: add a `LICENSE` file and update the author name and year).

Built with [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [pandas](https://pandas.pydata.org/), [Plotly](https://plotly.com/python/), and [Streamlit](https://streamlit.io/). The flow model is based on the classical **Hagen-Poiseuille** equation for laminar flow in a cylindrical channel.
