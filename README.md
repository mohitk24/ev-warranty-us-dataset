<div align="center">

# 🩸 Microneedle Array Optimizer

**Find the needle count, needle radius, needle length, and array radius that maximize blood flow through a circular microneedle array, then explore why.**

Built on a Hagen-Poiseuille flow model with skin-penetration penalties, solved with SciPy's **SLSQP** and **trust-constr**, and wrapped in an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-SLSQP%20%7C%20trust--constr-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualizations-3F4F75?logo=plotly&logoColor=white)

[Overview](#-overview) · [How It Works](#-how-it-works) · [Key Features](#-key-features) · [Project Structure](#-project-structure) · [Use Cases](#-use-cases)

</div>

---

## 📖 Overview

A microneedle array draws blood through tiny hollow needles, and every design choice pulls against another. Wider, shorter needles flow faster (flow scales as **R⁴/L**), but packing more needles into a small footprint starves each one of nearby tissue, and spreading the same insertion force across too many needles leaves each tip too blunt to puncture skin (the *bed-of-nails effect*). Too few needles, and total flow is limited by count alone.

This project turns that trade-off into a **constrained optimization** problem. For every needle count from **2 to 15**, it finds the best needle radius, needle length, and array radius, then compares the results to reveal the optimal count. An **interactive dashboard** lets you change the physics and design limits live and see how the answer moves, which is much faster than editing code and rerunning scripts.

---

## 🧠 How It Works

### 1. Mathematical model

Total flow is Hagen-Poiseuille flow through each needle, scaled by four multipliers between 0 and 1:

$$
Q = n \cdot \frac{\pi R^4 \,\Delta P}{8 \mu L} \cdot f_{\text{vascular}} \cdot f_{\text{puncture}} \cdot f_{\text{buckle}} \cdot f_{\text{tent}}
$$

| Multiplier | Physical effect | What it penalizes |
|---|---|---|
| `f_vascular` | Less tissue per needle lowers the effective driving pressure | Crowded arrays |
| `f_puncture` | Insertion force is shared across needles, so tip stress drops below the skin-puncture stress | **Too many** needles |
| `f_buckle` | Per-needle load vs. critical buckling load (Euler, silicon) | Thin, long needles |
| `f_tent` | Large arrays stretch skin instead of piercing it | Large array radius |

The puncture and buckling terms use smooth logistic "gates" (steepness is adjustable), which keeps the objective differentiable for gradient-based solvers.

### 2. Constrained optimization

The needle count is an integer, so the problem is mixed-integer. With only 14 candidates, the code uses **exact enumeration**: for each `n`, it solves a small continuous problem and keeps the best.

- **Variables:** needle radius `R`, needle length `L`, array radius.
- **Objective:** maximize total flow rate.
- **Bounds:** user-adjustable limits on all three variables.
- **Spacing constraint:** neighboring needles on the circle must keep a minimum gap, so the chord distance `2·a·sin(π/n)` must be at least `2R + clearance`.
- **Conditioning:** variables are normalized to [0, 1] so both solvers behave well despite micrometer-to-millimeter scale differences.

### 3. Multi-start sweep and solver comparison

| Solver | Type | Role |
|---|---|---|
| `SLSQP` | Sequential least-squares quadratic programming | Fast; default workhorse |
| `trust-constr` | Trust-region constrained method | Independent cross-check |

The app reports whether both solvers **agree** on the optimum (same `n`, flow within 0.1%).

Each solve runs from **three starting points** to guard against local optima, and only feasible results are kept. The whole sweep runs with both **SLSQP** and **trust-constr**, and the dashboard reports whether the two solvers agree on the optimum.

With default settings, both solvers find **6 needles** (≈ 62.8 nL/s), with flow rising from 2 to 6 and falling after 7.

> 💡 **Tip:** Flow scales as R⁴/L, so the optimizer pushes radius to its upper limit and length to its lower limit. The dashboard flags when a variable sits on a bound, which signals that you should widen the limits or add a physical penalty for extreme geometries.

---

## ⚡ Key Features

- **Dual-solver numerical optimization:** SLSQP and trust-constr side by side, with an automatic agreement check.
- **Interactive dashboard:** sliders for insertion force, puncture stress, design limits, minimum gap, and advanced physics, with one-click reset.
- **Explainable results:** a "Why it peaks" view plots each penalty multiplier against needle count, so you can see which physics bends the curve.
- **Sensitivity analysis:** a parameter sweep over insertion force shows how stable the optimal needle count is.
- **Data visualization and export:** Plotly charts with hover and zoom, a sortable results table, and CSV download.
- **Edge-case handling:** infeasible settings produce a clear message instead of a crash, needle counts with no feasible design are reported, and solver warnings are suppressed.
- **Performance and caching:** results are cached with `st.cache_data`, so moving unrelated widgets doesn't re-run the solvers. A full SLSQP sweep takes roughly 0.1 s, and trust-constr about 3 s.

---

## 📁 Project Structure

| File | Role |
|---|---|
| `app.py` | Core model: physics equations and multipliers, `Params` dataclass, spacing constraint, SLSQP / trust-constr routines, multi-start search, and the needle-count sweep. Also runs as a command-line report. |
| `frontend.py` | Streamlit dashboard: sidebar controls, result cards, flow-vs-needles chart, design table with CSV export, penalty breakdown, force sensitivity analysis, and model notes. Imports the model from `app.py`. |
| `requirements.txt` | Pinned dependencies for **reproducibility**: `pandas`, `numpy`, `scipy`, `plotly`, `streamlit`. |

---

## 💡 Use Cases

- **Design exploration:** quickly test how insertion force, skin stiffness (puncture stress), and footprint limits shift the best needle count before committing to fabrication.
- **Teaching and communication:** show students or collaborators how competing physical effects produce an interior optimum, using the penalty-breakdown view.
- **Solver benchmarking:** compare SLSQP and trust-constr on a small nonlinear problem with bounds and a nonlinear inequality constraint.
- **Next-step extensions:** replace the calibrated force with measured skin-puncture data, add a penalty for extreme radius or length so optima land inside the bounds, add a mechanism that penalizes too few needles (such as skin draping between widely spaced needles), or add pain and tissue-damage limits to the objective.

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
