# Drone Arm Impact Simulator

Simulates a drone arm striking a human head-neck system, coupling structural failure of the arm with biomechanical injury assessment. The model is designed to evaluate the risk posed by a falling or colliding drone to a bystander's head.

---

## Physics Overview

### Drone arm — Euler-Bernoulli beam (FEM)

The arm is discretised into a 1D co-rotational finite element mesh. Each element uses the standard 6-DOF Euler-Bernoulli stiffness matrix in its local frame, rotated into the global frame using the current chord angle. This co-rotational approach correctly handles large rigid-body rotations while keeping the local elastic deformations small.

- **Node 0** carries the concentrated body mass of the drone chassis
- **Last node** carries the rotor motor mass
- All remaining nodes have distributed beam mass (lumped diagonal mass matrix)
- Rayleigh damping is applied with coefficients auto-computed from a target damping ratio at two user-specified frequencies

### Head-neck system — 2-DOF double pendulum

The human head-neck is modelled as two rigid bars pinned end-to-end:

- **Bar 1 (neck):** rotates by angle φ about a fixed base point
- **Bar 2 (head):** rotates by angle θ about the neck tip

The equations of motion are derived from the Lagrangian, producing a 2×2 coupled mass matrix where the off-diagonal term `m_head · L_neck · r_com · cos(θ−φ)` represents the geometric inertia coupling between the two bars. Passive resistance is provided by a two-level series spring-damper (lower joint at the base, upper joint at the neck-head junction). Voluntary muscle activation follows an exponential ramp after a user-specified reflex delay.

An optional **scalp compliance sub-mass** sits between the contact surface and the head rigid body, low-pass filtering the impact pulse and reducing peak head CoM acceleration at high velocities.

### Contact model — Hunt-Crossley (Hertzian)

Contact between each beam node and the head/neck segments is detected by proximity to the closest point on each rigid segment. The normal force follows:

```
F_n = k · δ^n  −  c · δ^(n-1) · v_n        (Hertzian, n = 1.5 by default)
F_n = k · δ    −  c · v_n                   (linear, optional)
```

The stiffness `k` is auto-computed from an energy balance so that peak penetration stays within a target value at the given impact velocity. The damping coefficient `c` is derived from a target coefficient of restitution. Coulomb friction is also included.

### Damage and plasticity

Two independent degradation mechanisms act on each beam element:

**Plasticity** — a closest-point return mapping on a coupled N–M linear interaction yield surface with linear isotropic hardening. The yield surface expands with the accumulated plastic multiplier α, and both stiffness and yield limits are degraded by the scalar damage factor `(1−d)^n`.

**Damage** — a scalar damage variable `d` grows once stress exceeds `damage_init_ratio × σ_u` for at least `incubation_steps` consecutive steps, at a rate proportional to the stress overshoot divided by a time constant `damage_tau`. When `d ≥ 0.99` or stress exceeds `σ_u` directly, the element is killed.

Plasticity return mapping is applied **before** damage evaluation each step, so the damage model always sees the post-plasticity elastic stress.

### Time integration — Newmark-β

The beam and head-neck subsystems are integrated with the constant-average-acceleration Newmark-β scheme (β = 0.25, γ = 0.5), which is unconditionally stable. Each step uses a predictor-corrector structure:

1. Predict displacements and velocities at t + dt
2. Evaluate contact forces and head-neck accelerations at the predicted state
3. Solve the implicit corrector equation for the beam
4. Apply plasticity return mapping
5. Check damage and fracture; re-seed accelerations if elements break

---

## Injury Metrics

After the simulation, three standard biomechanical injury criteria are computed from the head CoM acceleration and neck load history:

| Metric | Description | Limit |
|--------|-------------|-------|
| **HIC15** | Head Injury Criterion over a 15 ms rolling window: `(t2−t1) · ā^2.5` | 700 |
| **BrIC** | Brain Rotational Injury Criterion (sagittal plane): `ω_peak / ω_yC` | 1.0 |
| **Nij** | Neck injury index: `Fz/Fint + My/Mint`, sign-dependent intercepts | 1.0 |

Each metric is also converted to an **AIS 3+ injury probability** using published logistic or Weibull fits. The full head CoM acceleration curve is exported to `head_acceleration.mat` for further post-processing in MATLAB.

---

## Installation

Python 3.8 or later is required. Install dependencies with:

```bash
pip install numpy scipy matplotlib
```

No other packages are needed.

---

## Usage

```bash
python COUPLED_SYSTEM_5_M_S_WITH_PLASTICITY_2_3_1.py
```

The script runs the simulation, displays a real-time animation of the impact, then shows two post-processing figures: the head acceleration curve with the HIC window highlighted, and a four-panel plasticity report. Progress is printed to the console every ~5% of simulation time.

All parameters are controlled through three dataclasses at the top of the file — no command-line arguments are needed.

---

## Parameters

### `BeamParams` — drone arm

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L0` | 0.07 m | Total beam length |
| `b`, `h` | 0.020 m | Cross-section width and height |
| `E` | 2.6 GPa | Young's modulus |
| `rho` | 1198 kg/m³ | Material density |
| `m_body` | 0.85 kg | Drone body concentrated mass |
| `m_motor` | 0.028 kg | Rotor motor concentrated mass |
| `n_nodes` | 20 | Number of FEM nodes |
| `v0` | 19 m/s | Impact velocity |
| `sigma_y` | 55 MPa | Initial yield stress |
| `H_hard` | 260 MPa | Linear hardening modulus |
| `ultimate_strength` | 90 MPa | Fracture stress |
| `use_plasticity` | True | Enable coupled N–M plasticity |
| `use_hertz` | True | Hertzian vs linear contact |
| `use_auto_k_contact` | True | Auto-compute k from energy balance |
| `delta_max_target` | 5 mm | Target peak penetration depth |

### `HeadNeckParams` — head-neck system

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L_neck` | 0.10 m | Neck bar length |
| `L_head` | 0.15 m | Head bar length |
| `m_head` | 5.5 kg | Head mass |
| `m_neck` | 1.6 kg | Neck mass |
| `k_global` | 10 N/m | Passive joint stiffness |
| `use_muscle` | True | Enable muscle reflex |
| `t_delay` | 70 ms | Muscle reflex delay |
| `use_scalp_layer` | True | Enable scalp compliance sub-mass |
| `HIC_limit` | 700 | HIC injury threshold |

### `SimParams` — integration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 1 µs | Time step (auto-adjusted if use_auto_contact) |
| `T_end` | 20 ms | Total simulation duration |
| `frame_stride` | 30 | Steps between animation frames |

---

## Outputs

| Output | Description |
|--------|-------------|
| Console injury report | HIC15, BrIC, Nij, AIS3+ probabilities |
| Animation window | Real-time beam deformation + head-neck motion |
| Head acceleration plot | CoM acceleration in g's, HIC window shaded |
| Plasticity plot | Final ε_p, κ_p, α_p per element + time evolution |
| `head_acceleration.mat` | Time vector and acceleration curve for MATLAB |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array math, linear algebra |
| `scipy` | Matrix solves, MATLAB export |
| `matplotlib` | Animation and plots |
