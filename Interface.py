"""
╔══════════════════════════════════════════════════════╗
║   DRONE IMPACT SAFETY SIMULATOR  —  GUI  v3          ║
║   Backend: Drone_Soft_Target_Impact_Model            ║
╚══════════════════════════════════════════════════════╝
Requirements: pip install customtkinter matplotlib numpy scipy
Run:          python drone_impact_gui_v3.py
"""

import threading
import queue
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
import math
import os

import numpy as np
import scipy.linalg as la
import scipy.io
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
import math
import os


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation geometry constants
# ─────────────────────────────────────────────────────────────────────────────
# Half-width of each arm element rectangle (perpendicular to the beam axis).
ARM_HALF_W   = 0.0022   # m  — roughly matches the real cross-section

# The body (node 0) is drawn as a square centred on that node.
BODY_HALF    = 0.006    # m  — half-side of the square body

# The motor (last node) is drawn as a rectangle aligned with the last element.
MOTOR_HALF_W = 0.006    # m  — half-width perpendicular to arm
MOTOR_HALF_L = 0.009    # m  — half-length along arm axis

# Colours (face / edge)
COL_ARM_FACE   = '#4FC3F7'   # light blue — healthy arm elements
COL_ARM_EDGE   = '#0277BD'   # dark blue edge
COL_BROKEN_F   = '#EF9A9A'   # red face when element is dead
COL_BROKEN_E   = '#B71C1C'
COL_BODY_FACE  = '#78909C'   # blue-grey body square
COL_BODY_EDGE  = '#263238'
COL_MOTOR_FACE = '#FF8A65'   # orange motor rectangle
COL_MOTOR_EDGE = '#BF360C'
COL_NECK       = '#AB47BC'   # purple neck segment
COL_HEAD       = '#7E57C2'   # deeper purple head segment


# ─────────────────────────────────────────────────────────────────────────────
# Parameter classes
# ─────────────────────────────────────────────────────────────────────────────

# =============================================================================
# BeamParams
# Holds every physical and numerical parameter that describes the drone arm.
# The arm is modelled as an Euler-Bernoulli beam with n_nodes nodes.
# Node 0 carries the main drone body mass; the last node carries the rotor mass.
# Derived cross-section properties (A, I, c, N_p, M_p) are computed from b and h
# via @property so they are always consistent with the raw dimensions.
# =============================================================================
class BeamParams:
    def __init__(self):
        self.L0  = 0.07  #[m]
        self.E   = 2.6e9 #[Pa]
        self.rho = 1198  #[kg/m³]

        # ── Cross-section (rectangular b × h) ────────────────────────────────
        # b : width  (in-plane, perpendicular to bending)   [m]
        # h : height (out-of-plane, in the bending direction) [m]
        # A, I and c are derived automatically — do NOT set them directly.
        self.b = 0.020   # section width  [m]  → 20 mm
        self.h = 0.020   # section height [m]  →  20 mm

        self.m_body  = 0.85  #[kg]
        self.m_motor = 0.028 #[kg]
        self.n_nodes = 20
        self.initial_angle_deg = 0.0 # deg
        self.v0 = 19 #m/s
        # If True  → velocity is directed along the beam axis (same angle as
        #            initial_angle_deg).  This is the natural free-flight case.
        # If False → velocity is purely horizontal (0°, i.e. vx = v0, vy = 0)
        #            regardless of the beam angle.  Use this when the drone
        #            travels horizontally and only its arm is angled.
        self.velocity_angled = False
        self.tip_x0 = -0.02  #[m]
        self.tip_y0 = 0.175  #[m]
        self.ultimate_strength = 90e6 #[Pa]
        self.damage_init_ratio = 0.70 
        self.damage_tau = 2e-4 #when this decrease
        self.t_inc = 50e-6  #[s]
        self.incubation_steps = 5 
        self.max_final_fail_per_step = 1
        self.damage_exponent = 1
        self.k_contact = 4.5e5   #[N/m or N/m^1.5 for Hertz]
        self.c_contact = 10.0 # [N*s/m]
        self.mu = 0.2
        self.e_target = 0.45
        self.dt_ratio_contact = 0.05
        self.use_auto_contact = True

        # ── Hertzian contact ──────────────────────────────────────────────────
        # use_hertz      : True  → F = k·δ^n  (nonlinear, geometry-consistent)
        #                  False → F = k·δ    (linear penalty, original model)
        # hertz_exponent : 1.5 for sphere-on-flat (Hertz); 1.0 for cylinder-on-flat.
        # use_auto_k_contact  : automatically compute k_contact each run from the
        #                       energy balance so that peak penetration does not
        #                       exceed delta_max_target.
        # delta_max_target    : desired maximum penetration depth [m].
        self.use_hertz          = True
        self.hertz_exponent     = 1.5
        self.use_auto_k_contact = True
        self.delta_max_target   = 0.005  # [m]  5 mm — reasonable blunt-impact limit
        self.use_auto_incubation = True
        self.use_auto_rayleigh = True
        self.zeta_target = 0.02
        self.f_rayleigh_1 = 200
        self.f_rayleigh_2 = 1500
        self.rayleigh_alpha = 0.0
        self.rayleigh_beta = 0.0
        self.contact_radius = 0.010
        self.k_eps = 1e-6
        self.m_eps = 1e-9

        # ── Plasticity parameters ─────────────────────────────────────────────
        # Linear isotropic hardening on a coupled N–M yield surface.
        # sigma_y  : initial yield stress [Pa]  — must be ≤ ultimate_strength.
        # H_hard   : linear hardening modulus [Pa].  0 = elastic-perfectly-plastic;
        #            typical range E/100 … E/10 for polymers/composites.
        # use_plasticity : set False to disable (pure damage model).
        self.sigma_y       = 55e6   # [Pa]  initial yield stress
        self.H_hard        = 260e6  # [Pa]  hardening modulus (~E/10)
        self.use_plasticity = True

    # ── Derived cross-section properties (computed from b and h) ─────────────
    @property
    def A(self):
        """Cross-sectional area  [m²]  = b × h"""
        return self.b * self.h

    @property
    def I(self):
        """Second moment of area [m⁴]  = b·h³/12  (bending about the h axis)"""
        return self.b * self.h**3 / 12.0

    @property
    def c(self):
        """Outer-fibre distance  [m]   = h/2  (for stress recovery: σ = N/A + M·c/I)"""
        return self.h / 2.0

    @property
    def N_p(self):
        """Initial plastic axial force capacity  [N]  = σ_y · A"""
        return self.sigma_y * self.A

    @property
    def M_p(self):
        """Initial full plastic moment of rectangular section  [N·m]  = σ_y · b · h² / 4"""
        return self.sigma_y * self.b * self.h**2 / 4.0


# =============================================================================
# HeadNeckParams
# Holds every parameter for the head-neck double-pendulum model.
# The neck is bar 1 (angle phi, length L_neck) attached at hp.base.
# The head is bar 2 (angle theta, length L_head) attached at the neck tip.
# Passive stiffness/damping is split into an upper (neck-head joint) spring
# and a lower (base-neck) spring in series via split_series().
# An optional scalp compliance sub-mass low-pass filters the contact force.
# update_inertias() must be called after changing any inertia-related field.
# =============================================================================
class HeadNeckParams:
    def __init__(self):
        self.L_neck = 0.10
        self.L_head = 0.15
        self.base = np.array([0.0, 0.0], dtype=float)
        self.phi0 = np.pi / 2
        self.theta0 = np.pi / 2
        self.phi_rest = self.phi0
        self.theta_rest = self.theta0
        self.m_head = 5.5
        self.m_neck = 1.6
        self.r_head_inertia = 0.060
        self.r_head_com = 0.060   # m  — occipital-condyles-to-CoM offset
        self.k_global = 10.0
        self.c_global = 10
        self.r_ratio = 1.5
        self.use_muscle = True
        self.t_delay = 0.070
        self.tau_act = 0.030
        self.k_muscle = 60.0
        self.c_muscle = 1.0
        self.HIC_window = 0.015
        self.HIC_limit = 700.0
        self.omega_yC = 56.45       # sagittal (flexion/extension) critical angular velocity [rad/s]
        self.Fint_tension = 6806.0
        self.Fint_compression = 6160.0
        self.Mint_flexion = 310.0
        self.Mint_extension = 135.0
        self.Nij_limit = 1.0

        # ── Scalp compliance layer ────────────────────────────────────────────
        # Models the local deformability of scalp + outer skull diploe.
        # The contact force hits m_scalp first; m_scalp is connected to the
        # head rigid body by k_scalp / c_scalp.  This low-pass filters the
        # contact pulse, reducing peak head CoM acceleration at high velocity.
        #
        # Calibration guide (from Willinger et al. blunt impact literature):
        #   m_scalp  ≈ 0.3 – 0.8 kg   (effective local inertia of scalp+skull cap)
        #   k_scalp  ≈ 1e5 – 5e5 N/m  (scalp + diploe stiffness)
        #   c_scalp  from zeta_scalp = c/(2*sqrt(k*m)) ≈ 0.3 – 0.5
        # Start with the values below and tune m_scalp and k_scalp to match FEM.
        self.use_scalp_layer = True
        self.m_scalp         = 0.5     # [kg]
        self.k_scalp         = 1.3e5   # [N/m]
        self.zeta_scalp      = 0.4     # damping ratio — c_scalp derived automatically
        self.update_inertias()

    def update_inertias(self):
        # I_head: rotational inertia of head about neck-head joint, using parallel-axis
        #         theorem — r_head_inertia is the gyration radius, r_head_com is the
        #         offset from the joint to the head centre of mass.
        self.I_head = self.m_head * (self.r_head_inertia**2 + self.r_head_com**2)
        # I_bar: neck modelled as a uniform bar rotating about its base end: I = mL²/3
        self.I_bar  = self.m_neck * self.L_neck**2 / 3.0
        # c_scalp: damping coefficient derived from the user-specified damping ratio
        #          zeta_scalp so c = 2 * zeta * sqrt(k * m)
        self.c_scalp = 2.0 * self.zeta_scalp * np.sqrt(self.k_scalp * self.m_scalp)


# =============================================================================
# SimParams
# Time-integration settings: step size, total duration, and animation stride.
# =============================================================================
class SimParams:
    def __init__(self):
        self.dt = 1e-6
        self.T_end = 0.02
        self.frame_stride = 30
        self.animation_interval = 10


# =============================================================================
# FEM helper functions
# =============================================================================

def beam_element_stiffness_local(E, I, A, L):
    # Returns the 6x6 Euler-Bernoulli element stiffness matrix in the LOCAL
    # element frame. DOF order: [u1, v1, theta1, u2, v2, theta2].
    # Axial terms scale as EA/L; bending terms scale as EI/L^3 (shear) and
    # EI/L^2 / EI/L (moment coupling). The matrix is symmetric and singular
    # (rigid-body modes not restrained in local frame).
    return np.array([
        [ A*E/L,           0.0,          0.0,  -A*E/L,           0.0,          0.0],
        [ 0.0,   12*E*I/L**3,  6*E*I/L**2,   0.0,  -12*E*I/L**3,  6*E*I/L**2],
        [ 0.0,    6*E*I/L**2,    4*E*I/L,    0.0,   -6*E*I/L**2,    2*E*I/L],
        [-A*E/L,           0.0,          0.0,   A*E/L,           0.0,          0.0],
        [ 0.0,  -12*E*I/L**3, -6*E*I/L**2,   0.0,   12*E*I/L**3, -6*E*I/L**2],
        [ 0.0,    6*E*I/L**2,    2*E*I/L,    0.0,   -6*E*I/L**2,    4*E*I/L],
    ], dtype=float)


def T_beam_2d(angle_rad):
    # Returns the 6x6 rotation matrix that transforms a local-frame DOF vector
    # [u1_loc, v1_loc, theta1, u2_loc, v2_loc, theta2] into the global frame.
    # Block-diagonal: each 2x2 translational block rotates by angle_rad;
    # the rotational DOFs (theta) pass through unchanged.
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [ c,  s, 0., 0., 0., 0.],
        [-s,  c, 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0.,  c,  s, 0.],
        [0., 0., 0., -s,  c, 0.],
        [0., 0., 0., 0., 0., 1.],
    ], dtype=float)


def corot_local_deformation(x0, u, i, j, L0_elem,
                            eps_p_elem=0.0, kappa_p_elem=0.0):
    # Computes the co-rotational elastic local deformation vector for element (i,j).
    #
    # Steps:
    #   1. Compute the reference chord angle alpha_0 from the undeformed positions.
    #   2. Compute the current chord angle alpha from deformed positions.
    #   3. Remove rigid-body rotation: subtract delta_alpha from each nodal rotation
    #      and wrap with atan2(sin,cos) to avoid 2pi branch-cut jumps after fracture.
    #   4. Subtract plastic contributions:
    #        - Plastic axial elongation: eps_p * L0  from the chord elongation.
    #        - Plastic curvature: ±kappa_p * L0/2 added antisymmetrically to the
    #          end rotations (negative at node i, positive at node j).
    #
    # Returns: (d_local[6], alpha)
    #   d_local = [0, 0, phi_i_e, d_axial_e, 0, phi_j_e]  (elastic deformation only)
    #   alpha   = current chord angle [rad]  (used by caller to build rotation matrix T)

    # reference chord angle (from x0 only, never changes)
    xi0 = x0[3*i];    yi0 = x0[3*i+1]
    xj0 = x0[3*j];    yj0 = x0[3*j+1]
    alpha_0 = np.arctan2(yj0 - yi0, xj0 - xi0)

    # current chord geometry
    xi = xi0 + u[3*i];    yi = yi0 + u[3*i+1]
    xj = xj0 + u[3*j];    yj = yj0 + u[3*j+1]
    dx = xj - xi;  dy = yj - yi
    L_cur = np.sqrt(dx*dx + dy*dy)
    alpha = np.arctan2(dy, dx)

    # ── Co-rotational wrapping: atan2 prevents 2π branch-cut jumps ─────────────
    delta_alpha = alpha - alpha_0
    phi_i_raw   = u[3*i+2] - delta_alpha
    phi_j_raw   = u[3*j+2] - delta_alpha
    phi_i = np.arctan2(np.sin(phi_i_raw), np.cos(phi_i_raw))
    phi_j = np.arctan2(np.sin(phi_j_raw), np.cos(phi_j_raw))

    # ── Plastic strain subtraction ───────────────────────────────────────────
    # Elastic elongation = total elongation − plastic elongation
    d_axial_e = (L_cur - L0_elem) - eps_p_elem * L0_elem
    # Uniform plastic curvature κ_p adds −κ_p·L/2 at node i and +κ_p·L/2 at node j
    phi_i_e = phi_i - kappa_p_elem * L0_elem / 2.0
    phi_j_e = phi_j + kappa_p_elem * L0_elem / 2.0

    # node i is origin: u1=0, v1=0; node j carries elastic elongation
    return np.array([0.0, 0.0, phi_i_e, d_axial_e, 0.0, phi_j_e]), alpha


def generate_mesh(n_nodes, L0):
    # Creates a 1-D mesh of n_nodes equally spaced nodes along [0, L0] and
    # builds the connectivity list [(0,1), (1,2), ..., (n-2, n-1)].
    return np.linspace(0.0, L0, n_nodes), [(i, i + 1) for i in range(n_nodes - 1)]


def element_length(L0, n_nodes):
    # Returns the uniform element length for a mesh with n_nodes nodes over L0.
    return L0 / (n_nodes - 1)


def compute_initial_node_positions(bp):
    # Computes the (x, y) coordinates of all beam nodes in the global frame
    # at t = 0. The tip node is placed at (bp.tip_x0, bp.tip_y0) and the
    # remaining nodes are spaced ds apart along the beam axis direction given
    # by bp.initial_angle_deg.
    n   = bp.n_nodes
    ang = np.deg2rad(bp.initial_angle_deg)
    ds  = bp.L0 / (n - 1)
    dx, dy = np.cos(ang) * ds, np.sin(ang) * ds
    x_tip, y_tip = bp.tip_x0, bp.tip_y0
    x0_nodes = np.array([x_tip - (n - 1 - i) * dx for i in range(n)], dtype=float)
    y0_nodes = np.array([y_tip - (n - 1 - i) * dy for i in range(n)], dtype=float)
    return x0_nodes, y0_nodes


def dof_map_for_element(i, j):
    # Returns the 6-element index array that maps local DOFs [u1,v1,th1,u2,v2,th2]
    # to the positions of nodes i and j in the global DOF vector.
    # Each node occupies 3 consecutive DOFs: (3*node, 3*node+1, 3*node+2).
    return np.array([3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2], dtype=int)


def compute_contact_damping_from_restitution(k_w, m_eff, e):
    # Derives the contact damping coefficient c and damping ratio zeta from the
    # coefficient of restitution e using the logarithmic-decrement formula for a
    # damped harmonic oscillator:
    #   zeta = -ln(e) / sqrt(pi^2 + ln(e)^2)
    #   c    = 2 * zeta * sqrt(k_w * m_eff)
    e    = np.clip(e, 1e-6, 0.999999)
    ln_e = np.log(e)
    zeta = -ln_e / np.sqrt(np.pi**2 + ln_e**2)
    return 2.0 * zeta * np.sqrt(k_w * m_eff), zeta


def compute_contact_time_and_dt(k_w, m_eff, zeta, dt_ratio=0.05):
    # Estimates the contact half-period tc = pi / (omega0 * sqrt(1 - zeta^2))
    # for a linearised spring-mass-damper system and suggests a stable time step
    # dt = dt_ratio * tc so that tc is resolved by at least 1/dt_ratio steps.
    omega0 = np.sqrt(k_w / m_eff)
    tc     = np.pi / (omega0 * np.sqrt(max(1e-12, 1.0 - zeta**2)))
    return tc, dt_ratio * tc


def compute_rayleigh_from_two_freqs(zeta_target, f1, f2):
    # Solves the 2x2 linear system for Rayleigh coefficients alpha and beta
    # that produce damping ratio zeta_target at both f1 and f2 [Hz].
    # Uses: zeta = alpha/(2*omega) + beta*omega/2  at each frequency.
    w1 = 2.0 * np.pi * f1
    w2 = 2.0 * np.pi * f2
    A  = np.array([[1.0 / w1, w1], [1.0 / w2, w2]], dtype=float)
    b  = 2.0 * np.array([zeta_target, zeta_target], dtype=float)
    return np.linalg.solve(A, b)


def compute_hertz_k_from_penetration(bp, m_total):
    # Computes the Hertzian contact stiffness k_contact via an energy balance so
    # that peak penetration does not exceed bp.delta_max_target at impact velocity v0.
    #
    # Energy balance (rigid body, no damping):
    #   0.5 * M_total * v0^2  =  k * delta_max^(n+1) / (n+1)
    #   => k = (n+1) * M_total * v0^2 / (2 * delta_max^(n+1))
    #
    # The linearised stiffness k_lin = n * k * delta_max^(n-1) is also returned
    # so that the auto-damping and auto-dt routines can use a consistent tangent.
    # Note: k is velocity-dependent — a faster drone needs a stiffer contact
    # to keep peak penetration bounded.
    n       = bp.hertz_exponent
    delta   = bp.delta_max_target
    k       = (n + 1.0) * m_total * bp.v0**2 / (2.0 * delta**(n + 1.0))
    # Linearised stiffness at peak penetration (used for c_contact and dt)
    k_lin   = n * k * delta**(n - 1.0)
    return k, k_lin


def print_parameter_report(bp, hp, sp, m_eff, m_total, zeta_contact, tc):
    print("[Auto parameter report]")
    print(f"  m_eff          = {m_eff:.6e} kg")
    print(f"  m_total        = {m_total:.6e} kg  (used for k_contact energy balance)")
    contact_model = f"Hertzian  F = k·δ^{bp.hertz_exponent}" if bp.use_hertz else "Linear    F = k·δ"
    print(f"  contact model  = {contact_model}")
    print(f"  k_contact      = {bp.k_contact:.6e}  (auto={bp.use_auto_k_contact}, δ_max_target={bp.delta_max_target*1e3:.1f} mm)")
    print(f"  c_contact      = {bp.c_contact:.6e} N*s/m")
    print(f"  contact zeta   = {zeta_contact:.6f}")
    print(f"  contact time tc= {tc:.6e} s")
    print(f"  dt             = {sp.dt:.6e} s")
    print(f"  incubation_steps={bp.incubation_steps}")
    print(f"  Rayleigh alpha = {bp.rayleigh_alpha:.6e}")
    print(f"  Rayleigh beta  = {bp.rayleigh_beta:.6e}")
    print(f"  I_head         = {hp.I_head:.6e} kg*m^2 (m={hp.m_head:.3f}, r_gyr={hp.r_head_inertia:.3f}, r_com={hp.r_head_com:.3f})")
    print(f"  I_neck         = {hp.I_bar:.6e}  kg*m^2 (m={hp.m_neck:.3f}, L={hp.L_neck:.3f}, formula=m*L²/3)")
    if hp.use_scalp_layer:
        print(f"  [Scalp layer ON]  m={hp.m_scalp:.3f} kg  k={hp.k_scalp:.3e} N/m  "
              f"c={hp.c_scalp:.3e} N*s/m  zeta={hp.zeta_scalp:.2f}")
    else:
        print(f"  [Scalp layer OFF]  head is rigid")
    if bp.use_plasticity:
        print(f"  [Plasticity ON]")
        print(f"  sigma_y        = {bp.sigma_y:.3e} Pa  (yield stress)")
        print(f"  H_hard         = {bp.H_hard:.3e} Pa  (linear hardening modulus, H/E = {bp.H_hard/bp.E:.4f})")
        print(f"  N_p0           = {bp.N_p:.3e} N   (initial plastic axial limit = sigma_y * A)")
        print(f"  M_p0           = {bp.M_p:.3e} N*m (initial plastic moment    = sigma_y * b * h^2 / 4)")
        print(f"  sigma_damage   = {bp.damage_init_ratio * bp.ultimate_strength:.3e} Pa  "
              f"(damage onset; {bp.damage_init_ratio*100:.0f}% of sigma_u = {bp.ultimate_strength:.3e} Pa)")
        if bp.sigma_y >= bp.damage_init_ratio * bp.ultimate_strength:
            print(f"  *** NOTE: sigma_y >= sigma_damage — plasticity and damage onset overlap. ***")
            print(f"  *** Consider lowering sigma_y so plastic flow precedes damage.          ***")
    else:
        print(f"  [Plasticity OFF]")


def assemble_lumped_mass(bp, elements):
    # Builds the global diagonal (lumped) mass matrix for the beam.
    #
    # Translational lumping: half of each element's mass (rho*A*L_e) is assigned
    # to each of its two end nodes in the x and y DOFs.
    # Rotational lumping: each element contributes m_e * L_e^2 / 12 to the
    # rotational DOF of each of its end nodes (consistent with a uniform bar).
    # Concentrated masses: body mass added to node 0; motor mass to last node.
    # Concentrated rotational inertias: body modelled as a square plate,
    # motor modelled as a thin rod aligned with the arm.
    # Any node that ends up with zero rotational mass gets a small regularisation
    # term m_eps to avoid singularity in the equation-of-motion solve.
    n    = bp.n_nodes
    ndof = 3 * n
    M    = np.zeros((ndof, ndof), dtype=float)
    L_e  = element_length(bp.L0, bp.n_nodes)
    m_e  = bp.rho * bp.A * L_e
    for (i, j) in elements:
        for node in (i, j):
            M[3*node,   3*node]   += 0.5 * m_e
            M[3*node+1, 3*node+1] += 0.5 * m_e
    J_e = m_e * (L_e**2) / 12.0
    for (i, j) in elements:
        for node in (i, j):
            M[3*node+2, 3*node+2] += 0.5 * J_e
    # Translational mass for body and motor
    M[0,  0]  += bp.m_body
    M[1,  1]  += bp.m_body
    M[-3, -3] += bp.m_motor
    M[-2, -2] += bp.m_motor
    # Rotational inertia for body (square, side = 2*BODY_HALF) and motor (thin rod)
    I_body  = bp.m_body  * (2.0 * BODY_HALF)**2 / 6.0
    I_motor = bp.m_motor * (2.0 * MOTOR_HALF_L)**2 / 12.0
    M[2,  2]  += I_body
    M[-1, -1] += I_motor
    for node in range(n):
        if M[3*node+2, 3*node+2] == 0.0:
            M[3*node+2, 3*node+2] = 1e-12
    return M


def alive_adjacency(n_nodes, elements, element_alive):
    # Builds an adjacency list that only includes connections through alive (intact)
    # elements. Used by connected_components to find disconnected fragments.
    adj = [[] for _ in range(n_nodes)]
    for e_idx, (i, j) in enumerate(elements):
        if not element_alive[e_idx]:
            continue
        adj[i].append(j)
        adj[j].append(i)
    return adj


def connected_components(n_nodes, elements, element_alive):
    # Returns a list of node groups (components) where each group is a list of
    # node indices that are still connected through alive elements.
    # Uses iterative DFS (stack-based) to avoid recursion limit on large meshes.
    adj  = alive_adjacency(n_nodes, elements, element_alive)
    seen = np.zeros(n_nodes, dtype=bool)
    comps = []
    for start in range(n_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            uu = stack.pop()
            comp.append(uu)
            for vv in adj[uu]:
                if not seen[vv]:
                    seen[vv] = True
                    stack.append(vv)
        comps.append(comp)
    return comps


def stabilize_components(K, M, bp, elements, element_alive):
    # Adds small diagonal penalties (k_eps to K, m_eps to M) to every DOF of
    # any fragment that is disconnected from both anchor nodes (node 0 = body,
    # node n-1 = motor). This prevents singular matrices after fracture events
    # without affecting the physical response of the main connected body.
    n     = bp.n_nodes
    comps = connected_components(n, elements, element_alive)
    anchor = {0, n - 1}
    for comp in comps:
        if any(node in anchor for node in comp):
            continue
        for node in comp:
            for dof in (3*node, 3*node+1, 3*node+2):
                K[dof, dof] += bp.k_eps
                M[dof, dof] += bp.m_eps
    return K, M


def assemble_K_alive_current_angle(bp, elements, element_alive, element_damage, x0, u):
    # Assembles the global tangent stiffness matrix using co-rotational kinematics.
    # For each alive element:
    #   1. Compute the current chord angle alpha via corot_local_deformation.
    #   2. Build the rotation matrix T(alpha) and transform the constant local
    #      stiffness: K_e = T^T * K_local * T  (global-frame element stiffness).
    #   3. Apply damage scaling: multiply by (1 - d)^n so cracked elements
    #      contribute proportionally less stiffness.
    #   4. Scatter K_e into the global K using the element DOF map.
    n       = bp.n_nodes
    K       = np.zeros((3 * n, 3 * n), dtype=float)
    L_e     = element_length(bp.L0, bp.n_nodes)
    K_local = beam_element_stiffness_local(bp.E, bp.I, bp.A, L_e)
    for e_idx, (i, j) in enumerate(elements):
        if not element_alive[e_idx]:
            continue
        dm       = dof_map_for_element(i, j)
        _, alpha = corot_local_deformation(x0, u, i, j, L_e)
        T        = T_beam_2d(alpha)
        scale    = max(0.0, (1.0 - float(element_damage[e_idx]))) ** bp.damage_exponent
        K_e      = scale * (T.T @ K_local @ T)
        K[np.ix_(dm, dm)] += K_e
    return K


def internal_forces_alive_current_angle(bp, elements, element_alive, element_damage, x0, u,
                                        eps_p=None, kappa_p=None):
    # Computes the global internal force vector f_int = K_e * d_local (in global coords)
    # for all alive elements, accounting for co-rotation and damage.
    # If plastic strains are provided, they are passed to corot_local_deformation
    # so that f_int reflects only the elastic (recoverable) part of the deformation.
    # The result is the restoring force that resists the external load in the EOM.
    n       = bp.n_nodes
    f_int   = np.zeros(3 * n, dtype=float)
    L_e     = element_length(bp.L0, bp.n_nodes)
    K_local = beam_element_stiffness_local(bp.E, bp.I, bp.A, L_e)
    for e_idx, (i, j) in enumerate(elements):
        if not element_alive[e_idx]:
            continue
        ep = float(eps_p[e_idx])   if eps_p   is not None else 0.0
        kp = float(kappa_p[e_idx]) if kappa_p is not None else 0.0
        dm               = dof_map_for_element(i, j)
        d_local, alpha   = corot_local_deformation(x0, u, i, j, L_e, ep, kp)
        T                = T_beam_2d(alpha)
        scale            = max(0.0, (1.0 - float(element_damage[e_idx]))) ** bp.damage_exponent
        f_int[dm]       += T.T @ ((scale * K_local) @ d_local)
    return f_int


def element_max_stress(bp, x0, u, elements, e_idx, element_damage,
                       eps_p=None, kappa_p=None):
    # Recovers the maximum combined stress in element e_idx using:
    #   sigma = N/A + M * c / I
    # where N is the peak absolute axial force (from either end node),
    # M is the peak absolute bending moment (from either end), and
    # c = h/2 is the outer-fibre distance for the rectangular section.
    # The local force vector is obtained from the elastic deformation via
    # co-rotation and damage scaling.
    i, j    = elements[e_idx]
    L_e     = element_length(bp.L0, bp.n_nodes)
    K_local = beam_element_stiffness_local(bp.E, bp.I, bp.A, L_e)
    ep = float(eps_p[e_idx])   if eps_p   is not None else 0.0
    kp = float(kappa_p[e_idx]) if kappa_p is not None else 0.0
    d_local, _ = corot_local_deformation(x0, u, i, j, L_e, ep, kp)
    scale      = max(0.0, (1.0 - float(element_damage[e_idx]))) ** bp.damage_exponent
    f_local    = (scale * K_local) @ d_local
    # f_local[0] = axial reaction at node i; f_local[3] = axial force at node j
    N = max(abs(f_local[0]), abs(f_local[3]))
    M = max(abs(f_local[2]), abs(f_local[5]))
    return (N / bp.A) + (M * bp.c / bp.I)


def update_damage_and_failure(bp, x0, u, elements, element_alive, element_damage,
                              over_limit_counter, dt, eps_p=None, kappa_p=None):
    # Evaluates stress in every alive element and evolves the scalar damage variable d.
    #
    # Damage onset: stress must exceed damage_init_ratio * sigma_u for at least
    #   incubation_steps consecutive steps (incubation counter per element).
    # Damage growth: once incubation is satisfied,
    #   dd = (dt / damage_tau) * (sigma - sigma0) / (sigma_u - sigma0)
    #   d  = min(1, d + dd)
    # Fracture: element is killed (element_alive = False) when d >= 0.99
    #   or stress exceeds sigma_u directly.
    # Rate limiting: at most max_final_fail_per_step elements can fracture
    #   per time step (prevents cascade kill in a single step).
    # Returns: (newly_broken_indices, stress_array_per_element)
    n_e       = len(elements)
    sigma_arr = np.full(n_e, np.nan, dtype=float)
    sigma0    = bp.damage_init_ratio * bp.ultimate_strength
    for e_idx in range(n_e):
        if not element_alive[e_idx]:
            continue
        sigma            = element_max_stress(bp, x0, u, elements, e_idx, element_damage,
                                              eps_p, kappa_p)
        sigma_arr[e_idx] = sigma
        over_limit_counter[e_idx] = (over_limit_counter[e_idx] + 1
                                     if sigma > sigma0 else 0)
        if over_limit_counter[e_idx] >= bp.incubation_steps and sigma > sigma0:
            denom = max(1e-12, bp.ultimate_strength - sigma0)
            r     = np.clip((sigma - sigma0) / denom, 0.0, 1.0)
            dd    = (dt / max(1e-12, bp.damage_tau)) * r
            element_damage[e_idx] = min(1.0, element_damage[e_idx] + dd)
    candidates = []
    for e_idx in range(n_e):
        if not element_alive[e_idx]:
            continue
        sigma = sigma_arr[e_idx]
        d     = element_damage[e_idx]
        if (np.isfinite(sigma) and sigma > 1.0 * bp.ultimate_strength) or (d >= 0.99):
            ratio = float(sigma / bp.ultimate_strength) if np.isfinite(sigma) else 0.0
            candidates.append((ratio, e_idx))
    candidates.sort(reverse=True, key=lambda t: t[0])
    newly = []
    for _, e_idx in candidates[: int(max(0, bp.max_final_fail_per_step))]:
        element_alive[e_idx]  = False
        element_damage[e_idx] = 1.0
        newly.append(e_idx)
    return newly, sigma_arr


def plastic_return_mapping(bp, x0, u, elements, element_alive, element_damage,
                           eps_p, kappa_p, alpha_p):
    # Performs a closest-point return mapping on the coupled N–M linear
    # interaction yield surface with linear isotropic hardening.
    #
    # Yield surface:
    #   Phi(N, M, alpha) = |N| / N_p(alpha) + M_max / M_p(alpha) - 1 <= 0
    #
    # Plastic capacity with linear hardening and damage degradation:
    #   N_p(alpha) = scale * (N_p0 + H_N * alpha)    N_p0 = sigma_y * A
    #   M_p(alpha) = scale * (M_p0 + H_M * alpha)    M_p0 = sigma_y * b * h^2 / 4
    #   scale      = (1 - d)^n   (damage reduces both elastic stiffness and yield limits)
    #
    # Associated flow rule:
    #   Delta_eps_p   = Delta_lambda * sign(N_tr) / N_p_eff
    #   Delta_kappa_p = Delta_lambda * sign(M_tr) / M_p_eff
    #   Delta_alpha   = Delta_lambda
    #
    # Plastic multiplier increment (analytic, first-order Taylor consistency):
    #   D = EA*scale/N_p^2 + EI*scale/M_p^2 + H_N*|N_tr|/N_p^3 + H_M*M_tr/M_p^3
    #   Delta_lambda = max(0, Phi_tr / D)
    #
    # This function is called AFTER the Newmark corrector and BEFORE
    # update_damage_and_failure, so damage always sees the post-plasticity
    # elastic stress (plastic flow dissipates energy first).
    L_e     = element_length(bp.L0, bp.n_nodes)
    EA      = bp.E * bp.A
    EI      = bp.E * bp.I
    N_p0    = bp.N_p    # sigma_y * A
    M_p0    = bp.M_p    # sigma_y * b * h^2 / 4
    H_N     = bp.H_hard * bp.A
    H_M     = bp.H_hard * bp.b * bp.h**2 / 4.0
    K_local = beam_element_stiffness_local(bp.E, bp.I, bp.A, L_e)   # constant for all elements

    for e_idx, (i, j) in enumerate(elements):
        if not element_alive[e_idx]:
            continue

        # ── Damage scaling (same convention as stiffness/force routines) ──────
        scale = max(0.0, (1.0 - float(element_damage[e_idx]))) ** bp.damage_exponent

        # ── Hardened and damage-degraded yield limits ─────────────────────────
        alpha   = float(alpha_p[e_idx])
        N_p_eff = scale * (N_p0 + H_N * alpha)
        M_p_eff = scale * (M_p0 + H_M * alpha)
        if N_p_eff < 1e-12 or M_p_eff < 1e-12:
            continue   # element fully degraded — no plastic capacity remaining

        # ── Trial elastic deformation (subtracts current plastic strains) ─────
        d_local, _ = corot_local_deformation(
            x0, u, i, j, L_e, eps_p[e_idx], kappa_p[e_idx])

        # ── Trial stress resultants (elastic predictor, damage-scaled) ────────
        f_tr  = (scale * K_local) @ d_local
        N_tr  = f_tr[3]          # axial at node j  (+ = tension)
        Mi_tr = f_tr[2]          # bending moment at node i
        Mj_tr = f_tr[5]          # bending moment at node j
        # Use the end with larger |M| to drive the curvature return
        if abs(Mi_tr) >= abs(Mj_tr):
            M_tr   = abs(Mi_tr)
            M_sign = np.sign(Mi_tr) if Mi_tr != 0.0 else 1.0
        else:
            M_tr   = abs(Mj_tr)
            M_sign = np.sign(Mj_tr) if Mj_tr != 0.0 else 1.0

        # ── Yield function at trial state ─────────────────────────────────────
        phi_tr = abs(N_tr) / N_p_eff + M_tr / M_p_eff - 1.0
        if phi_tr <= 0.0:
            continue   # elastic — nothing to correct

        # ── Consistent linearised denominator ────────────────────────────────
        # Elastic return: (dΦ/dN)(dN/dΔλ) + (dΦ/dM)(dM/dΔλ)
        #   = (1/N_p_eff)(EA·scale/N_p_eff) + (1/M_p_eff)(EI·scale/M_p_eff)
        # Hardening: −(dΦ/dN_p)(dN_p/dΔλ) − (dΦ/dM_p)(dM_p/dΔλ)
        #   = (|N_tr|/N_p_eff²)·H_N + (M_tr/M_p_eff²)·H_M
        denom = ((EA * scale) / N_p_eff**2
               + (EI * scale) / M_p_eff**2
               + H_N * abs(N_tr) / N_p_eff**3
               + H_M * M_tr     / M_p_eff**3)
        if denom < 1e-30:
            continue

        d_lam = max(0.0, phi_tr / denom)   # plastic multiplier increment (≥ 0)

        # ── Update plastic strains and hardening variable ─────────────────────
        eps_p[e_idx]   += d_lam * np.sign(N_tr) / N_p_eff   # engineering strain
        kappa_p[e_idx] += d_lam * M_sign         / M_p_eff   # curvature [rad/m]
        alpha_p[e_idx] += d_lam                               # accumulated multiplier


def split_series(k_global, c_global, r_ratio):
    # Decomposes a single global stiffness k_global and damping c_global into
    # two springs/dampers in series: an upper joint (neck-head) and a lower
    # joint (base-neck), with stiffness ratio r_ratio = k_lower / k_upper.
    # For two springs in series with ratio r:
    #   1/k_global = 1/k_upper + 1/k_lower  =>  k_upper = k_global*(1+r)/r
    #                                            k_lower = r * k_upper
    # The same ratio is applied to the damping coefficients.
    r        = r_ratio
    k_upper  = k_global * (1.0 + r) / r
    k_lower  = r * k_upper
    c_upper  = c_global * (1.0 + r) / r
    c_lower  = r * c_upper
    return k_upper, k_lower, c_upper, c_lower


def muscle_activation(t, t_delay, tau_act):
    # Returns the normalised muscle activation level A(t) in [0, 1].
    # Models electromechanical delay: zero activation before t_delay, then
    # an exponential ramp-up with time constant tau_act:
    #   A(t) = 0                            for t < t_delay
    #   A(t) = 1 - exp(-(t - t_delay) / tau_act)   for t >= t_delay
    return 0.0 if t < t_delay else 1.0 - np.exp(-(t - t_delay) / tau_act)


def headneck_accel(hp, t, theta, theta_dot, phi, phi_dot, tau_ext_theta, tau_ext_phi):
    # Solves the fully coupled 2-DOF double-pendulum equations of motion for the
    # head-neck system using a Lagrangian formulation.
    #
    # State variables:
    #   phi   : neck angle [rad]  (bar 1, rotating about hp.base)
    #   theta : head angle [rad]  (bar 2, rotating about neck tip)
    #
    # 2x2 coupled mass matrix:
    #   | M_nn  M_nh | = | I_bar + m_head*L_neck^2       m_head*L_neck*r_com*cos(theta-phi) |
    #   | M_nh  M_hh |   | m_head*L_neck*r_com*cos(theta-phi)   I_head                      |
    #
    # Passive joint torques from series spring-damper (via split_series):
    #   tau_upper acts between neck and head (restores theta relative to phi)
    #   tau_lower acts at the base (restores phi to rest angle)
    #
    # Voluntary muscle torque (if hp.use_muscle is True):
    #   tau_mus = -A(t) * (k_muscle * d_theta + c_muscle * theta_dot)
    #   Applied as a restoring torque on the head and reaction on the neck.
    #
    # Coriolis / centrifugal terms on RHS:
    #   phi   eq: +coupling * sin(theta-phi) * theta_dot^2
    #   theta eq: -coupling * sin(theta-phi) * phi_dot^2
    #
    # Returns: (theta_dd, phi_dd, My_OC)
    #   theta_dd, phi_dd : angular accelerations [rad/s^2]
    #   My_OC            : internal neck bending moment at the occipital condyles [N*m]
    #                      = tau_upper_on_head + tau_mus  (used for Nij)
    kU, kL, cU, cL   = split_series(hp.k_global, hp.c_global, hp.r_ratio)
    dphi              = phi   - hp.phi_rest
    dtheta            = theta - hp.theta_rest
    tau_upper_on_head = -kU * (dtheta - dphi) - cU * (theta_dot - phi_dot)
    tau_lower_on_neck = -kL * dphi - cL * phi_dot
    tau_upper_on_neck = -tau_upper_on_head
    tau_mus           = 0.0
    if hp.use_muscle:
        A       = muscle_activation(t, hp.t_delay, hp.tau_act)
        tau_mus = -A * (hp.k_muscle * dtheta + hp.c_muscle * theta_dot)

    dangle   = theta - phi
    cos_d    = np.cos(dangle)
    sin_d    = np.sin(dangle)
    coupling = hp.m_head * hp.L_neck * hp.r_head_com   # appears in both off-diagonal and Coriolis

    # 2x2 coupled mass matrix
    M_nn = hp.I_bar + hp.m_head * hp.L_neck**2
    M_hh = hp.I_head
    M_nh = coupling * cos_d
    Mmat = np.array([[M_nn, M_nh],
                     [M_nh, M_hh]], dtype=float)

    # Generalised forces including Coriolis terms
    # Muscle reaction torque on neck is -tau_mus (Newton's 3rd law)
    rhs_phi   = (tau_ext_phi   + tau_upper_on_neck + tau_lower_on_neck
                 + coupling * sin_d * theta_dot**2
                 - tau_mus)
    rhs_theta = (tau_ext_theta + tau_upper_on_head + tau_mus
                 - coupling * sin_d * phi_dot**2)

    sol      = np.linalg.solve(Mmat, np.array([rhs_phi, rhs_theta], dtype=float))
    phi_dd   = sol[0]
    theta_dd = sol[1]

    # My_OC: internal neck reaction moment at the occipital condyles (for Nij)
    My_OC = tau_upper_on_head + tau_mus

    return theta_dd, phi_dd, My_OC


def segment_endpoints(hp, phi, theta):
    # Computes the three key points of the two-bar head-neck linkage
    # given the current neck angle phi and head angle theta.
    # base  : fixed attachment point (hp.base)
    # joint : neck tip / head pivot  = base + L_neck * [cos(phi), sin(phi)]
    # tip   : head tip               = joint + L_head * [cos(theta), sin(theta)]
    base  = hp.base
    joint = base  + hp.L_neck * np.array([np.cos(phi),   np.sin(phi)],   dtype=float)
    tip   = joint + hp.L_head * np.array([np.cos(theta), np.sin(theta)], dtype=float)
    return base, joint, tip


def point_segment_closest(P, A, B):
    # Returns the point Q on segment AB closest to point P, along with the
    # parametric coordinate t in [0, 1] such that Q = A + t*(B-A).
    # Used for contact detection: a beam node is in contact when |P - Q| < r_contact.
    AB    = B - A
    denom = float(np.dot(AB, AB))
    if denom < 1e-15:
        return A.copy(), 0.0
    t = np.clip(float(np.dot(P - A, AB) / denom), 0.0, 1.0)
    return A + t * AB, t


def vel_of_point_on_segment(pivot, omega, point):
    # Returns the 2D velocity of a material point on a rigid segment rotating
    # about 'pivot' with scalar angular velocity omega [rad/s].
    # v = omega x r  =>  [-omega*ry, omega*rx]  (cross product in 2D)
    r = point - pivot
    return np.array([-omega * r[1], omega * r[0]], dtype=float)


def _segment_perp_normal(A, B):
    # Returns the unit normal perpendicular to segment AB, rotated 90 deg CCW.
    # Falls back to [0, 1] if the segment is degenerate (zero length).
    # Used as a fallback contact normal when a node lands exactly on a segment endpoint.
    seg     = B - A
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-15:
        return np.array([0.0, 1.0], dtype=float)
    return np.array([-seg[1], seg[0]], dtype=float) / seg_len


def _compute_contact_at_segment(P, Vp, A, B, seg_vel, bp, Q_pre=None):
    # Shared contact kernel for one beam node (position P, velocity Vp) against
    # one rigid segment (A to B, with material velocity seg_vel at the contact point).
    #
    # Contact is detected when the node-to-segment distance is less than bp.contact_radius.
    # Penetration depth: delta = contact_radius - distance
    #
    # Normal force models:
    #   Linear (use_hertz=False):
    #     Fn = max(0, k * delta - c * vn)
    #   Hertz / Hunt-Crossley (use_hertz=True):
    #     Fn = max(0, k * delta^n - c * delta^(n-1) * vn)
    #     The delta^(n-1) factor on damping is proportional to the tangent stiffness,
    #     preventing unphysical attraction force as the node separates.
    #
    # Friction (Coulomb, if bp.mu > 0):
    #   Ft = -mu * |Fn| * sign(vt) * t_hat   (opposes relative tangential sliding)
    #
    # Q_pre: optional pre-computed closest point to avoid redundant calls.
    #
    # Returns: (F_on_node, F_reaction_on_segment, contact_bool, n_hat, Q)
    r0    = bp.contact_radius
    Q     = Q_pre if Q_pre is not None else point_segment_closest(P, A, B)[0]
    d_vec = P - Q
    d     = float(np.linalg.norm(d_vec))
    zero2 = np.zeros(2, dtype=float)
    if d >= r0:
        return zero2, zero2, False, zero2, Q
    n_hat  = d_vec / d if d >= 1e-12 else _segment_perp_normal(A, B)
    pen    = r0 - d
    rel_v  = Vp - seg_vel
    vn     = float(np.dot(rel_v, n_hat))
    if bp.use_hertz:
        n_exp   = bp.hertz_exponent
        pen_n   = pen ** n_exp
        pen_nm1 = pen ** (n_exp - 1.0)
        Fn_mag  = max(0.0, bp.k_contact * pen_n - bp.c_contact * pen_nm1 * vn)
    else:
        Fn_mag  = max(0.0, bp.k_contact * pen - bp.c_contact * vn)
    Fn     = Fn_mag * n_hat
    t_dir  = np.array([-n_hat[1], n_hat[0]], dtype=float)
    vt     = float(np.dot(rel_v, t_dir))
    if bp.mu > 0.0 and abs(vt) > 1e-12:
        Ft = -bp.mu * abs(Fn_mag) * np.sign(vt) * t_dir
    else:
        Ft = zero2.copy()
    F       = Fn + Ft
    F_react = -F
    return F, F_react, True, n_hat, Q


def contact_forces_beam_vs_segments(bp, hp, x0, u, v, phi, phi_dot, theta, theta_dot,
                                    element_alive, elements):
    # Computes contact forces between every live beam node and the two rigid
    # head-neck segments (head segment checked first, then neck segment).
    #
    # Live-node rule: a node participates in contact if it belongs to at least one
    # intact element, OR if it is node 0 (body) or node n-1 (motor), which always
    # participate because they carry concentrated masses. Isolated splinter nodes
    # from a fracture are suppressed so they do not generate spurious contact forces.
    #
    # For each node hit:
    #   - The contact force is added to the beam's global force vector f.
    #   - The reaction force on the segment generates generalised torques for the
    #     double-pendulum EOM: tau_ext_theta (about neck-head joint) and
    #     tau_ext_phi (about the base).
    #   - When the scalp compliance layer is active, the force on the head goes to
    #     F_contact_on_scalp; torques are computed later via scalp_torques() after
    #     the spring-damper filter is applied.
    #
    # Returns: (f, tau_ext_theta, tau_ext_phi, (base, joint, tip), metrics_dict)
    n              = bp.n_nodes
    f              = np.zeros(3 * n, dtype=float)
    base, joint, tip = segment_endpoints(hp, phi, theta)
    A_neck, B_neck = base, joint
    A_head, B_head = joint, tip
    v_joint        = vel_of_point_on_segment(base, phi_dot, joint)
    tau_ext_theta  = 0.0
    tau_ext_phi    = 0.0
    F_head_total   = np.zeros(2, dtype=float)
    F_contact_on_scalp = np.zeros(2, dtype=float)
    x_abs          = x0 + u
    xs, ys         = x_abs[0::3], x_abs[1::3]
    vxs, vys       = v[0::3], v[1::3]

    # Determine which nodes are live
    node_is_live = np.zeros(n, dtype=bool)
    for e_idx, (ni, nj) in enumerate(elements):
        if element_alive[e_idx]:
            node_is_live[ni] = True
            node_is_live[nj] = True
    node_is_live[0]     = True   # body node always live
    node_is_live[n - 1] = True   # motor node always live

    for i in range(n):
        if not node_is_live[i]:
            continue

        P  = np.array([xs[i],  ys[i]],  dtype=float)
        Vp = np.array([vxs[i], vys[i]], dtype=float)

        # Check head segment first
        Qh, _ = point_segment_closest(P, A_head, B_head)
        seg_vel_head = v_joint + vel_of_point_on_segment(joint, theta_dot, Qh)
        F, F_react, hit, _, Qh = _compute_contact_at_segment(
            P, Vp, A_head, B_head, seg_vel_head, bp, Q_pre=Qh)
        if hit:
            f[3*i]     += F[0]
            f[3*i + 1] += F[1]
            F_head_total += F_react
            if hp.use_scalp_layer:
                # Force goes to the scalp mass — torques computed externally
                # via scalp_accel() in the main loop
                F_contact_on_scalp += F_react
            else:
                # Direct rigid-head model: force creates torques immediately
                r_joint        = Qh - joint
                r_base         = Qh - base
                tau_ext_theta += r_joint[0] * F_react[1] - r_joint[1] * F_react[0]
                tau_ext_phi   += r_base[0]  * F_react[1] - r_base[1]  * F_react[0]
            continue

        # Check neck segment
        Qn, _ = point_segment_closest(P, A_neck, B_neck)
        seg_vel_neck = vel_of_point_on_segment(base, phi_dot, Qn)
        F, F_react, hit, _, Qn = _compute_contact_at_segment(
            P, Vp, A_neck, B_neck, seg_vel_neck, bp, Q_pre=Qn)
        if hit:
            f[3*i]     += F[0]
            f[3*i + 1] += F[1]
            r_base        = Qn - base
            tau_ext_phi  += r_base[0] * F_react[1] - r_base[1] * F_react[0]

    metrics = {
        "F_head_total":        F_head_total,
        "F_contact_on_scalp":  F_contact_on_scalp,
    }
    return f, tau_ext_theta, tau_ext_phi, (base, joint, tip), metrics


def head_com_kinematics(hp, phi, phi_dot, phi_dd, theta, theta_dot, theta_dd):
    # Computes the position and absolute acceleration of the head centre of mass
    # using the chain rule for the two-bar linkage.
    #
    # Joint acceleration (tip of neck bar):
    #   a_joint = L_neck * [-(phi_dd*sin(phi) + phi_dot^2*cos(phi)),
    #                        (phi_dd*cos(phi) - phi_dot^2*sin(phi))]
    #
    # Relative acceleration of head CoM from neck tip (offset r_com along head bar):
    #   a_rel = r_com * [-(theta_dd*sin(theta) + theta_dot^2*cos(theta)),
    #                     (theta_dd*cos(theta) - theta_dot^2*sin(theta))]
    #
    # Returns: (head_com_position, head_com_acceleration)  both as (2,) arrays [m], [m/s^2]
    base  = hp.base
    joint = base + hp.L_neck * np.array([np.cos(phi), np.sin(phi)], dtype=float)
    a_joint = np.array([
        -hp.L_neck * (phi_dd   * np.sin(phi)   + phi_dot**2   * np.cos(phi)),
         hp.L_neck * (phi_dd   * np.cos(phi)   - phi_dot**2   * np.sin(phi)),
    ], dtype=float)
    r        = hp.r_head_com
    head_com = joint + r * np.array([np.cos(theta), np.sin(theta)], dtype=float)
    a_rel    = np.array([
        -r * (theta_dd * np.sin(theta) + theta_dot**2 * np.cos(theta)),
         r * (theta_dd * np.cos(theta) - theta_dot**2 * np.sin(theta)),
    ], dtype=float)
    return head_com, a_joint + a_rel


def scalp_accel(hp, pos_s, vel_s, phi, phi_dot, theta, theta_dot, F_ext_on_scalp):
    # Computes the acceleration of the scalp sub-mass and the filtered force it
    # transmits to the head rigid body.
    #
    # The scalp mass m_scalp moves freely in 2D and is coupled to the nearest
    # point on the head segment by a spring-damper (k_scalp, c_scalp).
    # The contact force from the drone acts directly on the scalp mass.
    # Only the filtered spring-damper reaction reaches the head CoM, which
    # low-pass filters the impact pulse and reduces peak head acceleration.
    #
    # EOM of scalp mass:
    #   m_scalp * a_s = F_ext_on_scalp - F_spring
    #   F_spring = k_scalp * (pos_s - Q_h) + c_scalp * (vel_s - vel_Q_h)
    #
    # Returns: (a_s, F_on_head, Q_h)
    #   a_s       : scalp mass acceleration [m/s^2]
    #   F_on_head : spring-damper reaction on head rigid body [N]  (Newton's 3rd law)
    #   Q_h       : attachment point on head segment [m]
    base, joint, tip = segment_endpoints(hp, phi, theta)
    # Nearest point on head segment to current scalp mass position
    Q_h, _ = point_segment_closest(pos_s, joint, tip)
    # Velocity of that material point on the rotating head segment
    vel_joint = vel_of_point_on_segment(base, phi_dot, joint)
    vel_Q_h   = vel_joint + vel_of_point_on_segment(joint, theta_dot, Q_h)
    # Spring-damper: pulls scalp back toward the head surface
    disp      = pos_s - Q_h
    dvel      = vel_s - vel_Q_h
    F_spring  = hp.k_scalp * disp + hp.c_scalp * dvel
    # Scalp EOM: external contact force minus spring pulling it back
    a_s       = (F_ext_on_scalp - F_spring) / hp.m_scalp
    # Reaction on head (Newton's 3rd law): equal and opposite to F_spring on scalp
    F_on_head = F_spring
    return a_s, F_on_head, Q_h


def scalp_torques(hp, phi, theta, F_on_head, Q_h):
    # Converts the scalp-to-head spring-damper force F_on_head, applied at Q_h
    # on the head segment, into generalised torques for both pendulum equations.
    # tau_theta: torque about the neck-head joint (moment arm = Q_h - joint)
    # tau_phi  : torque about the base           (moment arm = Q_h - base)
    # Both are computed as the 2D cross product r x F = rx*Fy - ry*Fx.
    base, joint, _ = segment_endpoints(hp, phi, theta)
    r_joint    = Q_h - joint
    r_base     = Q_h - base
    tau_theta  = r_joint[0] * F_on_head[1] - r_joint[1] * F_on_head[0]
    tau_phi    = r_base[0]  * F_on_head[1] - r_base[1]  * F_on_head[0]
    return tau_theta, tau_phi


def compute_hic(accel_g, dt, window=0.015):
    # Computes the Head Injury Criterion (HIC) over a rolling time window.
    # HIC = max over all sub-intervals [t1, t2] with (t2-t1) <= window of:
    #   HIC = (t2 - t1) * (mean_a_g over [t1,t2])^2.5
    # where mean_a_g is the average resultant acceleration in g's.
    #
    # Implementation uses a prefix-sum array for O(1) interval-average queries,
    # making the overall complexity O(n * max_n) where max_n = window/dt.
    #
    # Returns: (hic_value, (i0, i1)) where i0, i1 are the time-step indices
    #          of the worst-case interval.
    a     = np.asarray(accel_g, dtype=float)
    n     = len(a)
    if n < 2:
        return 0.0, (0, 0)
    max_n  = max(1, int(np.floor(window / dt)))
    prefix = np.empty(n + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(a, out=prefix[1:])
    max_hic  = 0.0
    max_pair = (0, 0)
    for i in range(n - 1):
        j_lo = i + 1
        j_hi = min(n - 1, i + max_n)
        if j_lo > j_hi:
            continue
        js       = np.arange(j_lo, j_hi + 1)
        lens     = js - i
        sums     = prefix[js + 1] - prefix[i]
        avgs     = sums / lens
        durs     = lens * dt
        hic_vals = durs * np.power(np.maximum(avgs, 0.0), 2.5)
        best     = int(np.argmax(hic_vals))
        if hic_vals[best] > max_hic:
            max_hic  = float(hic_vals[best])
            max_pair = (i, int(js[best]))
    return max_hic, max_pair


def compute_bric_planar(theta_dot, hp):
    # Computes the Brain Rotational Injury Criterion (BrIC) in the sagittal plane.
    # BrIC = omega_peak / omega_yC
    # where omega_yC = 56.45 rad/s is the critical angular velocity for
    # flexion/extension from Takhounts et al. (2011).
    # Returns: (bric_value, omega_peak)
    omega_peak = float(np.max(np.abs(theta_dot)))
    return abs(omega_peak / hp.omega_yC), omega_peak


def compute_nij(axial_force, bending_moment, hp):
    # Computes the neck injury index Nij at every time step.
    # Nij = Fz/Fint + My/Mint
    # where Fint and Mint are intercept values that depend on the sign of the load:
    #   Fint = Fint_tension    if Fz >= 0 (tension)
    #   Fint = Fint_compression if Fz < 0 (compression)
    #   Mint = Mint_flexion    if My >= 0 (flexion)
    #   Mint = Mint_extension  if My < 0  (extension)
    # Returns: (nij_array, nij_peak)
    axial_force    = np.asarray(axial_force,    dtype=float)
    bending_moment = np.asarray(bending_moment, dtype=float)
    nij = np.zeros_like(axial_force)
    for i, (Fz, My) in enumerate(zip(axial_force, bending_moment)):
        Fint   = hp.Fint_tension  if Fz >= 0.0 else hp.Fint_compression
        Mint   = hp.Mint_flexion  if My >= 0.0 else hp.Mint_extension
        nij[i] = (Fz / Fint) + (My / Mint)
    return nij, float(np.max(np.abs(nij)))


def ais3p_from_hic15(hic15):
    # Returns AIS 3+ injury probability from HIC15 using the published logistic fit.
    return 1.0 / (1.0 + np.exp((3.39 + 140.0 / max(hic15, 1e-12)) - 0.00531 * hic15))

def ais3p_from_bric(bric):
    # Returns AIS 3+ injury probability from BrIC using a Weibull CDF fit.
    return 1.0 - np.exp(-(bric / 0.987) ** 2.84)

def ais3p_from_nij(nij):
    # Returns AIS 3+ injury probability from Nij using a logistic fit.
    return 1.0 / (1.0 + np.exp(3.227 - 1.969 * nij))


def print_injury_report(results, hp):
    # Prints a summary of the computed injury metrics and AIS 3+ probabilities
    # to standard output. Thresholds (HIC_limit, Nij_limit) from HeadNeckParams.
    ir = results["injury_report"]
    print("\n[Injury report]")
    print(f"  HIC15 = {ir['HIC15']:.3f}  (limit {hp.HIC_limit:.1f})")
    print(f"  BrIC  = {ir['BrIC']:.3f}  [sagittal: omega_yC = {hp.omega_yC:.2f} rad/s]")
    print(f"  Nij   = {ir['Nij_peak']:.3f}  (limit {hp.Nij_limit:.1f})")
    print(f"  AIS3+ from HIC15 = {ir['AIS3p_HIC15']:.3f}")
    print(f"  AIS3+ from BrIC  = {ir['AIS3p_BrIC']:.3f}")
    print(f"  AIS3+ from Nij   = {ir['AIS3p_Nij']:.3f}")


def plot_head_acceleration(results, hp):
    # Plots the head CoM resultant acceleration (in g's) over the full simulation time.
    # Highlights the worst-case HIC15 interval as a shaded region, annotates the
    # peak acceleration, and exports the full curve to a MATLAB .mat file
    # (head_acceleration.mat) in the same directory as this script.
    t_ms    = results["t_hist"] * 1e3          # time in milliseconds
    accel_g = results["accel_g"]               # resultant acceleration [g]

    ir      = results["injury_report"]
    hic15   = ir["HIC15"]
    i0, i1  = ir["HIC15_window_idx"]
    t0_ms   = t_ms[i0]
    t1_ms   = t_ms[i1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_ms, accel_g, color='steelblue', linewidth=1.2, label='Head CoM acceleration')

    # Shade the HIC-15 window
    ax.axvspan(t0_ms, t1_ms, alpha=0.18, color='tomato',
               label=f'HIC15 window  [{t0_ms:.2f} – {t1_ms:.2f} ms]')

    # Annotate peak
    peak_g   = float(accel_g.max())
    peak_idx = int(accel_g.argmax())
    ax.annotate(f'peak = {peak_g:.1f} g',
                xy=(t_ms[peak_idx], peak_g),
                xytext=(t_ms[peak_idx] + 0.3, peak_g * 0.88),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9, color='black')

    ax.set_xlabel('Time  [ms]')
    ax.set_ylabel('Resultant acceleration  [g]')
    ax.set_title(
        f'Head CoM acceleration — HIC15 = {hic15:.1f}  '
        f'(AIS3+ = {ir["AIS3p_HIC15"]:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()

    # ── Export acceleration curve to MATLAB ───────────────────────────────────
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _mat_path   = os.path.join(_script_dir, "head_acceleration.mat")
    scipy.io.savemat(_mat_path, {
        "t_s":     results["t_hist"],   # time vector [s]
        "t_ms":    t_ms,                # time vector [ms]
        "accel_g": accel_g,             # resultant head CoM acceleration [g]
        "HIC15":   hic15,               # scalar HIC15 value
    })
    print(f"[Export] head_acceleration.mat saved to: {_mat_path}")
    # ─────────────────────────────────────────────────────────────────────────

    plt.show()


# =============================================================================
# Main simulation loop
# =============================================================================

def simulate(bp, hp, sp, do_animation=True):
    # Master time-integration function. Couples the FEM beam subsystem with the
    # double-pendulum head-neck subsystem using the Newmark-beta method
    # (beta=0.25, gamma=0.5 — unconditionally stable constant-average-acceleration).
    #
    # Initialisation:
    #   - Mesh and element connectivity.
    #   - Initial node positions along the beam axis at the given angle.
    #   - Initial nodal velocities (horizontal or angled, controlled by velocity_angled).
    #   - Lumped mass matrix M (assembled once; updated only after fracture).
    #   - Auto-computed contact stiffness, damping, time step, incubation steps,
    #     and Rayleigh coefficients from the user-specified target values.
    #   - Scalp sub-mass placed at the midpoint of the head segment.
    #
    # Per-step sequence (Newmark predictor-corrector):
    #   1. Record history (positions, head-neck angles, plastic state).
    #   2. Predictor: extrapolate u, v, phi, theta, scalp using (0.5-beta) and (1-gamma).
    #   3. Evaluate contact forces and head-neck EOM at the predicted state.
    #   4. Corrector: implicit solve (M + gamma*dt*C + beta*dt^2*K) * a_new = RHS.
    #   5. Plasticity return mapping (before damage, so damage sees elastic overload).
    #   6. Damage/fracture check; if elements fracture, re-assemble K, M and re-seed a.
    #
    # Post-loop:
    #   - Compute HIC15, BrIC, Nij and AIS3+ probabilities.
    #   - Package all history arrays and injury report into the results dict.
    #   - Optionally run the animation.
    #
    # Returns: results dict with keys:
    #   u_hist, elements, seg_hist, fractures, frame_stride, dt,
    #   t_hist, accel_g, theta/phi/dot/dd histories,
    #   head_accel_res_hist, axial_force_hist, bending_moment_hist,
    #   injury_report, eps_p/kappa_p/alpha_p final and hist arrays.
    hp.update_inertias()
    dt = sp.dt
    _, elements          = generate_mesh(bp.n_nodes, bp.L0)
    element_alive        = np.ones(len(elements),  dtype=bool)
    element_damage       = np.zeros(len(elements), dtype=float)
    over_limit_counter   = np.zeros(len(elements), dtype=int)
    # ── Plasticity state (one value per element) ──────────────────────────────
    # eps_p    : accumulated plastic engineering axial strain (dimensionless)
    # kappa_p  : accumulated plastic curvature [rad/m]
    # alpha_p  : accumulated plastic multiplier (drives isotropic hardening)
    # All initialised to zero (fully virgin material at t = 0).
    n_elems   = len(elements)
    eps_p     = np.zeros(n_elems, dtype=float)
    kappa_p   = np.zeros(n_elems, dtype=float)
    alpha_p   = np.zeros(n_elems, dtype=float)
    # History snapshots recorded every frame_stride steps (same cadence as u_hist)
    eps_p_hist   = []
    kappa_p_hist = []
    alpha_p_hist = []
    # ─────────────────────────────────────────────────────────────────────────
    x_nodes0, y_nodes0  = compute_initial_node_positions(bp)
    ndof = 3 * bp.n_nodes
    x0   = np.zeros(ndof, dtype=float)
    for i in range(bp.n_nodes):
        x0[3*i]     = x_nodes0[i]
        x0[3*i + 1] = y_nodes0[i]
    u = np.zeros(ndof, dtype=float)
    v = np.zeros(ndof, dtype=float)
    ang = np.deg2rad(bp.initial_angle_deg)
    if bp.velocity_angled:
        # Velocity directed along the beam axis (same angle as the arm)
        v[0::3] = bp.v0 * np.cos(ang)
        v[1::3] = bp.v0 * np.sin(ang)
    else:
        # Velocity purely horizontal (0°) regardless of the beam angle
        v[0::3] = bp.v0
        v[1::3] = 0.0
    M       = assemble_lumped_mass(bp, elements)
    _, M    = stabilize_components(np.zeros_like(M), M, bp, elements, element_alive)
    tip_dof_x = 3 * (bp.n_nodes - 1)
    m_eff     = M[tip_dof_x, tip_dof_x]
    # Total drone mass = body + motor + distributed beam
    L_e_mass  = element_length(bp.L0, bp.n_nodes)
    m_beam    = bp.rho * bp.A * bp.L0
    m_total   = bp.m_body + bp.m_motor + m_beam
    # ── Auto k_contact from penetration target (energy balance) ──────────────
    if bp.use_auto_k_contact:
        bp.k_contact, k_lin = compute_hertz_k_from_penetration(bp, m_total)
    else:
        n_exp  = bp.hertz_exponent if bp.use_hertz else 1.0
        k_lin  = n_exp * bp.k_contact * bp.delta_max_target**(n_exp - 1.0)
    # ── Auto c_contact and dt (linearised at delta_max_target) ───────────────
    if bp.use_auto_contact:
        bp.c_contact, zeta_contact = compute_contact_damping_from_restitution(
            k_lin, m_eff, bp.e_target)
        tc, dt_new = compute_contact_time_and_dt(
            k_lin, m_eff, zeta_contact, bp.dt_ratio_contact)
        sp.dt = dt_new
        dt    = sp.dt
    else:
        _, zeta_contact = compute_contact_damping_from_restitution(
            k_lin, m_eff, bp.e_target)
        tc, _ = compute_contact_time_and_dt(
            k_lin, m_eff, zeta_contact, bp.dt_ratio_contact)
        dt = sp.dt
    if bp.use_auto_incubation:
        bp.incubation_steps = max(1, int(np.ceil(bp.t_inc / dt)))
    if bp.use_auto_rayleigh:
        bp.rayleigh_alpha, bp.rayleigh_beta = compute_rayleigh_from_two_freqs(
            bp.zeta_target, bp.f_rayleigh_1, bp.f_rayleigh_2)
    print_parameter_report(bp, hp, sp, m_eff, m_total, zeta_contact, tc)
    # ── Slenderness check (Euler-Bernoulli validity) ──────────────────────────
    L_e_check = element_length(bp.L0, bp.n_nodes)
    slenderness = L_e_check / bp.h
    print(f"  Element slenderness L_e/h = {slenderness:.4f}  "
          f"(L_e = {L_e_check*1e3:.3f} mm,  h = {bp.h*1e3:.1f} mm)")
    if slenderness < 1.0:
        print("  *** WARNING: L_e/h < 1 — element is DEEPER than it is long. ***")
        print("  *** Euler-Bernoulli beam theory is NOT valid at this slenderness. ***")
        print("  *** Consider reducing n_nodes or the cross-section height h.    ***")
    elif slenderness < 10.0:
        print("  NOTE: L_e/h < 10 — shear deformation effects are non-negligible;")
        print("        Timoshenko theory would be more accurate at this slenderness.")
    # ─────────────────────────────────────────────────────────────────────────
    steps = int(sp.T_end / dt)
    # Auto-adjust frame_stride so the animation never exceeds MAX_ANIM_FRAMES
    # frames.  With a fine dt (e.g. 1e-7) the default stride=30 would produce
    # ~6 000+ frames, causing FuncAnimation to take an extremely long time to
    # initialise.  We raise the stride as needed while always keeping the
    # user-specified stride as a lower bound (never reduce quality for coarse dt).
    MAX_ANIM_FRAMES = 500
    sp.frame_stride = max(sp.frame_stride, math.ceil(steps / MAX_ANIM_FRAMES))
    print(f"  frame_stride   = {sp.frame_stride}  (~{steps // sp.frame_stride} animation frames)")
    phi, theta         = hp.phi0, hp.theta0
    phi_dot = theta_dot = 0.0
    phi_dd  = theta_dd  = 0.0
    # ── Scalp mass initial state ──────────────────────────────────────────────
    # Place the scalp mass at the midpoint of the head segment at rest.
    if hp.use_scalp_layer:
        hp.update_inertias()   # ensure c_scalp is current
        _base0, _joint0, _tip0 = segment_endpoints(hp, hp.phi0, hp.theta0)
        pos_s = 0.5 * (_joint0 + _tip0)   # midpoint of head segment
        vel_s = np.zeros(2, dtype=float)
        acc_s = np.zeros(2, dtype=float)
    else:
        pos_s = vel_s = acc_s = None
    beta  = 0.25
    gamma = 0.5
    K    = assemble_K_alive_current_angle(bp, elements, element_alive, element_damage, x0, u)
    K, _ = stabilize_components(K, np.zeros_like(M), bp, elements, element_alive)
    f_int = internal_forces_alive_current_angle(
        bp, elements, element_alive, element_damage, x0, u, eps_p, kappa_p)
    f_contact, tau_theta, tau_phi, segs, contact_metrics = contact_forces_beam_vs_segments(
        bp, hp, x0, u, v, phi, phi_dot, theta, theta_dot, element_alive, elements)
    if hp.use_scalp_layer:
        acc_s, F_on_head_0, Q_h_0 = scalp_accel(
            hp, pos_s, vel_s, phi, phi_dot, theta, theta_dot,
            contact_metrics["F_contact_on_scalp"])
        dt_s, dt_p = scalp_torques(hp, phi, theta, F_on_head_0, Q_h_0)
        tau_theta += dt_s;  tau_phi += dt_p
    C = bp.rayleigh_alpha * M + bp.rayleigh_beta * K
    a = la.solve(M, f_contact - f_int - (C @ v), assume_a='sym')
    theta_dd, phi_dd, My_OC = headneck_accel(
        hp, 0.0, theta, theta_dot, phi, phi_dot, tau_theta, tau_phi)
    xabs_hist, seg_hist, fractures       = [], [], []
    t_hist                               = []
    theta_hist, theta_dot_hist, theta_dd_hist = [], [], []
    phi_hist,   phi_dot_hist,   phi_dd_hist   = [], [], []
    head_accel_res_hist                               = []
    axial_force_hist, bending_moment_hist             = [], []
    _print_every = max(1, steps // 20)   # print ~20 progress lines over the full run
    for step in range(steps + 1):
        t = step * dt
        if step % _print_every == 0 or step == steps:
            pct = 100.0 * t / sp.T_end
            print(f"  Progress: {pct:5.1f}%  (t = {t*1e3:.3f} ms / {sp.T_end*1e3:.3f} ms)")
        t_hist.append(t)
        theta_hist.append(theta);     theta_dot_hist.append(theta_dot)
        theta_dd_hist.append(theta_dd)
        phi_hist.append(phi);         phi_dot_hist.append(phi_dot)
        phi_dd_hist.append(phi_dd)
        _, a_head = head_com_kinematics(
            hp, phi, phi_dot, phi_dd, theta, theta_dot, theta_dd)
        head_accel_res_hist.append(float(np.linalg.norm(a_head)))

        # ── Nij: axial force and bending moment at the occipital condyles ────
        # Axial force from Newton's 2nd on the head:
        #   F_oc = m_head * a_head - F_contact_on_head  projected onto neck axis
        # Bending moment = internal neck spring-damper torque My_OC
        base, joint, _ = segment_endpoints(hp, phi, theta)
        neck_axis      = joint - base
        neck_axis_unit = neck_axis / (np.linalg.norm(neck_axis) + 1e-12)
        F_inertial_head = hp.m_head * a_head
        # When the scalp layer is on, the force reaching the head CoM is the
        # spring-damper reaction F_on_head, not the direct contact force.
        if hp.use_scalp_layer:
            _, F_on_head_rec, Q_h_rec = scalp_accel(
                hp, pos_s, vel_s, phi, phi_dot, theta, theta_dot,
                contact_metrics["F_contact_on_scalp"])
            F_oc = -(F_on_head_rec - F_inertial_head)
        else:
            F_oc = -(contact_metrics["F_head_total"] - F_inertial_head)
        axial_force_hist.append(float(np.dot(-F_oc, neck_axis_unit)))
        # My_OC = tau_upper_on_head + tau_mus, computed inside headneck_accel
        bending_moment_hist.append(float(My_OC))

        if step % sp.frame_stride == 0:
            xabs_hist.append((x0 + u).copy())  # absolute positions — immune to x0 mutation
            seg_hist.append(segs)
            eps_p_hist.append(eps_p.copy())
            kappa_p_hist.append(kappa_p.copy())
            alpha_p_hist.append(alpha_p.copy())
        if step == steps:
            break
        phi_pred     = phi     + dt * phi_dot     + (0.5 - beta) * dt**2 * phi_dd
        phi_dot_pred = phi_dot + (1.0 - gamma) * dt * phi_dd
        theta_pred     = theta     + dt * theta_dot     + (0.5 - beta) * dt**2 * theta_dd
        theta_dot_pred = theta_dot + (1.0 - gamma) * dt * theta_dd
        u_pred = u + dt * v + (0.5 - beta) * dt**2 * a
        v_pred = v + (1.0 - gamma) * dt * a
        # Scalp predictor
        if hp.use_scalp_layer:
            pos_s_pred = pos_s + dt * vel_s + (0.5 - beta) * dt**2 * acc_s
            vel_s_pred = vel_s + (1.0 - gamma) * dt * acc_s
        f_contact_pred, tau_theta_ext, tau_phi_ext, segs, contact_metrics = \
            contact_forces_beam_vs_segments(
                bp, hp, x0, u_pred, v_pred,
                phi_pred, phi_dot_pred, theta_pred, theta_dot_pred,
                element_alive, elements)
        # Add scalp-to-head torques at the predicted state
        if hp.use_scalp_layer:
            acc_s_new, F_on_head_pred, Q_h_pred = scalp_accel(
                hp, pos_s_pred, vel_s_pred,
                phi_pred, phi_dot_pred, theta_pred, theta_dot_pred,
                contact_metrics["F_contact_on_scalp"])
            dt_s, dt_p = scalp_torques(hp, phi_pred, theta_pred, F_on_head_pred, Q_h_pred)
            tau_theta_ext += dt_s;  tau_phi_ext += dt_p
        # Evaluate head-neck accelerations at predicted state (t+dt)
        theta_dd_new, phi_dd_new, _ = headneck_accel(
            hp, t + dt, theta_pred, theta_dot_pred, phi_pred, phi_dot_pred,
            tau_theta_ext, tau_phi_ext)
        theta     = theta_pred     + beta  * dt**2 * theta_dd_new
        theta_dot = theta_dot_pred + gamma * dt    * theta_dd_new
        phi       = phi_pred       + beta  * dt**2 * phi_dd_new
        phi_dot   = phi_dot_pred   + gamma * dt    * phi_dd_new
        theta_dd  = theta_dd_new
        phi_dd    = phi_dd_new
        # Recompute My_OC at the corrected state so the next recording step
        # uses the neck moment consistent with the updated angles/velocities.
        _, _, My_OC = headneck_accel(
            hp, t + dt, theta, theta_dot, phi, phi_dot,
            tau_theta_ext, tau_phi_ext)
        # Refresh segs to the corrected head-neck geometry
        segs = segment_endpoints(hp, phi, theta)
        # Scalp corrector
        if hp.use_scalp_layer:
            pos_s = pos_s_pred + beta  * dt**2 * acc_s_new
            vel_s = vel_s_pred + gamma * dt    * acc_s_new
            acc_s = acc_s_new
        K_pred    = assemble_K_alive_current_angle(
            bp, elements, element_alive, element_damage, x0, u_pred)
        K_pred, _ = stabilize_components(K_pred, np.zeros_like(M), bp, elements, element_alive)
        f_int_pred = internal_forces_alive_current_angle(
            bp, elements, element_alive, element_damage, x0, u_pred, eps_p, kappa_p)
        C_pred = bp.rayleigh_alpha * M + bp.rayleigh_beta * K_pred
        A_eff  = (M + gamma * dt * C_pred + beta * dt**2 * K_pred
                  + 1e-12 * np.eye(M.shape[0]))
        a_new  = la.solve(A_eff, f_contact_pred - f_int_pred - (C_pred @ v_pred),
                          assume_a='sym')
        u = u_pred + beta  * dt**2 * a_new
        v = v_pred + gamma * dt    * a_new
        a = a_new
        # Plasticity return mapping runs before damage so that damage sees
        # the post-plasticity elastic stress.
        if bp.use_plasticity:
            plastic_return_mapping(bp, x0, u, elements, element_alive,
                                   element_damage, eps_p, kappa_p, alpha_p)
        newly, _ = update_damage_and_failure(
            bp, x0, u, elements, element_alive, element_damage,
            over_limit_counter, dt, eps_p, kappa_p)
        if newly:
            fractures.append((step, newly))
            K_now    = assemble_K_alive_current_angle(
                bp, elements, element_alive, element_damage, x0, u)
            K_now, M = stabilize_components(K_now, M, bp, elements, element_alive)
            f_int_now = internal_forces_alive_current_angle(
                bp, elements, element_alive, element_damage, x0, u, eps_p, kappa_p)
            f_contact_now, _, _, segs, contact_metrics = contact_forces_beam_vs_segments(
                bp, hp, x0, u, v, phi, phi_dot, theta, theta_dot,
                element_alive, elements)
            C_now = bp.rayleigh_alpha * M + bp.rayleigh_beta * K_now
            # Re-seed acceleration with implicit solve after fracture
            A_eff_now = (M + gamma * dt * C_now + beta * dt**2 * K_now
                         + 1e-12 * np.eye(M.shape[0]))
            a = la.solve(A_eff_now,
                         f_contact_now - f_int_now - (C_now @ v),
                         assume_a='sym')
            # Re-seed scalp acceleration after fracture
            if hp.use_scalp_layer:
                acc_s, _, _ = scalp_accel(
                    hp, pos_s, vel_s, phi, phi_dot, theta, theta_dot,
                    contact_metrics["F_contact_on_scalp"])
    t_hist              = np.array(t_hist)
    theta_hist          = np.array(theta_hist)
    theta_dot_hist      = np.array(theta_dot_hist)
    theta_dd_hist       = np.array(theta_dd_hist)
    phi_hist            = np.array(phi_hist)
    phi_dot_hist        = np.array(phi_dot_hist)
    phi_dd_hist         = np.array(phi_dd_hist)
    head_accel_res_hist = np.array(head_accel_res_hist)
    axial_force_hist    = np.array(axial_force_hist)
    bending_moment_hist = np.array(bending_moment_hist)
    accel_g            = head_accel_res_hist / 9.81
    hic15, hic_pair    = compute_hic(accel_g, dt, hp.HIC_window)
    bric, omega_peak   = compute_bric_planar(theta_dot_hist, hp)
    _, nij_peak = compute_nij(axial_force_hist, bending_moment_hist, hp)
    injury_report = {
        "HIC15":            hic15,
        "HIC15_window_idx": hic_pair,
        "BrIC":             bric,
        "Nij_peak":         nij_peak,
        "AIS3p_HIC15":      ais3p_from_hic15(hic15),
        "AIS3p_BrIC":       ais3p_from_bric(bric),
        "AIS3p_Nij":        ais3p_from_nij(nij_peak),
    }
    results = {
        "u_hist":              np.array(xabs_hist),
        "elements":            elements,
        "seg_hist":            seg_hist,
        "fractures":           fractures,
        "frame_stride":        sp.frame_stride,
        "dt":                  sp.dt,
        "t_hist":              t_hist,
        "accel_g":             accel_g,
        "theta_hist":          theta_hist,
        "theta_dot_hist":      theta_dot_hist,
        "theta_dd_hist":       theta_dd_hist,
        "phi_hist":            phi_hist,
        "phi_dot_hist":        phi_dot_hist,
        "phi_dd_hist":         phi_dd_hist,
        "head_accel_res_hist": head_accel_res_hist,
        "axial_force_hist":    axial_force_hist,
        "bending_moment_hist": bending_moment_hist,
        "injury_report":       injury_report,
        "eps_p_final":         eps_p.copy(),
        "kappa_p_final":       kappa_p.copy(),
        "alpha_p_final":       alpha_p.copy(),
        "eps_p_hist":          eps_p_hist,
        "kappa_p_hist":        kappa_p_hist,
        "alpha_p_hist":        alpha_p_hist,
    }
    if do_animation:
        results["ani"] = animate(bp, sp, results)
    return results


# =============================================================================
# Visualisation helpers
# =============================================================================

def _rect_corners(cx, cy, half_len, half_wid, angle_rad):
    # Returns the four corner coordinates (as a (4,2) array) of a rectangle
    # centred at (cx, cy), with:
    #   half_len : half-length along the element axis direction angle_rad
    #   half_wid : half-width perpendicular to the axis
    # The rectangle is axis-aligned to the element chord direction, so it
    # rotates with the deformed beam in the animation.
    ca, sa  = np.cos(angle_rad), np.sin(angle_rad)
    ax      = np.array([ ca,  sa])   # unit vector along axis
    px      = np.array([-sa,  ca])   # unit vector perpendicular
    c       = np.array([cx, cy])
    return np.array([
        c + half_len * ax + half_wid * px,
        c + half_len * ax - half_wid * px,
        c - half_len * ax - half_wid * px,
        c - half_len * ax + half_wid * px,
    ])


def _arm_rect_corners(xi, yi, xj, yj):
    # Returns the four corners of a beam element rectangle whose long axis
    # exactly spans from node i (xi, yi) to node j (xj, yj).
    # The width is ARM_HALF_W on each side of the chord.
    dx      = xj - xi
    dy      = yj - yi
    L       = np.hypot(dx, dy)
    if L < 1e-12:
        ang = 0.0
    else:
        ang = np.arctan2(dy, dx)
    cx, cy  = 0.5 * (xi + xj), 0.5 * (yi + yj)
    return _rect_corners(cx, cy, L / 2.0, ARM_HALF_W, ang)


def _body_rect_corners(x0_node, y0_node, last_ang):
    # Returns the four corners of the body (drone chassis) square centred on
    # node 0, oriented along the first element's chord direction so it looks
    # naturally attached to the arm.
    return _rect_corners(x0_node, y0_node, BODY_HALF, BODY_HALF, last_ang)


def _motor_rect_corners(xm, ym, arm_ang):
    # Returns the four corners of the motor rectangle centred on the last node.
    # The motor's long axis is perpendicular to the arm (it is mounted across
    # the tip), so we rotate the arm angle by 90 degrees.
    perp_ang = arm_ang + np.pi / 2.0   # rotate 90 deg -> perpendicular to arm
    return _rect_corners(xm, ym, MOTOR_HALF_L, MOTOR_HALF_W, perp_ang)



# =============================================================================
# Animation
# =============================================================================

def animate(bp, sp, results):
    # Creates a matplotlib FuncAnimation showing the drone arm impact and the
    # head-neck pendulum motion side by side in the same axes.
    #
    # Visual elements updated each frame:
    #   - Arm element patches (blue rectangles): updated via _arm_rect_corners.
    #   - Broken element stubs (red half-rectangles): revealed at fracture frames;
    #     each broken element becomes two independent stubs following their nodes.
    #   - Body patch (grey square at node 0): follows node 0, oriented along element 0.
    #   - Motor patch (orange rectangle at last node): follows the tip node, oriented
    #     perpendicular to the last element.
    #   - Neck and head lines: drawn from the stored seg_hist snapshots.
    #   - Time label (top-left): shows current simulation time in milliseconds.
    #
    # Frame stride is auto-capped to at most MAX_ANIM_FRAMES=500 frames so that
    # FuncAnimation initialisation does not take excessively long for fine dt.
    u_hist       = results["u_hist"]
    elements     = results["elements"]
    seg_hist     = results["seg_hist"]
    fractures    = results["fractures"]
    frame_stride = results["frame_stride"]
    n_nodes      = bp.n_nodes
    n_elems      = len(elements)

    # Map each fracture step to the first animation frame at or after that step
    broken_at_frame = {}
    for st, elems in fractures:
        frame_idx = math.ceil(st / frame_stride)
        broken_at_frame.setdefault(frame_idx, []).extend(elems)

    alive = np.ones(n_elems, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Drone arm impact — FEM mesh visualisation")
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.05, 0.35)

    def _initial_node_xy(node_idx):
        return u_hist[0][3*node_idx], u_hist[0][3*node_idx + 1]

    # Compute initial element axis angle (needed to orient body & motor)
    x_i0, y_i0 = _initial_node_xy(0)
    x_j0, y_j0 = _initial_node_xy(1)
    first_ang0  = np.arctan2(y_j0 - y_i0, x_j0 - x_i0)

    x_last0, y_last0  = _initial_node_xy(n_nodes - 1)
    x_prev0, y_prev0  = _initial_node_xy(n_nodes - 2)
    last_ang0          = np.arctan2(y_last0 - y_prev0, x_last0 - x_prev0)

    # Arm element patches (all elements)
    arm_patches = []
    for e_idx, (i, j) in enumerate(elements):
        xi, yi = _initial_node_xy(i)
        xj, yj = _initial_node_xy(j)
        corners = _arm_rect_corners(xi, yi, xj, yj)
        patch   = Polygon(corners, closed=True,
                          facecolor=COL_ARM_FACE, edgecolor=COL_ARM_EDGE,
                          linewidth=0.8, zorder=2)
        ax.add_patch(patch)
        arm_patches.append(patch)

    # Pre-create two half-stub patches per element (invisible until fracture)
    _dummy = np.zeros((4, 2))
    stub_patches = []   # list of (patch_i, patch_j) tuples
    for e_idx in range(n_elems):
        pi = Polygon(_dummy.copy(), closed=True,
                     facecolor=COL_BROKEN_F, edgecolor=COL_BROKEN_E,
                     linewidth=0.8, alpha=0.75, zorder=3, visible=False)
        pj = Polygon(_dummy.copy(), closed=True,
                     facecolor=COL_BROKEN_F, edgecolor=COL_BROKEN_E,
                     linewidth=0.8, alpha=0.75, zorder=3, visible=False)
        ax.add_patch(pi)
        ax.add_patch(pj)
        stub_patches.append((pi, pj))

    # Body patch (node 0)
    body_patch = Polygon(
        _body_rect_corners(x_i0, y_i0, first_ang0),
        closed=True,
        facecolor=COL_BODY_FACE, edgecolor=COL_BODY_EDGE,
        linewidth=1.2, zorder=4)
    ax.add_patch(body_patch)

    # Motor patch (last node)
    motor_patch = Polygon(
        _motor_rect_corners(x_last0, y_last0, last_ang0),
        closed=True,
        facecolor=COL_MOTOR_FACE, edgecolor=COL_MOTOR_EDGE,
        linewidth=1.2, zorder=4)
    ax.add_patch(motor_patch)

    # Head-neck lines (drawn on top)
    neck_line, = ax.plot([], [], '-', lw=4, color=COL_NECK,  zorder=5, solid_capstyle='round')
    head_line, = ax.plot([], [], '-', lw=4, color=COL_HEAD,  zorder=5, solid_capstyle='round')

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=COL_BODY_FACE,  edgecolor=COL_BODY_EDGE,  label='Body (node 0)'),
        Patch(facecolor=COL_ARM_FACE,   edgecolor=COL_ARM_EDGE,   label='Arm elements'),
        Patch(facecolor=COL_MOTOR_FACE, edgecolor=COL_MOTOR_EDGE, label='Motor (tip node)'),
        Patch(facecolor=COL_BROKEN_F,   edgecolor=COL_BROKEN_E,   label='Broken element'),
        Line2D([0], [0], color=COL_NECK, lw=3, label='Neck'),
        Line2D([0], [0], color=COL_HEAD, lw=3, label='Head'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.85)

    # Time annotation
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top', family='monospace')

    # ── init / update callbacks ───────────────────────────────────────────────
    def _node_xy(x_abs, node_idx):
        return x_abs[3*node_idx], x_abs[3*node_idx + 1]

    all_stubs = [p for pair in stub_patches for p in pair]

    def init():
        neck_line.set_data([], [])
        head_line.set_data([], [])
        time_text.set_text('')
        return arm_patches + all_stubs + [body_patch, motor_patch, neck_line, head_line, time_text]

    def _stub_corners(xn, yn, xother, yother):
        """Half-length stub rectangle anchored at (xn, yn), pointing toward (xother, yother)."""
        dx = xother - xn
        dy = yother - yn
        L  = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx) if L > 1e-12 else 0.0
        half_len = min(L / 2.0, ARM_HALF_W * 3)   # cap so tiny fragments don't disappear
        ca, sa = np.cos(ang), np.sin(ang)
        ax_  = np.array([ca,  sa])
        px_  = np.array([-sa, ca])
        c    = np.array([xn, yn]) + half_len * ax_
        return np.array([
            c + half_len * ax_ + ARM_HALF_W * px_,
            c + half_len * ax_ - ARM_HALF_W * px_,
            c - half_len * ax_ - ARM_HALF_W * px_,
            c - half_len * ax_ + ARM_HALF_W * px_,
        ])

    def update(frame):
        # Mark newly broken elements: hide spanning patch, reveal two stubs
        if frame in broken_at_frame:
            for e_idx in broken_at_frame[frame]:
                alive[e_idx] = False
                arm_patches[e_idx].set_visible(False)   # hide the connector
                stub_patches[e_idx][0].set_visible(True)
                stub_patches[e_idx][1].set_visible(True)

        x_abs = u_hist[frame]  # stored as absolute positions

        # Update arm element rectangles (alive only) and stubs (broken only)
        for e_idx, (i, j) in enumerate(elements):
            xi, yi = _node_xy(x_abs, i)
            xj, yj = _node_xy(x_abs, j)
            if alive[e_idx]:
                arm_patches[e_idx].set_xy(_arm_rect_corners(xi, yi, xj, yj))
            else:
                # Move each stub to follow its own node, oriented away from the other
                stub_patches[e_idx][0].set_xy(_stub_corners(xi, yi, xj, yj))
                stub_patches[e_idx][1].set_xy(_stub_corners(xj, yj, xi, yi))

        # Update body rectangle (follows node 0, oriented along first element)
        x0n, y0n = _node_xy(x_abs, 0)
        x1n, y1n = _node_xy(x_abs, 1)
        ang0     = np.arctan2(y1n - y0n, x1n - x0n)
        body_patch.set_xy(_body_rect_corners(x0n, y0n, ang0))

        # Update motor rectangle (follows last node, perpendicular to last element)
        x_last, y_last = _node_xy(x_abs, n_nodes - 1)
        x_prev, y_prev = _node_xy(x_abs, n_nodes - 2)
        ang_last       = np.arctan2(y_last - y_prev, x_last - x_prev)
        motor_patch.set_xy(_motor_rect_corners(x_last, y_last, ang_last))

        # Head-neck segments
        base, joint, tip = seg_hist[frame]
        neck_line.set_data([base[0], joint[0]], [base[1], joint[1]])
        head_line.set_data([joint[0], tip[0]],  [joint[1], tip[1]])

        # Time label
        t_now = frame * frame_stride * results["dt"]
        time_text.set_text(f't = {t_now*1e3:.2f} ms')

        return arm_patches + all_stubs + [body_patch, motor_patch, neck_line, head_line, time_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(u_hist),
        init_func=init, blit=False,
        interval=sp.animation_interval, repeat=True)
    plt.tight_layout()
    plt.show()
    return ani


# ─────────────────────────────────────────────────────────────────────────────
# Plasticity post-processing
# ─────────────────────────────────────────────────────────────────────────────

def plot_plasticity(results, bp):
    # Produces a 4-panel figure summarising the plastic state of the beam.
    #
    # Panel 1 — Final plastic axial strain eps_p per element (bar chart).
    #           Reference lines at ±(N_p0 / EA) mark the first-yield axial strain.
    # Panel 2 — Final plastic curvature kappa_p per element [rad/m].
    #           Reference lines at ±(M_p0 / EI) mark the first-yield curvature.
    # Panel 3 — Final accumulated hardening variable alpha_p per element.
    #           alpha_p drives linear isotropic hardening; larger = more hardened.
    # Panel 4 — Time evolution of the peak values across all elements (dual y-axis).
    #
    # Broken elements are shown with a lighter red fill and hatching so their
    # residual plastic state (locked in at the moment of fracture) is still visible.
    # Does nothing if bp.use_plasticity is False.
    if not bp.use_plasticity:
        print("[plot_plasticity] plasticity was disabled — nothing to plot.")
        return

    eps_f   = results["eps_p_final"]
    kappa_f = results["kappa_p_final"]
    alpha_f = results["alpha_p_final"]
    n_elems = len(eps_f)
    elem_idx = np.arange(n_elems)

    # Identify broken elements (from fracture log)
    broken = set()
    for _, elems in results["fractures"]:
        broken.update(elems)
    is_broken = np.array([e in broken for e in range(n_elems)], dtype=bool)

    # Time axis for panel 4 (frame snapshots)
    dt          = results["dt"]
    frame_stride = results["frame_stride"]
    n_frames    = len(results["eps_p_hist"])
    t_frames_ms = np.arange(n_frames) * frame_stride * dt * 1e3   # [ms]

    # Peak values over elements at each frame
    max_eps_t   = [np.max(np.abs(s)) for s in results["eps_p_hist"]]
    max_kappa_t = [np.max(np.abs(s)) for s in results["kappa_p_hist"]]
    max_alpha_t = [np.max(s)         for s in results["alpha_p_hist"]]

    # Derived limits for reference lines
    N_p0 = bp.N_p
    M_p0 = bp.M_p
    eps_p_ref   = N_p0 / (bp.E * bp.A)           # eps at first yield (axial)
    kappa_p_ref = M_p0 / (bp.E * bp.I)           # kappa at first yield (bending)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Beam plasticity — post-simulation state", fontsize=11, fontweight='bold')
    ax1, ax2, ax3, ax4 = axes.flat

    # ── Panel 1: final plastic axial strain ──────────────────────────────────
    colors1 = np.where(is_broken, '#EF9A9A', '#4FC3F7')
    bars1   = ax1.bar(elem_idx, eps_f, color=colors1, edgecolor='#0277BD',
                      linewidth=0.6, zorder=2)
    # Hatching for broken elements
    for e, bar in enumerate(bars1):
        if is_broken[e]:
            bar.set_hatch('///')
            bar.set_edgecolor('#B71C1C')
    ax1.axhline(0, color='k', linewidth=0.6)
    ax1.axhline( eps_p_ref, color='tomato', linewidth=1.0, linestyle='--',
                 label=f'±ε at yield = {eps_p_ref:.2e}')
    ax1.axhline(-eps_p_ref, color='tomato', linewidth=1.0, linestyle='--')
    ax1.set_xlabel("Element index")
    ax1.set_ylabel("ε_p  [ — ]")
    ax1.set_title("Final plastic axial strain  ε_p")
    ax1.legend(fontsize=7)
    ax1.grid(True, axis='y', alpha=0.35)

    # ── Panel 2: final plastic curvature ─────────────────────────────────────
    colors2 = np.where(is_broken, '#EF9A9A', '#AB47BC')
    bars2   = ax2.bar(elem_idx, kappa_f, color=colors2, edgecolor='#6A1B9A',
                      linewidth=0.6, zorder=2)
    for e, bar in enumerate(bars2):
        if is_broken[e]:
            bar.set_hatch('///')
            bar.set_edgecolor('#B71C1C')
    ax2.axhline(0, color='k', linewidth=0.6)
    ax2.axhline( kappa_p_ref, color='tomato', linewidth=1.0, linestyle='--',
                 label=f'±κ at yield = {kappa_p_ref:.2e} rad/m')
    ax2.axhline(-kappa_p_ref, color='tomato', linewidth=1.0, linestyle='--')
    ax2.set_xlabel("Element index")
    ax2.set_ylabel("κ_p  [rad/m]")
    ax2.set_title("Final plastic curvature  κ_p")
    ax2.legend(fontsize=7)
    ax2.grid(True, axis='y', alpha=0.35)

    # ── Panel 3: final accumulated hardening variable ─────────────────────────
    colors3 = np.where(is_broken, '#EF9A9A', '#FF8A65')
    bars3   = ax3.bar(elem_idx, alpha_f, color=colors3, edgecolor='#BF360C',
                      linewidth=0.6, zorder=2)
    for e, bar in enumerate(bars3):
        if is_broken[e]:
            bar.set_hatch('///')
            bar.set_edgecolor('#B71C1C')
    ax3.axhline(0, color='k', linewidth=0.6)
    ax3.set_xlabel("Element index")
    ax3.set_ylabel("α_p  [plastic multiplier]")
    ax3.set_title("Accumulated hardening variable  α_p")
    ax3.grid(True, axis='y', alpha=0.35)
    ax3.legend(handles=[
        Patch(facecolor='#FF8A65', edgecolor='#BF360C', label='Intact'),
        Patch(facecolor='#EF9A9A', edgecolor='#B71C1C', hatch='///', label='Broken'),
    ], fontsize=7)

    # ── Panel 4: time evolution of peak plastic quantities ───────────────────
    ax4.plot(t_frames_ms, max_eps_t,   color='#0277BD', linewidth=1.5,
             label='max |ε_p|')
    ax4.plot(t_frames_ms, max_kappa_t, color='#6A1B9A', linewidth=1.5,
             label='max |κ_p|  [rad/m]')
    ax4r = ax4.twinx()
    ax4r.plot(t_frames_ms, max_alpha_t, color='#BF360C', linewidth=1.5,
              linestyle='--', label='max α_p')
    ax4r.set_ylabel("max α_p  [plastic multiplier]", color='#BF360C', fontsize=8)
    ax4r.tick_params(axis='y', labelcolor='#BF360C')
    ax4.set_xlabel("Time  [ms]")
    ax4.set_ylabel("Plastic strain / curvature")
    ax4.set_title("Time evolution of peak plastic quantities")
    ax4.grid(True, alpha=0.35)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    # Summary printout
    print("\n[Plasticity report]")
    print(f"  sigma_y    = {bp.sigma_y:.3e} Pa   "
          f"H_hard = {bp.H_hard:.3e} Pa  (H/E = {bp.H_hard/bp.E:.4f})")
    print(f"  N_p0       = {N_p0:.3e} N    "
          f"M_p0   = {M_p0:.3e} N*m")
    print(f"  Elements with plastic activity : "
          f"{int(np.sum(alpha_f > 0))} / {n_elems}")
    print(f"  Peak |eps_p|   = {float(np.max(np.abs(eps_f))):.4e}  "
          f"(element {int(np.argmax(np.abs(eps_f)))})")
    print(f"  Peak |kappa_p| = {float(np.max(np.abs(kappa_f))):.4e} rad/m  "
          f"(element {int(np.argmax(np.abs(kappa_f)))})")
    print(f"  Peak alpha_p   = {float(np.max(alpha_f)):.4e}  "
          f"(element {int(np.argmax(alpha_f))})")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Cool grey + blue palette: darker grey sidebar, white main area, soft blue accents
C = {
    "bg":           "#F7FAFC",   # main page background
    "sidebar":      "#5B6573",   # left nav dark grey-blue
    "panel":        "#EAF2FB",   # section background
    "card":         "#FFFFFF",   # inner card
    "border":       "#C7D5E5",   # borders
    "border_hi":    "#7EA6D8",   # highlighted border
    "nav_active":   "#7E8EA3",   # active nav item bg
    "accent":       "#2F6FB6",   # main blue accent
    "accent_hi":    "#4B86C7",   # lighter blue hover
    "accent_dim":   "#AFC6E6",   # dimmed blue
    "blue":         "#2F6FB6",   # data line color in preview
    "blue_dim":     "#6F9ACD",
    "green":        "#2E8B57",   # pass
    "green_dim":    "#D8EFE3",
    "yellow":       "#C69214",   # warn
    "yellow_dim":   "#F7E9BA",
    "red":          "#C45151",   # fail
    "red_dim":      "#F6DADA",
    "text":         "#1F2D3D",   # primary text
    "muted":        "#6B7C93",   # muted text on light panels
    "sidebar_text": "#F2F6FB",   # light text for sidebar
    "label":        "#4E647D",   # field labels
    "entry_bg":     "#FFFFFF",   # entry field background
    "entry_border": "#9DB9D9",
    "run_btn":      "#2F6FB6",
    "run_btn_hi":   "#4B86C7",
    "white":        "#FFFFFF",
}

MATERIAL_PRESETS = {
    "CFRP":            {"E": 2.6e9,  "rho": 1198.0, "sigma_u": 90e6},
    "CFRP (stiff)":    {"E": 6.5e9,  "rho": 1300.0, "sigma_u": 90e6},
    "Aluminium 6061":  {"E": 69e9,   "rho": 2700.0, "sigma_u": 276e6},
    "ABS Plastic":     {"E": 2.3e9,  "rho": 1050.0, "sigma_u": 40e6},
    "Steel":           {"E": 200e9,  "rho": 7800.0, "sigma_u": 400e6},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  REUSABLE WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class FieldRow(ctk.CTkFrame):
    """Label + entry on one row."""
    def __init__(self, parent, label, default, unit="", width=100, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=0)

        ctk.CTkLabel(self, text=label,
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=C["label"],
                     anchor="w").grid(row=0, column=0, sticky="w")

        self.var = tk.StringVar(value=str(default))
        self._entry = ctk.CTkEntry(self, textvariable=self.var,
                                   width=width, height=26,
                                   font=ctk.CTkFont("Courier", 11),
                                   fg_color=C["entry_bg"],
                                   border_color=C["entry_border"],
                                   text_color=C["text"])
        self._entry.grid(row=0, column=1, padx=(6, 4))

        if unit:
            ctk.CTkLabel(self, text=unit,
                         font=ctk.CTkFont("Helvetica", 9),
                         text_color=C["sidebar_text"],
                         width=44, anchor="w").grid(row=0, column=2)

    def get(self):
        return self.var.get()

    def set(self, val):
        self.var.set(str(val))


class AutoFieldRow(ctk.CTkFrame):
    """Checkbox 'Auto' + optional manual entry."""
    def __init__(self, parent, label, default_manual, unit="",
                 default_auto=True, width=100, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        self.columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text=label,
                     font=ctk.CTkFont("Helvetica", 11),
                     text_color=C["label"],
                     anchor="w").grid(row=0, column=0, sticky="w")

        self.auto_var = tk.BooleanVar(value=default_auto)
        ctk.CTkCheckBox(self, text="Auto",
                        variable=self.auto_var,
                        font=ctk.CTkFont("Helvetica", 10),
                        text_color=C["muted"],
                        fg_color=C["accent"],
                        hover_color=C["accent_hi"],
                        checkmark_color=C["white"],
                        border_color=C["border_hi"],
                        width=58, height=20,
                        command=self._toggle).grid(row=0, column=1, padx=(6, 4))

        self.manual_var = tk.StringVar(value=str(default_manual))
        self._entry = ctk.CTkEntry(self, textvariable=self.manual_var,
                                   width=width, height=26,
                                   font=ctk.CTkFont("Courier", 11),
                                   fg_color=C["entry_bg"],
                                   border_color=C["entry_border"],
                                   text_color=C["text"])
        self._entry.grid(row=0, column=2, padx=(0, 4))

        if unit:
            ctk.CTkLabel(self, text=unit,
                         font=ctk.CTkFont("Helvetica", 9),
                         text_color=C["muted"],
                         width=44, anchor="w").grid(row=0, column=3)

        self._toggle()

    def _toggle(self):
        is_auto = self.auto_var.get()
        self._entry.configure(
            state="disabled" if is_auto else "normal",
            fg_color=C["panel"] if is_auto else C["entry_bg"],
            text_color=C["muted"] if is_auto else C["text"])

    def is_auto(self):
        return self.auto_var.get()

    def get_manual(self):
        return self.manual_var.get()


def _section(parent, title):
    """Returns a labelled card frame."""
    outer = ctk.CTkFrame(parent, fg_color=C["panel"],
                         border_color=C["border"], border_width=1,
                         corner_radius=8)
    outer.pack(fill="x", padx=0, pady=(0, 10))
    outer.columnconfigure(0, weight=1)

    ctk.CTkLabel(outer, text=title.upper(),
                 font=ctk.CTkFont("Helvetica", 9, "bold"),
                 text_color=C["accent"],
                 anchor="w").pack(fill="x", padx=14, pady=(10, 2))

    sep = ctk.CTkFrame(outer, fg_color=C["border"], height=1)
    sep.pack(fill="x", padx=14, pady=(0, 8))

    body = ctk.CTkFrame(outer, fg_color="transparent")
    body.pack(fill="x", padx=14, pady=(0, 12))

    return body


def _row(parent, widget, pady=(2, 2)):
    widget.pack(fill="x", pady=pady)


class InjuryCard(ctk.CTkFrame):
    """Compact metric card for HIC / BrIC / Nij."""
    def __init__(self, parent, name, limit, unit="", **kw):
        super().__init__(parent, fg_color=C["card"],
                         border_color=C["border"], border_width=1,
                         corner_radius=8, **kw)
        self._limit = limit
        self._name  = name

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(top, text=name,
                     font=ctk.CTkFont("Helvetica", 11, "bold"),
                     text_color=C["label"], anchor="w").pack(side="left")
        self._lim_lbl = ctk.CTkLabel(top,
                     text=f"limit {limit:g}",
                     font=ctk.CTkFont("Helvetica", 9),
                     text_color=C["muted"], anchor="e")
        self._lim_lbl.pack(side="right")

        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.pack(fill="x", padx=14, pady=(0, 4))
        self._val_lbl = ctk.CTkLabel(mid, text="—",
                     font=ctk.CTkFont("Courier", 22, "bold"),
                     text_color=C["muted"], anchor="w")
        self._val_lbl.pack(side="left")
        self._ais_lbl = ctk.CTkLabel(mid, text="",
                     font=ctk.CTkFont("Helvetica", 10),
                     text_color=C["muted"], anchor="e")
        self._ais_lbl.pack(side="right", padx=(0, 4))

        self._bar_bg = ctk.CTkFrame(self, fg_color=C["border"],
                                    height=4, corner_radius=2)
        self._bar_bg.pack(fill="x", padx=14, pady=(0, 10))
        self._bar = ctk.CTkFrame(self._bar_bg, fg_color=C["muted"],
                                 height=4, corner_radius=2)
        self._bar.place(relx=0, rely=0, relwidth=0.0, relheight=1.0)

    def update(self, value, ais3p):
        ratio = value / self._limit
        if ratio < 0.75:
            col = C["green"]; bar_col = C["green"]
        elif ratio < 1.0:
            col = C["yellow"]; bar_col = C["yellow"]
        else:
            col = C["red"]; bar_col = C["red"]
        self._val_lbl.configure(text=f"{value:.2f}", text_color=col)
        self._ais_lbl.configure(text=f"AIS3+  {ais3p*100:.1f} %",
                                text_color=col)
        self._bar.configure(fg_color=bar_col)
        self._bar.place(relwidth=min(1.0, ratio))

    def reset(self):
        self._val_lbl.configure(text="—", text_color=C["muted"])
        self._ais_lbl.configure(text="", text_color=C["muted"])
        self._bar.place(relwidth=0.0)
        self._bar.configure(fg_color=C["muted"])


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class DroneImpactApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Drone Impact Safety Simulator")
        self.geometry("1380x860")
        self.minsize(1100, 700)
        self.configure(fg_color=C["bg"])

        self._results    = None
        self._sim_queue  = queue.Queue()
        self._run_count  = 0
        self._anim_win   = None

        self.columnconfigure(0, weight=0)   # sidebar
        self.columnconfigure(1, weight=1)   # content
        self.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_pages()
        self._navigate("drone")

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=200, fg_color=C["sidebar"],
                          corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.columnconfigure(0, weight=1)
        sb.rowconfigure(5, weight=1)   # pushes run btn to bottom

        # Logo
        logo = ctk.CTkFrame(sb, fg_color="transparent")
        logo.grid(row=0, column=0, sticky="ew", padx=18, pady=(22, 28))
        ctk.CTkLabel(logo, text="◈",
                     font=ctk.CTkFont("Helvetica", 26),
                     text_color=C["accent"]).pack(side="left")
        ctk.CTkLabel(logo, text="  DRONE\nIMPACT",
                     font=ctk.CTkFont("Helvetica", 11, "bold"),
                     text_color=C["sidebar_text"],
                     justify="left").pack(side="left", padx=(4, 0))

        sep = ctk.CTkFrame(sb, fg_color=C["border"], height=1)
        sep.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 16))

        # Nav buttons — store refs so we can highlight active
        self._nav_btns = {}
        nav_items = [
            ("drone",      "⬡",  "Drone Arm"),
            ("softtarget", "◎",  "Soft Target"),
            ("fem",        "▦",  "FEM & Solver"),
            ("results",    "◻",  "Results"),
        ]
        for row_idx, (key, icon, label) in enumerate(nav_items):
            btn = ctk.CTkButton(
                sb, text=f"  {icon}   {label}",
                font=ctk.CTkFont("Helvetica", 12),
                height=42, corner_radius=7,
                anchor="w",
                fg_color="transparent",
                text_color=C["muted"],
                hover_color=C["nav_active"],
                command=lambda k=key: self._navigate(k))
            btn.grid(row=2+row_idx, column=0, sticky="ew",
                     padx=10, pady=2)
            self._nav_btns[key] = btn

        # Spacer
        ctk.CTkFrame(sb, fg_color="transparent").grid(row=5, column=0, sticky="nsew")

        # Separator above run
        ctk.CTkFrame(sb, fg_color=C["border"], height=1).grid(
            row=6, column=0, sticky="ew", padx=14, pady=(0, 14))

        # Progress label
        self._pct_lbl = ctk.CTkLabel(sb, text="",
                     font=ctk.CTkFont("Courier", 10),
                     text_color=C["accent"])
        self._pct_lbl.grid(row=7, column=0, padx=18, sticky="w")

        self._progress = ctk.CTkProgressBar(sb, height=4,
                         fg_color=C["border"],
                         progress_color=C["accent"])
        self._progress.set(0)
        self._progress.grid(row=8, column=0, sticky="ew",
                            padx=14, pady=(4, 10))
        self._progress.grid_remove()

        # Run button
        self._run_btn = ctk.CTkButton(
            sb, text="▶  Run Simulation",
            font=ctk.CTkFont("Helvetica", 12, "bold"),
            height=42, corner_radius=8,
            fg_color=C["run_btn"],
            hover_color=C["run_btn_hi"],
            text_color=C["white"],
            command=self._start_simulation)
        self._run_btn.grid(row=9, column=0, sticky="ew",
                           padx=14, pady=(0, 20))

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────────────────────────────────
    def _navigate(self, key):
        self._current_page = key
        for k, btn in self._nav_btns.items():
            if k == key:
                btn.configure(fg_color=C["nav_active"],
                              text_color=C["white"])
            else:
                btn.configure(fg_color="transparent",
                              text_color=C["sidebar_text"])
        for k, frame in self._pages.items():
            if k == key:
                frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
            else:
                frame.grid_remove()

    # ─────────────────────────────────────────────────────────────────────────
    # Pages container
    # ─────────────────────────────────────────────────────────────────────────
    def _build_pages(self):
        self._pages = {}
        for key in ("drone", "softtarget", "fem", "results"):
            f = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
            f.columnconfigure(0, weight=1)
            # Keep the header row compact; page builders assign weight to the
            # scroll/content row below. A weighted row 0 creates a large empty
            # gap above and below the page title.
            f.rowconfigure(0, weight=0)
            self._pages[key] = f

        self._build_drone_page(self._pages["drone"])
        self._build_softtarget_page(self._pages["softtarget"])
        self._build_fem_page(self._pages["fem"])
        self._build_results_page(self._pages["results"])

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: page header
    # ─────────────────────────────────────────────────────────────────────────
    def _page_header(self, parent, title, subtitle):
        hdr = ctk.CTkFrame(parent, fg_color=C["panel"],
                           corner_radius=0)
        hdr.pack(fill="x", padx=0, pady=(0, 0))
        ctk.CTkLabel(hdr, text=title,
                     font=ctk.CTkFont("Georgia", 18, "bold"),
                     text_color=C["text"],
                     anchor="w").pack(side="left", padx=28, pady=(0, 0))
        ctk.CTkLabel(hdr, text=subtitle,
                     font=ctk.CTkFont("Helvetica", 10),
                     text_color=C["muted"],
                     anchor="w").pack(side="left", padx=(0, 28), pady=(0, 0))

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — Drone Arm
    # ─────────────────────────────────────────────────────────────────────────
    def _build_drone_page(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        # Header spans both columns
        hdr = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=0)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        ctk.CTkLabel(hdr, text="Drone Arm",
                     font=ctk.CTkFont("Georgia", 18, "bold"),
                     text_color=C["text"], anchor="w").pack(
                         side="left", padx=28, pady=(0, 0))
        ctk.CTkLabel(hdr, text="Geometry · Material · Impact · Damage · Plasticity",
                     font=ctk.CTkFont("Helvetica", 10),
                     text_color=C["muted"], anchor="w").pack(
                         side="left", pady=(0, 0))

        # Left: parameter form
        left_scroll = ctk.CTkScrollableFrame(parent, fg_color=C["bg"],
                      scrollbar_fg_color=C["bg"],
                      scrollbar_button_color=C["border"])
        left_scroll.grid(row=1, column=0, sticky="nsew", padx=(24, 12), pady=(0, 0))

        # ── Material preset ───────────────────────────────────────────────
        preset_card = ctk.CTkFrame(left_scroll, fg_color=C["panel"],
                                   border_color=C["border"], border_width=1,
                                   corner_radius=8)
        preset_card.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(preset_card, text="MATERIAL PRESET",
                     font=ctk.CTkFont("Helvetica", 9, "bold"),
                     text_color=C["accent"], anchor="w").pack(
                         fill="x", padx=14, pady=(10, 4))
        self._preset_var = tk.StringVar(value="CFRP")
        ctk.CTkOptionMenu(preset_card,
                          values=list(MATERIAL_PRESETS.keys()),
                          variable=self._preset_var,
                          font=ctk.CTkFont("Helvetica", 11),
                          fg_color=C["entry_bg"],
                          button_color=C["accent"],
                          button_hover_color=C["accent_hi"],
                          text_color=C["text"],
                          dropdown_fg_color=C["card"],
                          dropdown_hover_color=C["nav_active"],
                          dropdown_text_color=C["text"],
                          height=30,
                          command=self._apply_preset).pack(
                              fill="x", padx=14, pady=(0, 12))

        # ── Geometry ──────────────────────────────────────────────────────
        b = _section(left_scroll, "Arm Geometry")
        self._L0   = FieldRow(b, "Arm length",      "0.07", "m");     _row(b, self._L0)
        self._bw   = FieldRow(b, "Section width b", "20",   "mm");    _row(b, self._bw)
        self._bh   = FieldRow(b, "Section height h","20",   "mm");    _row(b, self._bh)
        self._E    = FieldRow(b, "Young's modulus",  "2.6",  "GPa");  _row(b, self._E)
        self._rho  = FieldRow(b, "Density",          "1198", "kg/m³");_row(b, self._rho)
        self._mbody= FieldRow(b, "Body mass",        "0.85", "kg");   _row(b, self._mbody)
        self._mmot = FieldRow(b, "Motor mass",       "0.028","kg");   _row(b, self._mmot)

        # ── Impact conditions ─────────────────────────────────────────────
        b2 = _section(left_scroll, "Impact Conditions")
        self._v0   = FieldRow(b2, "Velocity",       "19.0",  "m/s"); _row(b2, self._v0)
        self._ang  = FieldRow(b2, "Arm angle",      "0.0",   "deg"); _row(b2, self._ang)
        self._txo  = FieldRow(b2, "Tip x₀",         "-0.02", "m");   _row(b2, self._txo)
        self._tyo  = FieldRow(b2, "Tip y₀",         "0.175", "m");   _row(b2, self._tyo)
        self._vel_angled_var = tk.BooleanVar(value=False)
        _vc = ctk.CTkCheckBox(b2, text="Velocity along beam axis",
                              variable=self._vel_angled_var,
                              font=ctk.CTkFont("Helvetica", 11),
                              text_color=C["label"],
                              fg_color=C["accent"], hover_color=C["accent_hi"],
                              checkmark_color=C["white"],
                              border_color=C["border_hi"], height=22)
        _row(b2, _vc, pady=(6, 2))

        # ── Damage ────────────────────────────────────────────────────────
        b3 = _section(left_scroll, "Damage Model")
        self._sigU = FieldRow(b3, "Ultimate strength", "90",   "MPa"); _row(b3, self._sigU)
        self._di_r = FieldRow(b3, "Damage init ratio", "0.70", "");    _row(b3, self._di_r)
        self._dtau = FieldRow(b3, "Damage τ",           "200",  "µs"); _row(b3, self._dtau)

        # ── Plasticity ────────────────────────────────────────────────────
        b4 = _section(left_scroll, "Plasticity")
        self._use_plasticity_var = tk.BooleanVar(value=True)
        _pc = ctk.CTkCheckBox(b4, text="Enable plasticity",
                              variable=self._use_plasticity_var,
                              font=ctk.CTkFont("Helvetica", 11),
                              text_color=C["label"],
                              fg_color=C["accent"], hover_color=C["accent_hi"],
                              checkmark_color=C["white"],
                              border_color=C["border_hi"], height=22)
        _row(b4, _pc, pady=(0, 6))
        self._sigma_y = FieldRow(b4, "Yield stress σ_y",  "55",  "MPa"); _row(b4, self._sigma_y)
        self._H_hard  = FieldRow(b4, "Hardening mod. H",  "260", "MPa"); _row(b4, self._H_hard)

        # Right: live drone preview
        right = ctk.CTkFrame(parent, fg_color=C["panel"],
                             border_color=C["border"], border_width=1,
                             corner_radius=10)
        right.grid(row=1, column=1, sticky="nsew", padx=(12, 24), pady=(0, 0))
        right.rowconfigure(1, weight=1)

        ctk.CTkLabel(right, text="DRONE PREVIEW",
                     font=ctk.CTkFont("Helvetica", 9, "bold"),
                     text_color=C["accent"], anchor="w").pack(
                         fill="x", padx=16, pady=(14, 4))
        ctk.CTkFrame(right, fg_color=C["border"], height=1).pack(fill="x", padx=16)

        self._drone_fig = Figure(facecolor=C["panel"])
        self._drone_ax  = self._drone_fig.add_subplot(111)
        self._drone_fig.subplots_adjust(left=0.05, right=0.95,
                                        top=0.95, bottom=0.05)
        self._drone_canvas = FigureCanvasTkAgg(self._drone_fig, master=right)
        self._drone_canvas.get_tk_widget().pack(fill="both", expand=True,
                                                padx=10, pady=10)

        # Wire live preview traces
        for w in (self._L0, self._bw, self._bh, self._mbody,
                  self._mmot, self._v0, self._ang):
            w.var.trace_add("write", self._draw_drone_preview)
        self._draw_drone_preview()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — Soft Target
    # ─────────────────────────────────────────────────────────────────────────
    def _build_softtarget_page(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text="Soft Target",
                     font=ctk.CTkFont("Georgia", 18, "bold"),
                     text_color=C["text"], anchor="w").pack(
                         side="left", padx=28, pady=(0, 0))
        ctk.CTkLabel(hdr, text="Head-Neck double-pendulum model",
                     font=ctk.CTkFont("Helvetica", 10),
                     text_color=C["muted"], anchor="w").pack(
                         side="left", pady=(0, 0))

        scroll = ctk.CTkScrollableFrame(parent, fg_color=C["bg"],
                 scrollbar_fg_color=C["bg"],
                 scrollbar_button_color=C["border"])
        scroll.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 0))

        # Two-column layout inside scroll
        cols = ctk.CTkFrame(scroll, fg_color="transparent")
        cols.pack(fill="x")
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        left  = ctk.CTkFrame(cols, fg_color="transparent")
        right = ctk.CTkFrame(cols, fg_color="transparent")
        left.grid( row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        # ── Geometry ── (left)
        bl = _section(left, "Head-Neck Geometry")
        self._L_neck = FieldRow(bl, "Neck length L_neck", "0.10", "m");  _row(bl, self._L_neck)
        self._L_head = FieldRow(bl, "Head length L_head", "0.15", "m");  _row(bl, self._L_head)

        # ── Masses & Inertia ── (left)
        bm = _section(left, "Masses & Inertia")
        self._m_head       = FieldRow(bm, "Head mass",         "5.5",   "kg");   _row(bm, self._m_head)
        self._m_neck       = FieldRow(bm, "Neck mass",         "1.6",   "kg");   _row(bm, self._m_neck)
        self._r_head_in    = FieldRow(bm, "Head gyration r",   "0.060", "m");    _row(bm, self._r_head_in)
        self._r_head_com   = FieldRow(bm, "Head CoM offset",   "0.060", "m");    _row(bm, self._r_head_com)

        # ── Joint stiffness ── (left)
        bj = _section(left, "Passive Joint")
        self._k_global = FieldRow(bj, "Global stiffness k",  "10.0", "N/m"); _row(bj, self._k_global)
        self._c_global = FieldRow(bj, "Global damping c",    "10.0", "N·s/m"); _row(bj, self._c_global)
        self._r_ratio  = FieldRow(bj, "Stiffness ratio r",   "1.5",  "");      _row(bj, self._r_ratio)

        # ── Muscle activation ── (right)
        bmu = _section(right, "Muscle Activation")
        self._use_muscle_var = tk.BooleanVar(value=True)
        _mc = ctk.CTkCheckBox(bmu, text="Enable muscle reflex",
                              variable=self._use_muscle_var,
                              font=ctk.CTkFont("Helvetica", 11),
                              text_color=C["label"],
                              fg_color=C["accent"], hover_color=C["accent_hi"],
                              checkmark_color=C["white"],
                              border_color=C["border_hi"], height=22)
        _row(bmu, _mc, pady=(0, 6))
        self._t_delay  = FieldRow(bmu, "Reflex delay",    "70",   "ms");   _row(bmu, self._t_delay)
        self._tau_act  = FieldRow(bmu, "Activation τ",    "30",   "ms");   _row(bmu, self._tau_act)
        self._k_muscle = FieldRow(bmu, "Muscle stiffness","60.0", "N/m");  _row(bmu, self._k_muscle)
        self._c_muscle = FieldRow(bmu, "Muscle damping",  "1.0",  "N·s/m");_row(bmu, self._c_muscle)

        # ── Scalp compliance ── (right)
        bs = _section(right, "Scalp Compliance Layer")
        self._use_scalp_var = tk.BooleanVar(value=True)
        _sc = ctk.CTkCheckBox(bs, text="Enable scalp layer",
                              variable=self._use_scalp_var,
                              font=ctk.CTkFont("Helvetica", 11),
                              text_color=C["label"],
                              fg_color=C["accent"], hover_color=C["accent_hi"],
                              checkmark_color=C["white"],
                              border_color=C["border_hi"], height=22)
        _row(bs, _sc, pady=(0, 6))
        self._m_scalp    = FieldRow(bs, "Scalp mass",      "0.5",  "kg");    _row(bs, self._m_scalp)
        self._k_scalp    = FieldRow(bs, "Scalp stiffness", "130",  "kN/m");  _row(bs, self._k_scalp)
        self._zeta_scalp = FieldRow(bs, "Damping ratio ζ", "0.40", "");      _row(bs, self._zeta_scalp)

        # ── Injury thresholds ── (right)
        bit = _section(right, "Injury Thresholds")
        self._HIC_lim = FieldRow(bit, "HIC15 limit",    "700",   "");      _row(bit, self._HIC_lim)
        self._Nij_lim = FieldRow(bit, "Nij limit",      "1.0",   "");      _row(bit, self._Nij_lim)
        self._omYC    = FieldRow(bit, "ω_yC (BrIC)",    "56.45", "rad/s"); _row(bit, self._omYC)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — FEM & Solver
    # ─────────────────────────────────────────────────────────────────────────
    def _build_fem_page(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text="FEM & Solver",
                     font=ctk.CTkFont("Georgia", 18, "bold"),
                     text_color=C["text"], anchor="w").pack(
                         side="left", padx=28, pady=(0, 0))
        ctk.CTkLabel(hdr, text="Mesh · Integration · Damping · Contact · Fracture",
                     font=ctk.CTkFont("Helvetica", 10),
                     text_color=C["muted"], anchor="w").pack(
                         side="left", pady=(0, 0))

        scroll = ctk.CTkScrollableFrame(parent, fg_color=C["bg"],
                 scrollbar_fg_color=C["bg"],
                 scrollbar_button_color=C["border"])
        scroll.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 0))

        cols = ctk.CTkFrame(scroll, fg_color="transparent")
        cols.pack(fill="x")
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        left  = ctk.CTkFrame(cols, fg_color="transparent")
        right = ctk.CTkFrame(cols, fg_color="transparent")
        left.grid( row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        # ── Mesh ── (left)
        bm = _section(left, "Mesh")
        self._nn   = FieldRow(bm, "Number of nodes", "20", ""); _row(bm, self._nn)
        ctk.CTkLabel(bm, text="n_elements = n_nodes − 1",
                     font=ctk.CTkFont("Helvetica", 9),
                     text_color=C["muted"], anchor="w").pack(fill="x")

        # ── Time integration ── (left)
        bt = _section(left, "Time Integration")
        self._dt   = AutoFieldRow(bt, "Time step dt",   "0.1", "µs", default_auto=False); _row(bt, self._dt)
        self._Tend = FieldRow(bt, "Duration T",         "20",  "ms"); _row(bt, self._Tend)
        self._fstr = FieldRow(bt, "Frame stride",       "30",  "");   _row(bt, self._fstr)

        # ── Rayleigh Damping ── (left) — BUG FIX: default_auto=True
        br = _section(left, "Rayleigh Damping")
        self._auto_ray = tk.BooleanVar(value=True)   # FIX: was False in old GUI
        ray_cb = ctk.CTkCheckBox(br, text="Auto Rayleigh (from ζ and frequencies)",
                                 variable=self._auto_ray,
                                 font=ctk.CTkFont("Helvetica", 11),
                                 text_color=C["label"],
                                 fg_color=C["accent"], hover_color=C["accent_hi"],
                                 checkmark_color=C["white"],
                                 border_color=C["border_hi"],
                                 command=self._toggle_rayleigh)
        _row(br, ray_cb, pady=(0, 6))
        self._ray_alpha = FieldRow(br, "α (mass)",   "0.0",  "");   _row(br, self._ray_alpha)
        self._ray_beta  = FieldRow(br, "β (stiff.)", "0.0",  "");   _row(br, self._ray_beta)
        self._ray_zeta  = FieldRow(br, "Damping ζ",  "2.0",  "%");  _row(br, self._ray_zeta)
        self._ray_f1    = FieldRow(br, "f₁ (low)",   "200",  "Hz"); _row(br, self._ray_f1)
        self._ray_f2    = FieldRow(br, "f₂ (high)",  "1500", "Hz"); _row(br, self._ray_f2)
        self._toggle_rayleigh()

        # ── Fracture / incubation ── (right)
        bf = _section(right, "Fracture Timing")
        self._inc  = AutoFieldRow(bf, "Incubation steps", "5",  "",  default_auto=True); _row(bf, self._inc)
        self._tinc = FieldRow(bf, "t_inc (auto ref)",     "50", "µs"); _row(bf, self._tinc)

        # ── Contact ── (right)
        bc = _section(right, "Contact Model")
        self._use_hertz_var = tk.BooleanVar(value=True)
        _hc = ctk.CTkCheckBox(bc, text="Hertz contact  (F = k·δⁿ)",
                              variable=self._use_hertz_var,
                              font=ctk.CTkFont("Helvetica", 11),
                              text_color=C["label"],
                              fg_color=C["accent"], hover_color=C["accent_hi"],
                              checkmark_color=C["white"],
                              border_color=C["border_hi"], height=22)
        _row(bc, _hc, pady=(0, 6))
        self._hertz_exp = FieldRow(bc, "Hertz exponent n",  "1.5",  "");      _row(bc, self._hertz_exp)
        self._auto_k_var = tk.BooleanVar(value=True)
        _ak = ctk.CTkCheckBox(bc, text="Auto k_contact (energy balance)",
                              variable=self._auto_k_var,
                              font=ctk.CTkFont("Helvetica", 11),
                              text_color=C["label"],
                              fg_color=C["accent"], hover_color=C["accent_hi"],
                              checkmark_color=C["white"],
                              border_color=C["border_hi"], height=22)
        _row(bc, _ak, pady=(0, 4))
        self._kc        = FieldRow(bc, "k_contact (manual)","450",  "kN/m"); _row(bc, self._kc)
        self._delta_max = FieldRow(bc, "δ_max target",       "5.0",  "mm");  _row(bc, self._delta_max)
        self._cc        = FieldRow(bc, "c_contact",          "10",   "N·s/m");_row(bc, self._cc)
        self._mu        = FieldRow(bc, "Friction µ",         "0.20", "");    _row(bc, self._mu)
        self._e_r       = FieldRow(bc, "Restitution e",      "0.45", "");    _row(bc, self._e_r)
        self._cr        = FieldRow(bc, "Contact radius",     "10.0", "mm");  _row(bc, self._cr)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — Results
    # ─────────────────────────────────────────────────────────────────────────
    def _build_results_page(self, parent):
        parent.columnconfigure(0, weight=0)   # left: metrics
        parent.columnconfigure(1, weight=1)   # right: plot
        parent.rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=0)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        ctk.CTkLabel(hdr, text="Results",
                     font=ctk.CTkFont("Georgia", 18, "bold"),
                     text_color=C["text"], anchor="w").pack(
                         side="left", padx=28, pady=(0, 0))
        self._status_lbl = ctk.CTkLabel(hdr, text="No simulation run yet",
                     font=ctk.CTkFont("Helvetica", 10),
                     text_color=C["muted"], anchor="w")
        self._status_lbl.pack(side="left", pady=(0, 0))

        # Left metrics column (fixed width)
        left = ctk.CTkScrollableFrame(parent, width=240, fg_color=C["bg"],
               scrollbar_fg_color=C["bg"],
               scrollbar_button_color=C["border"])
        left.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=(0, 0))

        ctk.CTkLabel(left, text="INJURY METRICS",
                     font=ctk.CTkFont("Helvetica", 9, "bold"),
                     text_color=C["accent"], anchor="w").pack(
                         fill="x", pady=(0, 8))

        self._card_hic  = InjuryCard(left, "HIC 15", 700)
        self._card_hic.pack(fill="x", pady=(0, 10))
        self._card_bric = InjuryCard(left, "BrIC", 1.0)
        self._card_bric.pack(fill="x", pady=(0, 10))
        self._card_nij  = InjuryCard(left, "Nij", 1.0)
        self._card_nij.pack(fill="x", pady=(0, 18))

        ctk.CTkFrame(left, fg_color=C["border"], height=1).pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(left, text="FRACTURE LOG",
                     font=ctk.CTkFont("Helvetica", 9, "bold"),
                     text_color=C["accent"], anchor="w").pack(fill="x")
        self._frac_box = ctk.CTkTextbox(left, height=140,
                         font=ctk.CTkFont("Courier", 9),
                         fg_color=C["panel"],
                         border_color=C["border"], border_width=1,
                         text_color=C["muted"],
                         corner_radius=6)
        self._frac_box.insert("0.0", "—")
        self._frac_box.configure(state="disabled")
        self._frac_box.pack(fill="x", pady=(6, 0))

        self._anim_btn = ctk.CTkButton(
            left, text="▶  View Animation",
            font=ctk.CTkFont("Helvetica", 11, "bold"),
            height=38, corner_radius=8,
            fg_color=C["accent_dim"],
            hover_color=C["accent"],
            text_color=C["white"],
            state="disabled",
            command=self._show_animation)
        self._anim_btn.pack(fill="x", pady=(12, 0))

        # Right: head acceleration plot
        plot_outer = ctk.CTkFrame(parent, fg_color=C["panel"],
                                  border_color=C["border"], border_width=1,
                                  corner_radius=10)
        plot_outer.grid(row=1, column=1, sticky="nsew", padx=(8, 24), pady=(0, 0))
        plot_outer.rowconfigure(1, weight=1)
        plot_outer.columnconfigure(0, weight=1)

        ctk.CTkLabel(plot_outer, text="HEAD CoM ACCELERATION",
                     font=ctk.CTkFont("Helvetica", 9, "bold"),
                     text_color=C["accent"], anchor="w").grid(
                         row=0, column=0, sticky="w", padx=16, pady=(14, 4))

        self._res_fig = Figure(facecolor=C["panel"])
        self._res_ax  = self._res_fig.add_subplot(111)
        self._res_fig.subplots_adjust(left=0.09, right=0.97,
                                      top=0.92, bottom=0.12)
        self._style_res_ax()

        self._res_canvas = FigureCanvasTkAgg(self._res_fig, master=plot_outer)
        self._res_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew",
                                              padx=10, pady=(4, 10))

    def _style_res_ax(self):
        ax = self._res_ax
        ax.set_facecolor(C["card"])
        ax.tick_params(colors=C["muted"], labelsize=8)
        ax.xaxis.label.set_color(C["label"])
        ax.yaxis.label.set_color(C["label"])
        ax.title.set_color(C["label"])
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.grid(True, alpha=0.25, color=C["border"])

    # ─────────────────────────────────────────────────────────────────────────
    # Drone preview
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_drone_preview(self, *_):
        try:
            L0  = float(self._L0.get())
            v0  = float(self._v0.get())
            ang = float(self._ang.get())
            tip_x = float(self._txo.get())
            tip_y = float(self._tyo.get())
            m_body = float(self._mbody.get())
            m_motor = float(self._mmot.get())
        except ValueError:
            return

        ax = self._drone_ax
        ax.clear()
        ax.set_facecolor(C["card"])
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.grid(True, alpha=0.18, color=C["border"])

        ang_r = math.radians(ang)

        # Scale the visual size with the entered masses so heavier body / motors
        # become visibly larger while keeping the preview stable.
        body_scale = min(1.9, max(0.7, (max(m_body, 1e-6) / 0.85) ** 0.22))
        motor_scale = min(2.1, max(0.65, (max(m_motor, 1e-6) / 0.028) ** 0.28))

        # Central body of a quadcopter
        body_w = max(0.018, 0.26 * L0) * body_scale
        body_h = body_w
        body_x = tip_x - 0.5 * body_w
        body_y = tip_y - 0.5 * body_h
        body = plt.Rectangle((body_x, body_y), body_w, body_h,
                             facecolor=C["accent"], edgecolor=C["border_hi"],
                             linewidth=1.4, zorder=4)
        ax.add_patch(body)

        cx, cy = tip_x, tip_y
        arm_len = max(0.020, 0.55 * L0)
        rotor_r = max(0.006, 0.11 * L0) * motor_scale

        arm_dirs = [
            ( math.cos(ang_r),  math.sin(ang_r)),
            (-math.cos(ang_r), -math.sin(ang_r)),
            (-math.sin(ang_r),  math.cos(ang_r)),
            ( math.sin(ang_r), -math.cos(ang_r)),
        ]

        rotor_centres = []
        for dx, dy in arm_dirs:
            rx = cx + arm_len * dx
            ry = cy + arm_len * dy
            rotor_centres.append((rx, ry))
            ax.plot([cx, rx], [cy, ry], color=C["blue"], linewidth=3.0,
                    solid_capstyle="round", zorder=2)

        for rx, ry in rotor_centres:
            ring = plt.Circle((rx, ry), rotor_r, facecolor=C["panel"],
                              edgecolor=C["accent_hi"], linewidth=1.6, zorder=5)
            hub = plt.Circle((rx, ry), rotor_r * 0.22, facecolor=C["accent_hi"],
                             edgecolor=C["border_hi"], linewidth=1.0, zorder=6)
            ax.add_patch(ring)
            ax.add_patch(hub)
            ax.plot([rx - rotor_r * 0.9, rx + rotor_r * 0.9], [ry, ry],
                    color=C["muted"], linewidth=1.0, zorder=6)
            ax.plot([rx, rx], [ry - rotor_r * 0.9, ry + rotor_r * 0.9],
                    color=C["muted"], linewidth=1.0, zorder=6)

        # Forward direction / arm orientation hint
        arrow_len = max(0.018, 0.35 * L0)
        vx = arrow_len * math.cos(ang_r)
        vy = arrow_len * math.sin(ang_r)
        ax.annotate("", xy=(cx + vx, cy + vy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=C["accent_hi"],
                                    lw=1.8), zorder=7)

        # Frame bounds with gentle padding
        all_x = [cx] + [pt[0] for pt in rotor_centres]
        all_y = [cy] + [pt[1] for pt in rotor_centres]
        pad = max(0.02, 0.35 * L0)
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

        ax.set_xlabel("x  [m]", color=C["muted"], fontsize=8)
        ax.set_ylabel("y  [m]", color=C["muted"], fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(f"Quadcopter preview   L₀ = {L0*100:.1f} cm   m_body = {m_body:.3g} kg   m_rotor = {m_motor:.3g} kg",
                     color=C["muted"], fontsize=8, pad=6)
        self._drone_fig.tight_layout(pad=0.5)
        self._drone_canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Rayleigh toggle
    # ─────────────────────────────────────────────────────────────────────────
    def _toggle_rayleigh(self):
        is_auto = self._auto_ray.get()
        for w in (self._ray_alpha, self._ray_beta):
            w._entry.configure(
                state="disabled" if is_auto else "normal",
                fg_color=C["panel"] if is_auto else C["entry_bg"],
                text_color=C["muted"] if is_auto else C["text"])
        for w in (self._ray_zeta, self._ray_f1, self._ray_f2):
            w._entry.configure(
                state="normal" if is_auto else "disabled",
                fg_color=C["entry_bg"] if is_auto else C["panel"],
                text_color=C["text"] if is_auto else C["muted"])

    # ─────────────────────────────────────────────────────────────────────────
    # Material preset
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_preset(self, name):
        p = MATERIAL_PRESETS[name]
        self._E.set(f"{p['E']/1e9:.4g}")
        self._rho.set(f"{p['rho']:.4g}")
        self._sigU.set(f"{p['sigma_u']/1e6:.4g}")

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter collection
    # ─────────────────────────────────────────────────────────────────────────
    def _collect_params(self):
        def flt(w): return float(w.get())
        def iflt(w): return int(float(w.get()))

        bp = BeamParams()
        bp.L0                = flt(self._L0)
        bp.b                 = flt(self._bw) * 1e-3
        bp.h                 = flt(self._bh) * 1e-3
        bp.E                 = flt(self._E)  * 1e9
        bp.rho               = flt(self._rho)
        bp.m_body            = flt(self._mbody)
        bp.m_motor           = flt(self._mmot)
        bp.n_nodes           = max(3, iflt(self._nn))
        bp.v0                = flt(self._v0)
        bp.initial_angle_deg = flt(self._ang)
        bp.velocity_angled   = self._vel_angled_var.get()
        bp.tip_x0            = flt(self._txo)
        bp.tip_y0            = flt(self._tyo)
        bp.ultimate_strength = flt(self._sigU) * 1e6
        bp.damage_init_ratio = flt(self._di_r)
        bp.damage_tau        = flt(self._dtau) * 1e-6
        bp.mu                = flt(self._mu)
        bp.e_target          = flt(self._e_r)
        bp.contact_radius    = flt(self._cr)  * 1e-3

        bp.use_hertz          = self._use_hertz_var.get()
        bp.hertz_exponent     = flt(self._hertz_exp)
        bp.use_auto_k_contact = self._auto_k_var.get()
        if not bp.use_auto_k_contact:
            bp.k_contact      = flt(self._kc) * 1e3
        bp.delta_max_target   = flt(self._delta_max) * 1e-3
        bp.use_auto_contact   = True
        bp.c_contact          = flt(self._cc)

        bp.use_auto_rayleigh = self._auto_ray.get()
        if not bp.use_auto_rayleigh:
            bp.rayleigh_alpha = flt(self._ray_alpha)
            bp.rayleigh_beta  = flt(self._ray_beta)
        else:
            bp.zeta_target    = flt(self._ray_zeta) / 100.0
            bp.f_rayleigh_1   = flt(self._ray_f1)
            bp.f_rayleigh_2   = flt(self._ray_f2)

        bp.use_auto_incubation = self._inc.is_auto()
        if not bp.use_auto_incubation:
            bp.incubation_steps = max(1, int(float(self._inc.get_manual())))
        bp.t_inc = flt(self._tinc) * 1e-6

        bp.use_plasticity = self._use_plasticity_var.get()
        bp.sigma_y        = flt(self._sigma_y) * 1e6
        bp.H_hard         = flt(self._H_hard)  * 1e6

        hp = HeadNeckParams()
        hp.L_neck          = flt(self._L_neck)
        hp.L_head          = flt(self._L_head)
        hp.m_head          = flt(self._m_head)
        hp.m_neck          = flt(self._m_neck)
        hp.r_head_inertia  = flt(self._r_head_in)
        hp.r_head_com      = flt(self._r_head_com)
        hp.k_global        = flt(self._k_global)
        hp.c_global        = flt(self._c_global)
        hp.r_ratio         = flt(self._r_ratio)
        hp.use_muscle      = self._use_muscle_var.get()
        hp.t_delay         = flt(self._t_delay)  * 1e-3
        hp.tau_act         = flt(self._tau_act)  * 1e-3
        hp.k_muscle        = flt(self._k_muscle)
        hp.c_muscle        = flt(self._c_muscle)
        hp.use_scalp_layer = self._use_scalp_var.get()
        hp.m_scalp         = flt(self._m_scalp)
        hp.k_scalp         = flt(self._k_scalp) * 1e3
        hp.zeta_scalp      = flt(self._zeta_scalp)
        hp.HIC_limit       = flt(self._HIC_lim)
        hp.Nij_limit       = flt(self._Nij_lim)
        hp.omega_yC        = flt(self._omYC)
        hp.phi_rest        = hp.phi0
        hp.theta_rest      = hp.theta0
        hp.update_inertias()

        sp = SimParams()
        if not self._dt.is_auto():
            sp.dt = float(self._dt.get_manual()) * 1e-6
        else:
            sp.dt = 1e-6
        sp.T_end        = flt(self._Tend) * 1e-3
        sp.frame_stride = max(1, iflt(self._fstr))

        return bp, hp, sp

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────────────────────────────────
    def _start_simulation(self):
        try:
            bp, hp, sp = self._collect_params()
        except Exception as ex:
            messagebox.showerror("Parameter error", str(ex))
            return

        self._results = None
        self._anim_btn.configure(state="disabled", fg_color=C["accent_dim"])
        self._card_hic.reset()
        self._card_bric.reset()
        self._card_nij.reset()
        self._run_btn.configure(state="disabled",
                                text="⏳  Running…",
                                fg_color=C["accent_dim"])
        self._status_lbl.configure(text="Running simulation…",
                                   text_color=C["accent"])
        self._pct_lbl.configure(text="0 %")
        self._progress.set(0)
        self._progress.grid()

        def worker():
            try:
                import sys
                class _Cap:
                    def __init__(self, q):
                        self._q = q
                        self._o = sys.stdout
                    def write(self, s):
                        self._o.write(s)
                        if "Progress:" in s:
                            try:
                                pct = float(s.split("%")[0].split()[-1])
                                self._q.put(("progress", pct))
                            except Exception:
                                pass
                    def flush(self): self._o.flush()

                old = sys.stdout
                sys.stdout = _Cap(self._sim_queue)
                try:
                    res = simulate(bp, hp, sp, do_animation=False)
                finally:
                    sys.stdout = old
                res["bp"] = bp
                res["sp"] = sp
                self._sim_queue.put(("done", res))
            except Exception:
                import traceback
                self._sim_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()
        self._poll_sim()

    def _poll_sim(self):
        try:
            while True:
                msg, data = self._sim_queue.get_nowait()
                if msg == "progress":
                    self._progress.set(data / 100.0)
                    self._pct_lbl.configure(text=f"{data:.0f} %")
                elif msg == "done":
                    self._on_done(data)
                    return
                elif msg == "error":
                    self._on_error(data)
                    return
        except queue.Empty:
            pass
        self.after(80, self._poll_sim)

    def _on_done(self, res):
        self._results = res
        self._run_count += 1
        self._run_btn.configure(state="normal",
                                text="▶  Run Simulation",
                                fg_color=C["run_btn"])
        self._progress.grid_remove()
        self._pct_lbl.configure(text="")
        self._status_lbl.configure(
            text=f"Complete ✓   (run #{self._run_count})",
            text_color=C["green"])
        self._anim_btn.configure(state="normal", fg_color=C["run_btn"])

        ir = res["injury_report"]
        self._card_hic.update(ir["HIC15"],    ir["AIS3p_HIC15"])
        self._card_bric.update(ir["BrIC"],    ir["AIS3p_BrIC"])
        self._card_nij.update(ir["Nij_peak"], ir["AIS3p_Nij"])

        # Fracture log
        self._frac_box.configure(state="normal")
        self._frac_box.delete("0.0", "end")
        if res["fractures"]:
            for st, elems in res["fractures"]:
                t_ms = st * res["dt"] * 1e3
                self._frac_box.insert("end",
                    f"t={t_ms:.3f} ms  elem={elems}\n")
        else:
            self._frac_box.insert("end", "No fractures detected.")
        self._frac_box.configure(state="disabled")

        # Draw head acceleration plot
        self._draw_accel_plot(res)

        # Auto-navigate to results
        self._navigate("results")

    def _on_error(self, msg):
        self._run_btn.configure(state="normal",
                                text="▶  Run Simulation",
                                fg_color=C["run_btn"])
        self._progress.grid_remove()
        self._pct_lbl.configure(text="")
        self._status_lbl.configure(text="⚠  Error", text_color=C["red"])
        self._anim_btn.configure(state="disabled", fg_color=C["accent_dim"])
        messagebox.showerror("Simulation error", msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Animation viewer
    # ─────────────────────────────────────────────────────────────────────────
    def _show_animation(self):
        if not self._results:
            messagebox.showinfo("Animation", "Run a simulation first.")
            return
        try:
            animate(self._results["bp"], self._results["sp"], self._results)
        except Exception as ex:
            messagebox.showerror("Animation error", str(ex))

    # ─────────────────────────────────────────────────────────────────────────
    # Head acceleration plot
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_accel_plot(self, res):
        ax = self._res_ax
        ax.clear()
        self._style_res_ax()

        t_ms    = res["t_hist"] * 1e3
        accel_g = res["accel_g"]
        ir      = res["injury_report"]
        i0, i1  = ir["HIC15_window_idx"]

        # Fill under curve
        ax.fill_between(t_ms, accel_g, alpha=0.12, color=C["blue"])
        ax.plot(t_ms, accel_g, color=C["blue"], lw=1.4,
                label="Resultant acceleration [g]")

        # HIC15 window
        if i1 > i0:
            ax.axvspan(t_ms[i0], t_ms[i1], alpha=0.18, color=C["red"],
                       label=f"HIC15 window  ({ir['HIC15']:.0f})")

        # Fracture markers
        for st, _ in res["fractures"]:
            ax.axvline(st * res["dt"] * 1e3, color=C["accent"],
                       lw=0.8, linestyle=":", alpha=0.7)

        # Peak annotation
        peak_g   = float(accel_g.max())
        peak_idx = int(accel_g.argmax())
        ax.annotate(f"peak  {peak_g:.1f} g",
                    xy=(t_ms[peak_idx], peak_g),
                    xytext=(t_ms[peak_idx] + (t_ms[-1]-t_ms[0])*0.04,
                            peak_g * 0.85),
                    arrowprops=dict(arrowstyle="->", color=C["text"], lw=1.0),
                    fontsize=8, color=C["text"])

        ax.set_xlabel("Time  [ms]", fontsize=9)
        ax.set_ylabel("Acceleration  [g]", fontsize=9)
        ax.set_title(
            f"HIC15 = {ir['HIC15']:.1f}   "
            f"BrIC = {ir['BrIC']:.3f}   "
            f"Nij = {ir['Nij_peak']:.3f}   "
            f"AIS3+(HIC) = {ir['AIS3p_HIC15']*100:.1f} %",
            fontsize=8, color=C["label"])

        legend = ax.legend(fontsize=8, framealpha=0.5,
                           facecolor=C["card"], edgecolor=C["border"],
                           labelcolor=C["label"])
        self._res_fig.tight_layout(pad=0.6)
        self._res_canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = DroneImpactApp()
    app.mainloop()
