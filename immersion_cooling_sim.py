"""
Immersion Cooling Simulation – Methyl Myristate (Methyl Tetradecanoate)
=======================================================================
PDF Equations implemented:
  [Eq. 2]  NS:     ρ(∂u/∂t + u·∇u) = -∇p + μ_eff∇²u + ρgβ(T-T0)
  [Eq. 3]  Energy: ρcp(∂T/∂t + u·∇T) = ∇·((k + k_t)∇T) + Φ
  [Eq. 7]  dP      = f·(L/Dh)·(ρu²/2)         Darcy-Weisbach
  [Eq. 8]  PUE     = (P_IT + P_pump + P_chiller) / P_IT
  [Eq. 9]  P_pump  = V_dot·dP / eta_p
  [Eq. 10] P_chil  = (P_IT + P_pump) / COP
  [Eq. 5]  Nu      = 0.023·Re^0.8·Pr^0.4  Dittus-Boelter  → h_avg = Nu·k/Dh
  [Eq. 11] J       = w1·PUE + w2·ΔTmax-min − w3·h_avg

Turbulence model  (Smagorinsky SGS – full 2-D strain rate tensor):
  |S|² = 2(∂u/∂x)² + 2(∂v/∂y)² + (∂u/∂y + ∂v/∂x)²
  ν_t  = (C_s · Δ)² · |S|        C_s=0.17, Δ=√(dx·dy)
  k_t  = ρ · cp · ν_t / Pr_t     Pr_t = 0.85  (turbulent Prandtl)
  μ_eff = μ_lam + ρ·ν_t          used in momentum
  α_eff = k/(ρ·cp) + ν_t/Pr_t    used in energy   ← was missing before

  Turbulent inlet BC: mean plug-flow + random 5% perturbation to seed turbulence

Inter-server gradient:
  - Server 1 and Server 2 have INDEPENDENT load sliders (Load1 / Load2)
  - Gap zone between servers tracked explicitly: T_gap_avg
  - ΔT_inter = T_srv2_avg − T_srv1_avg displayed live on dashboard & CMD
  - Gradient map shows thermal plumes rising from each server independently

Sliders:
  K         – inlet flow rate multiplier    [0.1 – 2.0]
  Load1     – server 1 heat load            [0.5 – 3.0]
  Load2     – server 2 heat load            [0.5 – 3.0]
  Sim Speed – steps per frame               [1 – 60]
  w1/w2/w3  – multi-objective weights
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRID / TIME
# ─────────────────────────────────────────────────────────────────────────────
Nx, Ny    = 100, 50           # slightly larger for better inter-server detail
Lx, Ly    = 0.70, 0.35
dx, dy    = Lx / Nx, Ly / Ny
dt        = 3e-4
PROP_SKIP = 5
T_IN      = 298.15            # K  (25 °C)
g         = 9.81

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TURBULENCE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C_S     = 0.17               # Smagorinsky constant
DELTA   = np.sqrt(dx * dy)   # filter width
PR_T    = 0.85               # turbulent Prandtl number
CS2D2   = (C_S * DELTA) ** 2 # (C_s · Δ)²  precomputed

# ─────────────────────────────────────────────────────────────────────────────
# 3.  METHYL MYRISTATE – T-DEPENDENT PROPERTIES  (270–420 K)
# ─────────────────────────────────────────────────────────────────────────────
T_REF = 298.15

def fluid_props(T_field, out):
    """
    Fills out[5, Ny, Nx]:
      [0] rho  kg/m³    linear:  870 - 0.72·ΔT
      [1] cp   J/kg·K   linear: 1990 + 2.50·ΔT
      [2] mu   Pa·s     Andrade: 1.42e-5·exp(2680/T)
      [3] k    W/m·K    linear:  0.155 - 6e-5·ΔT
      [4] beta 1/K      8.0e-4 (const for organic esters)
    """
    dT = T_field - T_REF
    np.subtract(870.0,  0.72  * dT, out=out[0])
    np.add(1990.0,      2.5   * dT, out=out[1])
    np.clip(T_field, 270, 430,      out=out[2])
    np.divide(2680.0, out[2],       out=out[2])
    np.exp(out[2],                  out=out[2])
    out[2] *= 1.42e-5
    np.subtract(0.155,  6e-5  * dT, out=out[3])
    out[4].fill(8.0e-4)

_p0 = np.empty((5, 1, 1))
fluid_props(np.full((1,1), T_REF), _p0)
rho0, cp0, mu0, k0, beta0 = [float(_p0[i,0,0]) for i in range(5)]

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SERVER MASKS  (independent loads, explicit gap zone)
# ─────────────────────────────────────────────────────────────────────────────
# Server positions chosen so there is a clear gap between them
#   S1: cols 15–32   S2: cols 45–62   Gap: cols 32–45
srv1 = np.zeros((Ny, Nx), dtype=bool);  srv1[16:32, 15:33] = True
srv2 = np.zeros((Ny, Nx), dtype=bool);  srv2[16:32, 45:63] = True
srv_mask = srv1 | srv2

# Inter-server gap zone (fluid region between the two server blocks)
gap_mask = np.zeros((Ny, Nx), dtype=bool)
gap_mask[16:32, 33:45] = True   # same y-band, x between servers

BASE_Q = 1.2e5   # W/m³  reference heat generation
Dh     = 0.05    # hydraulic diameter [m]
eta_p  = 0.80    # pump efficiency
COP    = 4.5     # chiller COP

# Q arrays built dynamically each frame from slider values (no hardcoded ratio)
Q1 = np.zeros((Ny, Nx))   # server 1 heat map (updated each step)
Q2 = np.zeros((Ny, Nx))   # server 2 heat map

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STATE + PRE-ALLOCATED SCRATCH BUFFERS
# ─────────────────────────────────────────────────────────────────────────────
T = np.full((Ny, Nx), T_IN)
u = np.zeros((Ny, Nx))
v = np.zeros((Ny, Nx))

_props  = np.empty((5, Ny, Nx))

# Momentum / energy scratch
_lap    = np.empty((Ny, Nx))
_adv    = np.empty((Ny, Nx))
_tmp    = np.empty((Ny, Nx))
_nu_t   = np.empty((Ny, Nx))   # turbulent kinematic viscosity ν_t
_nu_eff = np.empty((Ny, Nx))   # total eff. kinematic viscosity
_alpha_eff = np.empty((Ny, Nx))# total thermal diffusivity
_buoy   = np.empty((Ny, Nx))
_u_new  = np.empty((Ny, Nx))
_v_new  = np.empty((Ny, Nx))
_src    = np.empty((Ny, Nx))
_rhocp  = np.empty((Ny, Nx))

# Strain-rate scratch for Smagorinsky
_dudx   = np.empty((Ny, Nx))
_dudy   = np.empty((Ny, Nx))
_dvdx   = np.empty((Ny, Nx))
_dvdy   = np.empty((Ny, Nx))
_S2     = np.empty((Ny, Nx))   # |S|²

# Flux-split velocity components (no np.where)
_up     = np.empty((Ny, Nx))
_um     = np.empty((Ny, Nx))
_vp     = np.empty((Ny, Nx))
_vm     = np.empty((Ny, Nx))

# Two pad buffers (never shared between concurrent stencil calls)
_pad_A  = np.empty((Ny+2, Nx+2))
_pad_B  = np.empty((Ny+2, Nx+2))

_prop_tick = [0]

# ─────────────────────────────────────────────────────────────────────────────
# 6.  STENCIL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _fill_pad(f, buf):
    """Edge-replicate f into buf.  No heap allocation."""
    global _pad_A, _pad_B
    buf[1:-1, 1:-1] = f
    buf[0,   1:-1]  = f[0,  :]
    buf[-1,  1:-1]  = f[-1, :]
    buf[1:-1,  0]   = f[:,  0]
    buf[1:-1, -1]   = f[:, -1]
    buf[0,  0] = f[0,  0];  buf[0,  -1] = f[0,  -1]
    buf[-1, 0] = f[-1, 0];  buf[-1, -1] = f[-1, -1]
    return buf


def laplacian_into(f, out):
    """5-point Laplacian → out.  Uses _pad_A."""
    global _pad_A
    p = _fill_pad(f, _pad_A)
    np.add(p[2:, 1:-1], p[:-2, 1:-1], out=out)
    out += p[1:-1, 2:]
    out += p[1:-1, :-2]
    out -= 4.0 * f
    out /= (dx * dy)


def upwind_into(f, uf, vf, out):
    """Flux-split upwind advection → out.  Uses _pad_B."""
    global _up, _um, _vp, _vm, _adv, _buoy, _pad_B
    p  = _fill_pad(f, _pad_B)
    fi = p[1:-1, 1:-1]
    fE = p[1:-1,  2:]
    fW = p[1:-1, :-2]
    fN = p[2:,   1:-1]
    fS = p[:-2,  1:-1]
    np.maximum(uf, 0.0, out=_up);  np.minimum(uf, 0.0, out=_um)
    np.maximum(vf, 0.0, out=_vp);  np.minimum(vf, 0.0, out=_vm)
    np.subtract(fi, fW, out=out);   out  /= dx;  out  *= _up
    np.subtract(fE, fi, out=_adv);  _adv /= dx;  _adv *= _um;  out += _adv
    np.subtract(fi, fS, out=_buoy); _buoy /= dy; _buoy *= _vp; out += _buoy
    np.subtract(fN, fi, out=_buoy); _buoy /= dy; _buoy *= _vm; out += _buoy


def grad_x(f, out, buf):
    """Central-difference ∂f/∂x → out."""
    p = _fill_pad(f, buf)
    np.subtract(p[1:-1, 2:], p[1:-1, :-2], out=out)
    out /= (2.0 * dx)


def grad_y(f, out, buf):
    """Central-difference ∂f/∂y → out."""
    p = _fill_pad(f, buf)
    np.subtract(p[2:, 1:-1], p[:-2, 1:-1], out=out)
    out /= (2.0 * dy)


def smagorinsky_nu_t(u_f, v_f, out):
    """
    Smagorinsky SGS turbulent kinematic viscosity → out:
        |S|² = 2(∂u/∂x)² + 2(∂v/∂y)² + (∂u/∂y + ∂v/∂x)²
        ν_t  = (C_s·Δ)² · |S|
    Uses full 2-D strain rate tensor — not just one shear component.
    """
    global _dudx, _dudy, _dvdx, _dvdy, _S2, _pad_A, _pad_B
    grad_x(u_f, _dudx, _pad_A)
    grad_y(u_f, _dudy, _pad_B)
    grad_x(v_f, _dvdx, _pad_A)
    grad_y(v_f, _dvdy, _pad_B)
    # |S|² = 2·(∂u/∂x)² + 2·(∂v/∂y)² + (∂u/∂y + ∂v/∂x)²
    np.multiply(_dudx, _dudx, out=_S2); _S2 *= 2.0
    np.multiply(_dvdy, _dvdy, out=_tmp); _tmp *= 2.0; _S2 += _tmp
    np.add(_dudy, _dvdx, out=_tmp); _tmp **= 2; _S2 += _tmp
    np.sqrt(_S2, out=out)      # |S|
    out *= CS2D2               # ν_t = (C_s·Δ)²·|S|

# ─────────────────────────────────────────────────────────────────────────────
# 7.  PHYSICS STEP
# ─────────────────────────────────────────────────────────────────────────────
def step_physics(flow_K, load1, load2):
    global T, u, v
    global _props, _lap, _adv, _tmp, _nu_t, _nu_eff, _alpha_eff
    global _buoy, _u_new, _v_new, _src, _rhocp
    global _pad_A, _pad_B, _up, _um, _vp, _vm
    global _dudx, _dudy, _dvdx, _dvdy, _S2, _prop_tick
    global Q1, Q2

    # ── Fluid properties (cached) ──────────────────────────────────────────
    _prop_tick[0] += 1
    if _prop_tick[0] >= PROP_SKIP:
        fluid_props(T, _props);  _prop_tick[0] = 0
    rho_f  = _props[0];  cp_f = _props[1]
    mu_f   = _props[2];  k_f  = _props[3]
    beta_f = _props[4]

    inlet_vel = 0.25 * flow_K

    # ── Smagorinsky ν_t (full strain rate) ────────────────────────────────
    smagorinsky_nu_t(u, v, _nu_t)           # ν_t = (C_s·Δ)²·|S|

    # ── Effective kinematic viscosity for momentum ─────────────────────────
    #   ν_eff = μ_lam/ρ + ν_t
    np.divide(mu_f, np.clip(rho_f, 1.0, None), out=_nu_eff)
    _nu_eff += _nu_t

    # ── Effective thermal diffusivity for energy ───────────────────────────
    #   α_eff = k/(ρ·cp) + ν_t/Pr_t
    np.multiply(rho_f, cp_f, out=_rhocp)
    np.clip(_rhocp, 1.0, None, out=_rhocp)
    np.divide(k_f, _rhocp, out=_alpha_eff)  # molecular α
    np.divide(_nu_t, PR_T, out=_tmp)
    _alpha_eff += _tmp                       # + turbulent α_t = ν_t/Pr_t

    # ── Buoyancy (Boussinesq) ─────────────────────────────────────────────
    T_mean = float(T.mean())
    np.subtract(T, T_mean, out=_buoy)
    _buoy *= g * beta_f

    # ── u-momentum ────────────────────────────────────────────────────────
    laplacian_into(u, _lap);  _lap *= _nu_eff
    upwind_into(u, u, v, _adv)
    np.subtract(_lap, _adv, out=_u_new)
    _u_new *= dt;  _u_new += u
    np.clip(_u_new, -4.0, 4.0, out=_u_new)

    # ── v-momentum ────────────────────────────────────────────────────────
    laplacian_into(v, _lap);  _lap *= _nu_eff
    upwind_into(v, u, v, _adv)
    np.subtract(_lap, _adv, out=_v_new)
    _v_new += _buoy
    _v_new *= dt;  _v_new += v
    np.clip(_v_new, -4.0, 4.0, out=_v_new)

    # ── Velocity BCs ──────────────────────────────────────────────────────
    # Turbulent inlet: plug flow + 5% random perturbation to seed turbulence
    noise = np.random.uniform(-0.05, 0.05, Ny) * inlet_vel
    _u_new[:, 0]  = inlet_vel + noise
    _v_new[:, 0]  = 0.0
    _u_new[:, -1] = _u_new[:, -2]   # outlet zero-gradient
    _v_new[:, -1] = _v_new[:, -2]
    _u_new[0, :]  = 0.0;  _u_new[-1, :] = 0.0   # no-slip walls
    _v_new[0, :]  = 0.0;  _v_new[-1, :] = 0.0
    np.copyto(u, _u_new);  np.copyto(v, _v_new)

    # ── Heat sources (independent loads for each server) ──────────────────
    Q1[:] = 0.0;  Q1[srv1] = BASE_Q * load1
    Q2[:] = 0.0;  Q2[srv2] = BASE_Q * load2
    np.add(Q1, Q2, out=_src)
    _src /= _rhocp                  # Q / (ρ·cp)

    # ── Energy equation (with turbulent thermal diffusivity) ──────────────
    laplacian_into(T, _lap)
    _lap *= _alpha_eff              # (α_mol + α_t)·∇²T
    upwind_into(T, u, v, _adv)
    np.subtract(_lap, _adv, out=_lap)
    _lap += _src
    _lap *= dt
    T    += _lap
    np.clip(T, T_IN - 5.0, 450.0, out=T)

    # ── Temperature BCs ───────────────────────────────────────────────────
    T[:, 0]  = T_IN            # fixed inlet
    T[:, -1] = T[:, -2]        # outlet zero-gradient
    T[0,  :] = T[1,  :]        # adiabatic walls
    T[-1, :] = T[-2, :]


# Warm start
fluid_props(T, _props)
smagorinsky_nu_t(u, v, _nu_t)  # pre-fill ν_t

# ─────────────────────────────────────────────────────────────────────────────
# 8.  METRICS  [PDF Eq. 7–11]
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(flow_K, load1, load2):
    iv    = 0.25 * flow_K
    Re    = (rho0 * iv * Dh) / (mu0 + 1e-12)
    Pr    = (mu0  * cp0) / (k0 + 1e-12)
    Re_t  = max(Re, 2300.0)
    Nu    = 0.023 * Re_t**0.8 * Pr**0.4
    h_avg = Nu * k0 / Dh

    f_d = (64.0 if Re < 1 else 64.0/Re if Re < 2300 else 0.316/Re**0.25)
    dP        = f_d * (Lx/Dh) * rho0 * iv**2 / 2.0
    cell_area = dx * dy
    P_IT      = BASE_Q * (load1*float(srv1.sum()) + load2*float(srv2.sum())) * cell_area
    V_dot     = iv * Ly
    P_pump    = (V_dot * dP) / eta_p
    P_chiller = (P_IT + P_pump) / COP
    PUE       = (P_IT + P_pump + P_chiller) / max(P_IT, 1e-9)

    T_s1      = T[srv1]
    T_s2      = T[srv2]
    T_gap_arr = T[gap_mask]

    dT_stack   = float(T[srv_mask].max() - T[srv_mask].min())
    T_s1_avg   = float(T_s1.mean())
    T_s2_avg   = float(T_s2.mean())
    dT_inter   = T_s2_avg - T_s1_avg      # inter-server thermal gradient
    T_gap_avg  = float(T_gap_arr.mean())  # gap-zone temperature
    nu_t_mean  = float(_nu_t.mean())      # mean turbulent viscosity

    return dict(PUE=PUE, Re=Re, Pr=Pr, Nu=Nu, h_avg=h_avg,
                dP=dP, P_IT=P_IT, P_pump=P_pump, P_chiller=P_chiller,
                dT_stack=dT_stack,
                T_s1_avg=T_s1_avg, T_s2_avg=T_s2_avg,
                dT_inter=dT_inter, T_gap_avg=T_gap_avg,
                nu_t_mean=nu_t_mean)


def compute_J(m, w1, w2, w3):
    return w1*m['PUE'] + w2*m['dT_stack'] - w3*(m['h_avg']/1e4)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family'     : 'monospace',
    'axes.facecolor'  : '#0d1117',
    'figure.facecolor': '#0d1117',
    'text.color'      : '#c9d1d9',
    'axes.labelcolor' : '#c9d1d9',
    'xtick.color'     : '#8b949e',
    'ytick.color'     : '#8b949e',
    'axes.edgecolor'  : '#30363d',
    'grid.color'      : '#21262d',
    'axes.titlecolor' : '#e6edf3',
})

fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    "Immersion Cooling  |  Methyl Myristate  |  "
    "Smagorinsky Turbulence + Inter-Server Gradient  [PDF Eq. 2,3,8–11]",
    fontsize=11, fontweight='bold', color='#58a6ff', y=0.995)

outer = gridspec.GridSpec(
    6, 1, figure=fig,
    height_ratios=[5, 3.8, 3.8, 3.2, 1.0, 1.0],
    hspace=0.56, left=0.06, right=0.97, top=0.965, bottom=0.03)

r0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.32)
r1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.32)
r2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.32)
r3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], wspace=0.32)
r4 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[4], wspace=0.28)
r5 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[5], wspace=0.28)

ax_temp  = fig.add_subplot(r0[0])   # heatmap + quiver
ax_grad  = fig.add_subplot(r0[1])   # ||∇T||
ax_loop  = fig.add_subplot(r1[0])   # T_in / T_out / ΔT
ax_pue   = fig.add_subplot(r1[1])   # PUE
ax_J     = fig.add_subplot(r2[0])   # multi-obj J
ax_con   = fig.add_subplot(r2[1])   # constraints
ax_inter = fig.add_subplot(r3[0])   # inter-server ΔT + T_s1 / T_s2 / T_gap
ax_turb  = fig.add_subplot(r3[1])   # turbulent ν_t mean

ax_sk    = fig.add_subplot(r4[0])   # K slider
ax_sl1   = fig.add_subplot(r4[1])   # Load1 slider
ax_sl2   = fig.add_subplot(r4[2])   # Load2 slider
ax_spd   = fig.add_subplot(r4[3])   # Sim Speed

ax_sw1   = fig.add_subplot(r5[0])
ax_sw2   = fig.add_subplot(r5[1])
ax_sw3   = fig.add_subplot(r5[2])

# ── Heatmap ──────────────────────────────────────────────────────────────────
img_temp = ax_temp.imshow(T, cmap='magma', origin='lower',
                          extent=[0, Lx, 0, Ly],
                          vmin=T_IN-1, vmax=T_IN+55,
                          aspect='auto', interpolation='bilinear')
cb_t = fig.colorbar(img_temp, ax=ax_temp, fraction=0.046, pad=0.04)
cb_t.set_label("T [K]", color='#8b949e', fontsize=8)

# ── Gradient map ─────────────────────────────────────────────────────────────
img_grad = ax_grad.imshow(np.zeros((Ny, Nx)), cmap='plasma',
                          origin='lower', extent=[0, Lx, 0, Ly],
                          vmin=0, vmax=800, aspect='auto',
                          interpolation='bilinear')
cb_g = fig.colorbar(img_grad, ax=ax_grad, fraction=0.046, pad=0.04)
cb_g.set_label("||∇T|| [K/m]", color='#8b949e', fontsize=8)

# ── Quiver on heatmap ────────────────────────────────────────────────────────
qs  = 5
Xg, Yg = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
quiver  = ax_temp.quiver(
    Xg[::qs, ::qs], Yg[::qs, ::qs],
    u[::qs, ::qs],  v[::qs, ::qs],
    color='#7dd3fc', alpha=0.72,
    scale=2.8, width=0.004,
    headwidth=4, headlength=5, zorder=10)
ax_temp.quiverkey(quiver, X=0.87, Y=1.045, U=1.0, label='Unit vel',
                  labelpos='E', color='#7dd3fc',
                  labelcolor='#7dd3fc', fontproperties={'size': 7})

# ── Server + gap zone overlays ────────────────────────────────────────────────
for ax_ in (ax_temp, ax_grad):
    for msk, col, lbl in [
            (srv1,     '#ff7b72', 'S1'),
            (srv2,     '#ffa657', 'S2'),
            (gap_mask, '#3fb950', 'GAP')]:
        ry, cx = np.where(msk)
        ax_.add_patch(plt.Rectangle(
            (cx.min()*dx, ry.min()*dy),
            (cx.max()-cx.min()+1)*dx, (ry.max()-ry.min()+1)*dy,
            lw=1.8, edgecolor=col,
            facecolor='#3fb950' if lbl=='GAP' else 'none',
            alpha=0.08 if lbl=='GAP' else 1.0,
            ls='--', zorder=11))
        ax_.text(cx.min()*dx+0.004, ry.min()*dy+0.004,
                 lbl, color=col, fontsize=7,
                 fontweight='bold', zorder=12)
    ax_.annotate("← Inlet", xy=(0.01, Ly*0.12),
                 xytext=(0.05, Ly*0.07), textcoords='data',
                 color='#3fb950', fontsize=7,
                 arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.2))
    ax_.annotate("Outlet →", xy=(Lx-0.01, Ly*0.12),
                 xytext=(Lx-0.16, Ly*0.07), textcoords='data',
                 color='#f78166', fontsize=7,
                 arrowprops=dict(arrowstyle='->', color='#f78166', lw=1.2))
    ax_.set_xlabel("x [m]", fontsize=8)
    ax_.set_ylabel("y [m]", fontsize=8)

ax_temp.set_title(
    "Coolant Flow & Temperature [K]  —  arrows = velocity field  "
    "(Smagorinsky turbulence)",
    fontsize=9, pad=5)
ax_grad.set_title("Temperature Gradient ||∇T|| [K/m]", fontsize=10, pad=5)

# ── Large PUE + Tin/Tout overlay on heatmap ───────────────────────────────────
pue_overlay = ax_temp.text(
    0.02, 0.98, "PUE = --",
    transform=ax_temp.transAxes, fontsize=14, fontweight='bold',
    color='#58a6ff', va='top', zorder=15,
    bbox=dict(boxstyle='round,pad=0.4', fc='#0d1117',
              ec='#58a6ff', alpha=0.90, lw=2))

tin_overlay = ax_temp.text(
    0.02, 0.81,
    "T_in  = -- °C\nT_out = -- °C\nΔT    = -- K",
    transform=ax_temp.transAxes, fontsize=8,
    color='#e6edf3', va='top', zorder=15,
    bbox=dict(boxstyle='round,pad=0.35', fc='#0d1117',
              ec='#3fb950', alpha=0.88, lw=1.5))

# ── T_in / T_out / ΔT ────────────────────────────────────────────────────────
ax_loop.set_title("Temperature Loop  (T_in · T_out · ΔT)", fontsize=10, pad=5)
ax_loop.set_xlabel("Frame", fontsize=8);  ax_loop.set_ylabel("K", fontsize=8)
ax_loop.grid(True, alpha=0.25)
line_in,  = ax_loop.plot([], [], color='#3fb950', lw=1.8, label='T_in')
line_out, = ax_loop.plot([], [], color='#ff7b72', lw=1.8, label='T_out')
line_dt,  = ax_loop.plot([], [], color='#f0e040', lw=1.4, ls='--', label='ΔT')
ax_loop.legend(loc='upper left', fontsize=7, framealpha=0.3)
txt_loop = ax_loop.text(
    0.56, 0.97, "T_in=--\nT_out=--\nΔT=--",
    transform=ax_loop.transAxes, fontsize=8, color='#e6edf3', va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='#161b22', ec='#3fb950', alpha=0.88))

# ── PUE ──────────────────────────────────────────────────────────────────────
ax_pue.set_title("Power Usage Effectiveness  [Eq. 8]", fontsize=10, pad=5)
ax_pue.set_xlabel("Frame", fontsize=8);  ax_pue.set_ylabel("PUE", fontsize=8)
ax_pue.grid(True, alpha=0.25)
ax_pue.axhline(1.15, color='#ff7b72', ls=':', lw=2.0, label='Limit 1.15 [§3.4]')
ax_pue.axhline(1.00, color='#3fb950', ls=':', lw=1.0, label='Ideal = 1')
line_pue, = ax_pue.plot([], [], color='#58a6ff', lw=2.2, label='PUE')
ax_pue.legend(loc='upper right', fontsize=7, framealpha=0.3)
txt_pue = ax_pue.text(
    0.03, 0.97,
    "PUE   = --\nRe    = --\nNu    = --\nh_avg = --",
    transform=ax_pue.transAxes, fontsize=8, color='#e6edf3', va='top',
    bbox=dict(boxstyle='round,pad=0.35', fc='#161b22', ec='#58a6ff',
              alpha=0.88, lw=1.5))

# ── Multi-objective J ─────────────────────────────────────────────────────────
ax_J.set_title(
    "Multi-Objective  J = w1·PUE + w2·ΔTmax-min − w3·h_avg  [Eq. 11]",
    fontsize=10, pad=5)
ax_J.set_xlabel("Frame", fontsize=8)
ax_J.set_ylabel("J  (lower = better)", fontsize=8)
ax_J.grid(True, alpha=0.25)
line_J,    = ax_J.plot([], [], color='#bc8cff', lw=2.2, label='J total')
line_Jpue, = ax_J.plot([], [], color='#58a6ff', lw=1.2, ls='--', label='w1·PUE')
line_JdT,  = ax_J.plot([], [], color='#f0e040', lw=1.2, ls='--', label='w2·ΔT')
line_Jh,   = ax_J.plot([], [], color='#3fb950', lw=1.2, ls='--', label='-w3·h̃')
ax_J.legend(loc='upper left', fontsize=7, framealpha=0.3)
txt_J = ax_J.text(
    0.55, 0.97, "J=--\nΔTstack=--\nh=--  Pr=--",
    transform=ax_J.transAxes, fontsize=8, color='#e6edf3', va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='#161b22', ec='#bc8cff', alpha=0.88))

# ── Constraint status ─────────────────────────────────────────────────────────
ax_con.set_title("Constraint Status  [§3.4]", fontsize=10, pad=5)
ax_con.set_xlim(0, 1);  ax_con.set_ylim(0, 5);  ax_con.axis('off')
CONSTS = [
    ("T_GPU,max < 85 °C",    '#ffa657'),
    ("ΔT_stack  < 15 K",     '#f0e040'),
    ("Re > 2300 (turbulent)",'#58a6ff'),
    ("PUE < 1.15",           '#ff7b72'),
    ("P_pump / P_IT < 5%",   '#3fb950'),
]
con_texts = []
for i, (lbl, col) in enumerate(CONSTS):
    ax_con.add_patch(plt.Rectangle(
        (0.02, 4.0-i*0.9), 0.96, 0.75,
        fc='#161b22', ec=col, lw=1.2, zorder=1))
    ct = ax_con.text(0.50, 4.37-i*0.9, f"[?]  {lbl}  →  --",
                     ha='center', va='center', fontsize=8.5,
                     color=col, zorder=2)
    con_texts.append(ct)

# ── Inter-server gradient panel ───────────────────────────────────────────────
ax_inter.set_title(
    "Inter-Server Thermal Analysis  (S1 · GAP · S2 · ΔT_inter)",
    fontsize=10, pad=5)
ax_inter.set_xlabel("Frame", fontsize=8)
ax_inter.set_ylabel("°C", fontsize=8)
ax_inter.grid(True, alpha=0.25)
line_ts1,   = ax_inter.plot([], [], color='#ff7b72', lw=1.8, label='T_S1 avg')
line_ts2,   = ax_inter.plot([], [], color='#ffa657', lw=1.8, label='T_S2 avg')
line_tgap,  = ax_inter.plot([], [], color='#3fb950', lw=1.4, ls=':', label='T_gap avg')
line_dti,   = ax_inter.plot([], [], color='#f0e040', lw=1.8, ls='--', label='ΔT_inter (S2-S1)')
ax_inter.legend(loc='upper left', fontsize=7, framealpha=0.3)
txt_inter = ax_inter.text(
    0.56, 0.97,
    "T_S1=--\nT_S2=--\nT_gap=--\nΔT_inter=--",
    transform=ax_inter.transAxes, fontsize=8, color='#e6edf3', va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='#161b22', ec='#ffa657', alpha=0.88))

# ── Turbulent viscosity panel ─────────────────────────────────────────────────
ax_turb.set_title(
    "Turbulent Viscosity ν_t (Smagorinsky SGS)  —  mean over domain",
    fontsize=10, pad=5)
ax_turb.set_xlabel("Frame", fontsize=8)
ax_turb.set_ylabel("ν_t  [m²/s]", fontsize=8)
ax_turb.grid(True, alpha=0.25)
line_nut, = ax_turb.plot([], [], color='#bc8cff', lw=2.0, label='ν_t mean')
ax_turb.legend(loc='upper left', fontsize=7, framealpha=0.3)
txt_turb = ax_turb.text(
    0.56, 0.97, "ν_t_mean=--",
    transform=ax_turb.transAxes, fontsize=8, color='#e6edf3', va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='#161b22', ec='#bc8cff', alpha=0.88))

# ── Sliders ───────────────────────────────────────────────────────────────────
sk_kw = dict(color='#1f6feb', track_color='#21262d')
s_K   = Slider(ax_sk,  'K  (flow)',   0.10, 2.00, valinit=0.50, **sk_kw)
s_l1  = Slider(ax_sl1, 'Load S1',     0.50, 3.00, valinit=1.00, **sk_kw)
s_l2  = Slider(ax_sl2, 'Load S2',     0.50, 3.00, valinit=1.20, **sk_kw)
s_spd = Slider(ax_spd, 'Sim Speed',   1,    60,   valinit=20,
               valstep=1, **sk_kw)
s_w1  = Slider(ax_sw1, 'w1  (PUE)',   0.00, 2.00, valinit=1.00, **sk_kw)
s_w2  = Slider(ax_sw2, 'w2  (ΔT)',    0.00, 2.00, valinit=0.50, **sk_kw)
s_w3  = Slider(ax_sw3, 'w3  (h_avg)', 0.00, 2.00, valinit=0.30, **sk_kw)
for ax_ in (ax_sk, ax_sl1, ax_sl2, ax_spd, ax_sw1, ax_sw2, ax_sw3):
    ax_.set_facecolor('#161b22')

# ─────────────────────────────────────────────────────────────────────────────
# 10. HISTORY BUFFERS
# ─────────────────────────────────────────────────────────────────────────────
HIST = 300
hist = {k: [] for k in (
    'frame','t_in','t_out','dt','pue',
    'J','Jpue','JdT','Jh',
    'ts1','ts2','tgap','dti','nut')}

# ─────────────────────────────────────────────────────────────────────────────
# 11. ANIMATION UPDATE
# ─────────────────────────────────────────────────────────────────────────────
def update(frame):
    K              = s_K.val
    load1, load2   = s_l1.val, s_l2.val
    w1, w2, w3     = s_w1.val, s_w2.val, s_w3.val
    SPF            = int(s_spd.val)

    for _ in range(SPF):
        step_physics(K, load1, load2)

    m    = compute_metrics(K, load1, load2)
    J    = compute_J(m, w1, w2, w3)
    Jpue =  w1 * m['PUE']
    JdT  =  w2 * m['dT_stack']
    Jh   = -w3 * (m['h_avg'] / 1e4)

    t_in  = float(T[:, 0].mean())
    t_out = float(T[:, -1].mean())
    dt_io = t_out - t_in
    T_max = float(T[srv_mask].max())

    # -- Heatmap --
    img_temp.set_data(T)
    img_temp.set_clim(max(T_IN-1, T.min()), min(T_IN+60, T.max()+1))

    # -- Gradient --
    gy, gx = np.gradient(T, dy, dx)
    gm = np.hypot(gx, gy)
    img_grad.set_data(gm)
    img_grad.set_clim(0, max(1.0, gm.max()*0.9))

    # -- Quiver (normalised) --
    U_q = u[::qs, ::qs]
    V_q = v[::qs, ::qs]
    spd_max = max(float(np.hypot(U_q, V_q).max()), 1e-8)
    quiver.set_UVC(U_q/spd_max, V_q/spd_max)

    # -- PUE overlay on heatmap --
    pue_col = '#ff7b72' if m['PUE'] > 1.15 else '#58a6ff'
    pue_overlay.set_text(f"PUE = {m['PUE']:.4f}")
    pue_overlay.set_color(pue_col)
    pue_overlay.get_bbox_patch().set_edgecolor(pue_col)
    tin_overlay.set_text(
        f"T_in  = {t_in -273.15:+.2f} °C\n"
        f"T_out = {t_out-273.15:+.2f} °C\n"
        f"ΔT    = {dt_io:+.3f} K")

    # -- History --
    hist['frame'].append(frame)
    hist['t_in'].append(t_in);   hist['t_out'].append(t_out)
    hist['dt'].append(dt_io);    hist['pue'].append(m['PUE'])
    hist['J'].append(J);         hist['Jpue'].append(Jpue)
    hist['JdT'].append(JdT);     hist['Jh'].append(Jh)
    hist['ts1'].append(m['T_s1_avg']-273.15)
    hist['ts2'].append(m['T_s2_avg']-273.15)
    hist['tgap'].append(m['T_gap_avg']-273.15)
    hist['dti'].append(m['dT_inter'])
    hist['nut'].append(m['nu_t_mean'])
    if len(hist['frame']) > HIST:
        for lst in hist.values(): del lst[0]
    xs = hist['frame']

    def _xlim(ax): ax.set_xlim(xs[0], xs[-1]+5) if len(xs)>1 else None

    # T_in / T_out / ΔT
    line_in.set_data(xs, hist['t_in'])
    line_out.set_data(xs, hist['t_out'])
    line_dt.set_data(xs, hist['dt'])
    _xlim(ax_loop)
    if len(xs)>1:
        ax_loop.set_ylim(
            min(min(hist['t_in']),min(hist['dt']))-1,
            max(hist['t_out'])+3)
    txt_loop.set_text(
        f"T_in  = {t_in -273.15:.2f} °C\n"
        f"T_out = {t_out-273.15:.2f} °C\n"
        f"ΔT    = {dt_io:+.3f} K")

    # PUE
    line_pue.set_data(xs, hist['pue'])
    _xlim(ax_pue)
    if len(xs)>1:
        ax_pue.set_ylim(0.95, max(max(hist['pue'])+0.05, 1.20))
    txt_pue.set_text(
        f"PUE   = {m['PUE']:.4f}\n"
        f"Re    = {m['Re']:.0f}\n"
        f"Nu    = {m['Nu']:.1f}\n"
        f"h_avg = {m['h_avg']:.1f} W/m²K")

    # J
    line_J.set_data(xs, hist['J'])
    line_Jpue.set_data(xs, hist['Jpue'])
    line_JdT.set_data(xs, hist['JdT'])
    line_Jh.set_data(xs, hist['Jh'])
    _xlim(ax_J)
    if len(xs)>1:
        all_J = hist['J']+hist['Jpue']+hist['JdT']+hist['Jh']
        ax_J.set_ylim(min(all_J)-0.05, max(all_J)+0.1)
    txt_J.set_text(
        f"J      = {J:.4f}\n"
        f"ΔTstack= {m['dT_stack']:.2f} K\n"
        f"h={m['h_avg']:.1f}  Pr={m['Pr']:.2f}")

    # Constraints
    checks = [
        (T_max-273.15 < 85.0,
         f"T_GPU,max < 85 °C   →  {T_max-273.15:.1f} °C"),
        (m['dT_stack'] < 15.0,
         f"ΔT_stack  < 15 K    →  {m['dT_stack']:.2f} K"),
        (m['Re'] > 2300,
         f"Re > 2300           →  {m['Re']:.0f}"),
        (m['PUE'] < 1.15,
         f"PUE < 1.15          →  {m['PUE']:.4f}"),
        (m['P_pump']/max(m['P_IT'],1e-9) < 0.05,
         f"P_pump/P_IT < 5%%   →  "
         f"{100*m['P_pump']/max(m['P_IT'],1e-9):.2f}%%"),
    ]
    for (ok, lbl), ct in zip(checks, con_texts):
        ct.set_text(("[OK]  " if ok else "[!!]  ") + lbl)
        ct.set_color('#3fb950' if ok else '#ff7b72')

    # Inter-server gradient
    line_ts1.set_data(xs, hist['ts1'])
    line_ts2.set_data(xs, hist['ts2'])
    line_tgap.set_data(xs, hist['tgap'])
    line_dti.set_data(xs, hist['dti'])
    _xlim(ax_inter)
    if len(xs)>1:
        all_it = hist['ts1']+hist['ts2']+hist['tgap']+hist['dti']
        ax_inter.set_ylim(min(all_it)-1, max(all_it)+2)
    txt_inter.set_text(
        f"T_S1    = {m['T_s1_avg']-273.15:.2f} °C\n"
        f"T_S2    = {m['T_s2_avg']-273.15:.2f} °C\n"
        f"T_gap   = {m['T_gap_avg']-273.15:.2f} °C\n"
        f"ΔT_inter= {m['dT_inter']:+.3f} K")

    # Turbulent viscosity
    line_nut.set_data(xs, hist['nut'])
    _xlim(ax_turb)
    if len(xs)>1:
        ax_turb.set_ylim(0, max(hist['nut'])*1.3 + 1e-9)
    txt_turb.set_text(f"ν_t_mean = {m['nu_t_mean']:.3e} m²/s")

    # CMD – single overwriting line
    pf = " !! PUE OVER LIMIT" if m['PUE']>1.15 else "                  "
    sys.stdout.write(
        f"\r[{frame:05d}]  "
        f"T_in={t_in-273.15:6.2f}°C  "
        f"T_out={t_out-273.15:6.2f}°C  "
        f"ΔT={dt_io:+5.3f}K  "
        f"PUE={m['PUE']:.4f}{pf}  "
        f"Re={m['Re']:6.0f}  "
        f"ΔT_inter={m['dT_inter']:+.3f}K  "
        f"ν_t={m['nu_t_mean']:.2e}  "
        f"SPF={SPF}  K={K:.2f}  L1={load1:.2f}  L2={load2:.2f}"
    )
    sys.stdout.flush()

    return (img_temp, img_grad, quiver,
            pue_overlay, tin_overlay,
            line_in, line_out, line_dt,
            line_pue, line_J, line_Jpue, line_JdT, line_Jh,
            line_ts1, line_ts2, line_tgap, line_dti, line_nut)

# ─────────────────────────────────────────────────────────────────────────────
# 12. RUN
# ─────────────────────────────────────────────────────────────────────────────
ani = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)

print("=" * 110)
print("  Immersion Cooling Simulation  |  Methyl Myristate")
print("  Turbulence: Smagorinsky SGS  |  Full 2D strain-rate tensor |  "
      "Turbulent thermal diffusivity in energy")
print("  Inter-server: independent Load1/Load2 sliders  |  "
      "Gap-zone T tracked  |  ΔT_inter displayed")
print()
print("  SLIDERS:")
print("    K         – inlet velocity multiplier")
print("    Load S1   – server 1 heat load  (independent)")
print("    Load S2   – server 2 heat load  (independent)")
print("    Sim Speed – physics steps per frame  (1–60, drag right = faster)")
print("    w1/w2/w3  – multi-objective weights for J  [Eq. 11]")
print("=" * 110)
print()

plt.show()
print()