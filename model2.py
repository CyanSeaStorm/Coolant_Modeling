"""
Immersion-Cooled Server CFD — Methyl Myristate
═══════════════════════════════════════════════════════════════════════════════
Physics  (ALL equations from mathematical framework implemented)
───────
Eq.1  Continuity          ∇·u = 0
Eq.2  Navier-Stokes       ρ(T)(∂u/∂t + u·∇u) = −∇p + μeff∇²u + ρ(T)g β(T)ΔT ĵ
Eq.3  Energy              ρcp(∂T/∂t + u·∇T) = ∇·(keff∇T) + Φ(T)
                          Φ = μ(T)[2(∂u/∂x)²+2(∂v/∂y)²+(∂u/∂y+∂v/∂x)²]
Eq.4  Conjugate solid     ∇·(ks∇Ts) = 0
Eq.5  Dittus-Boelter      Nu = 0.023 Re^0.8 Pr(T)^0.4 (μb/μw)^0.14
Eq.6  CDU ODE             dTb/dt=[ṁcp(Tb)(Tin−Tb)+Q_IT−UA(Tb−Tcool)]/(Mcp(Tb))
Eq.7  Darcy-Weisbach      ΔP = [f(Re)·L/Dh + ΣK]·ρ(T̄)u²/2
Eq.8  PUE                 (PIT+Ppump+Pchiller)/PIT
        Ppump    = V̇ΔP / ηp
        Pchiller = (PIT+Ppump) / COP
Eq.9  k-ω SST (NEW)       Menter 1994 — transitional turbulence
        ∂k/∂t  + u·∇k  = Pk − β*kω  + ∇·[(ν+σk νt)∇k]
        ∂ω/∂t  + u·∇ω  = α(ω/k)Pk − βω² + ∇·[(ν+σω νt)∇ω] + CDkω
        νt = a1·k / max(a1·ω, |Ω|·F2)
        μeff = μ(T) + ρ(T)·νt
        keff = k(T) + ρcp·νt/Prt    (turbulent thermal diffusion)
Vorticity                 ∂ω/∂t = −u·∇ω + ∇·(νeff∇ω) − gβ(T)∂T/∂x

ALL fluid properties are LOCAL 2D fields evaluated from T(x,y) each frame.
No property is a constant during the simulation.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from collections import deque
from scipy.ndimage import distance_transform_edt
import time as _time

# ═════════════════════════════════════════════════════════════════════════════
# 1.  REFERENCE CONSTANTS  (properties at T_REF = 25 °C)
# ═════════════════════════════════════════════════════════════════════════════
T_REF     = 25.0

# Methyl Myristate at 25 °C
RHO0      = 865.0   # Density              [kg/m³]
MU0       = 0.0051  # Dynamic viscosity    [Pa·s]
CP0       = 2080.0  # Specific heat        [J/kg·K]
K0        = 0.145   # Thermal conductivity [W/m·K]
BETA0     = 0.00085 # Thermal expansion    [1/K]
G_ACC     = 9.81    # Gravity              [m/s²]

# Server solid (aluminium)
K_SOLID   = 200.0
CP_SOLID  = 900.0

# CDU / chiller
T_IN      = 25.0
T_COOL    = 20.0
UA_CDU    = 800.0
COP_CHILL = 3.5
ETA_PUMP  = 0.75

# Server heat loads
Q1_W      = 2000.0
Q2_W      = 2500.0
V_LPM     = 3.5

# Time-varying workload scale (updated every frame in update())
# Represents server duty-cycle variation — drives meaningful PUE changes
Q_SCALE   = 1.0

# ── k-ω SST Model Constants  (Menter 1994) ───────────────────────────────
KW_SIGMA_K1  = 0.85;    KW_SIGMA_K2  = 1.00
KW_SIGMA_W1  = 0.50;    KW_SIGMA_W2  = 0.856
KW_BETA1     = 0.075;   KW_BETA2     = 0.0828
KW_BETA_STAR = 0.09
KW_ALPHA1    = 5.0/9.0; KW_ALPHA2    = 0.44
KW_A1        = 0.31
KW_PR_T      = 0.85     # turbulent Prandtl number


# ── Temperature-dependent property functions (scalar or array) ────────────
def rho_f(T):
    """ρ(T) [kg/m³]"""
    return np.clip(RHO0 - 0.71 * (T - T_REF), 800.0, 900.0)

def mu_f(T):
    """μ(T) [Pa·s]"""
    return np.clip(MU0 * np.exp(-0.030 * (T - T_REF)), 0.002, 0.015)

def cp_f(T):
    """cp(T) [J/kg·K]"""
    return np.clip(CP0 + 2.8 * (T - T_REF), 1900.0, 2500.0)

def k_f(T):
    """k(T) [W/m·K]"""
    return np.clip(K0 - 0.00018 * (T - T_REF), 0.120, 0.160)

def beta_f(T):
    """β(T) [1/K]"""
    return np.clip(BETA0 * (1.0 + 0.001 * (T - T_REF)), 0.0007, 0.001)

def nu_f(T):
    """ν(T) = μ/ρ [m²/s]"""
    return mu_f(T) / rho_f(T)

def pr_f(T):
    """Pr(T) = μcp/k"""
    return mu_f(T) * cp_f(T) / k_f(T)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════
L_CH, W_CH  = 0.6, 0.3
NX,   NY    = 100, 50
u = np.zeros((NY, NX))
v = np.zeros((NY, NX))
p = np.zeros((NY, NX))
SIM_DT      = 0.01
BURN_THRESH = 85.0

S1_XY  = (0.15, 0.15)
S2_XY  = (0.40, 0.15)
S_SIZE = (0.12, 0.08)
SERVERS = list(zip([S1_XY, S2_XY], [Q1_W, Q2_W]))

dx = L_CH / NX
dy = W_CH / NY
x  = np.linspace(0, L_CH, NX)
y  = np.linspace(0, W_CH, NY)
X, Y = np.meshgrid(x, y)

V_DOT    = V_LPM / 60_000
CH_DEPTH = 0.05
DH       = 2 * W_CH * CH_DEPTH / (W_CH + CH_DEPTH)
M_FLUID  = RHO0 * L_CH * W_CH * CH_DEPTH
K_FIT    = 2.5

# ═════════════════════════════════════════════════════════════════════════════
# 3.  SOLID / FLUID / SURFACE MASKS
# ═════════════════════════════════════════════════════════════════════════════
SOLID = np.zeros((NY, NX), dtype=bool)
for (sx, sy), _ in SERVERS:
    SOLID |= ((X >= sx - S_SIZE[0]/2) & (X <= sx + S_SIZE[0]/2) &
              (Y >= sy - S_SIZE[1]/2) & (Y <= sy + S_SIZE[1]/2))
FLUID = ~SOLID

SURFACE = np.zeros((NY, NX), dtype=bool)
for i in range(1, NY-1):
    for j in range(1, NX-1):
        if FLUID[i, j] and (SOLID[i-1,j] or SOLID[i+1,j] or
                            SOLID[i,j-1] or SOLID[i,j+1]):
            SURFACE[i, j] = True

SERVER_HALO = []
for (sx, sy), _ in SERVERS:
    hm = (SURFACE
          & (X >= sx - S_SIZE[0]/2 - dx) & (X <= sx + S_SIZE[0]/2 + dx)
          & (Y >= sy - S_SIZE[1]/2 - dy) & (Y <= sy + S_SIZE[1]/2 + dy))
    SERVER_HALO.append(hm)

def apply_solid_bc(field, fill=0.0):
    field[SOLID] = fill
    return field

# ── Wall distance field for k-ω SST (precomputed once at startup) ─────────
_wdist_cells = distance_transform_edt(FLUID)
WALL_DIST    = np.maximum(_wdist_cells * min(dx, dy), 1e-4)
WALL_DIST[SOLID] = 0.0

# ═════════════════════════════════════════════════════════════════════════════
# 4.  PERSISTENT SOLVER STATE
# ═════════════════════════════════════════════════════════════════════════════
OMEGA_MAX   = 30.0
omega_field = np.zeros((NY, NX))

T_solid = np.full((NY, NX), T_IN + 15.0)
Tb_cdu  = float(T_IN) 
# Fluid temperature field (persistent)
T_field = np.full((NY, NX), T_IN)
T_field[SOLID] = T_solid[SOLID]

# k-ω SST persistent fields
_nu0       = MU0 / RHO0
_omg_wbc   = 60.0 * _nu0 / (KW_BETA1 * min(dx, dy)**2)
k_turb     = np.full((NY, NX), 1e-4)
omega_sst  = np.full((NY, NX), _omg_wbc * 0.05)
k_turb[SOLID]    = 0.0
omega_sst[SOLID] = _omg_wbc

# ═════════════════════════════════════════════════════════════════════════════
# 5.  DASHBOARD LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
plt.style.use('dark_background')
fig = plt.figure(figsize=(22, 14), facecolor='#0a0a0a')
fig.suptitle(
    "Immersion-Cooled Server CFD  ·  Methyl Myristate  ·  "
    "k-ω SST  +  T-Dependent  ρ μ cp k β  +  Full PDE Set  (Eqs 1-9)",
    color='#00e5ff', fontsize=11, fontweight='bold', y=0.998)

gs = gridspec.GridSpec(3, 3,
                       height_ratios=[1.45, 1.15, 1.30],
                       hspace=0.50, wspace=0.32)

ax_temp  = fig.add_subplot(gs[0, 0])
ax_turb  = fig.add_subplot(gs[0, 1])   # k-ω SST νt/ν field
ax_eddy  = fig.add_subplot(gs[0, 2])
ax_vort  = fig.add_subplot(gs[1, 0])
ax_vel   = fig.add_subplot(gs[1, 1])
ax_ros   = fig.add_subplot(gs[1, 2])
ax_temps = fig.add_subplot(gs[2, 0:2])
ax_pue   = fig.add_subplot(gs[2, 2])

state       = {"t": 0.0, "step": 0}
pue_history = []
TBUF = 600
buf = {k: deque(maxlen=TBUF) for k in
       ['t', 'pue', 'pue_avg', 't_in', 't_out', 'delta_t', 'ts1', 'ts2']}

_ros_last = _time.monotonic()
ROS_HZ    = 10.0


# ═════════════════════════════════════════════════════════════════════════════
# 6.  CFD SOLVER
# ═════════════════════════════════════════════════════════════════════════════
# def compute_frame():
#     global p, u, v, T_field
#     global omega_field, T_solid, Tb_cdu, k_turb, omega_sst

#     Q_IT  = (Q1_W + Q2_W) * Q_SCALE   # scaled by current workload factor
#     m_dot = rho_f(T_IN) * V_DOT
#     U_inf = V_DOT / (W_CH * CH_DEPTH)

#     # ────────────────────────────────────────────────────────────────────────
#     # A.  TEMPERATURE FIELD
#     # ────────────────────────────────────────────────────────────────────────
#     # T_field = np.full((NY, NX), float(T_IN))
#     # for (sx, sy), Q in SERVERS:
#     #     dx_s  = X - sx
#     #     dy_s  = Y - sy
#     #     sigma = 0.025 + 0.06 * np.maximum(0.0, dx_s)
#     #     plume = (Q * Q_SCALE / 500.0) * np.exp(-4.5 * np.maximum(0.0, dx_s)) \
#     #           * np.exp(-(dy_s**2) / (2.0 * sigma**2 + 1e-12))
#     #     heat  = np.where(dx_s >= -S_SIZE[0]/2, plume * 15.0, 0.0)
#     #     heat[SOLID] = 0.0
#     #     T_field += heat
#     T_field[SOLID] = T_solid[SOLID]
#     T_field = np.clip(T_field, T_IN - 2.0, T_IN + 120.0)
#     # Temperature difference driving buoyancy
#     dT = T_field - T_IN


#     # ────────────────────────────────────────────────────────────────────────
#     # B.  LOCAL PROPERTY FIELDS  (all 2D, T-dependent)
#     # ────────────────────────────────────────────────────────────────────────
#     RHO  = rho_f(T_field)
#     MU   = mu_f(T_field)
#     CP   = cp_f(T_field)
#     K    = k_f(T_field)
#     BETA = beta_f(T_field)
#     NU   = MU / RHO
#     PR   = MU * CP / K

#     rho_mean = float(np.nanmean(RHO[FLUID]))
#     nu_mean  = float(np.nanmean(NU[FLUID]))
#     mu_mean  = float(np.nanmean(MU[FLUID]))
#     cp_mean  = float(np.nanmean(CP[FLUID]))

#     # ────────────────────────────────────────────────────────────────────────
#     # C.  VISCOUS DISSIPATION  Φ(T)  (Eq.3)
#     # ────────────────────────────────────────────────────────────────────────
#     # u_pre = np.full((NY, NX), U_inf); u_pre[SOLID] = 0.0
#     # v_pre = np.zeros((NY, NX)) 

#     # Use REAL velocity field
#     u_pre = u.copy()
#     v_pre = v.copy()
#     dudx_c = np.gradient(u_pre, dx, axis=1)
#     dudy_c = np.gradient(u_pre, dy, axis=0)
#     dvdx_c = np.gradient(v_pre, dx, axis=1)
#     dvdy_c = np.gradient(v_pre, dy, axis=0)

#     Phi    = MU * (2*dudx_c**2 + 2*dvdy_c**2 + (dudy_c + dvdx_c)**2)
#     Phi[SOLID] = 0.0
#     T_field += (Phi / (RHO * CP + 1e-9)) * SIM_DT * 0.5
#     T_field[SOLID] = T_solid[SOLID]
#     T_field = np.clip(T_field, T_IN - 2.0, T_IN + 120.0)

#     # ────────────────────────────────────────────────────────────────────────
#     # D.  BASE VELOCITY FIELD  (Eq.1 ∇·u=0, Eq.2 NS)
#     # ────────────────────────────────────────────────────────────────────────
#     #u = np.full((NY, NX), U_inf)
#     #v = np.zeros((NY, NX))
#     #for (sx, sy), _ in SERVERS:
#     #    dist_x = X - sx
#     #    dist_y = Y - sy
#     #    bypass = ((np.abs(dist_y) > S_SIZE[1]/2 + dy) &
#     #              (np.abs(dist_x) <= S_SIZE[0]/2 + dx))
#     #    wake   = ((dist_x > 0) & (dist_x < S_SIZE[0]) &
#     #              (np.abs(dist_y) < S_SIZE[1]/1.5))
#     #    u[bypass] *= 1.5
#     #    u[wake]   *= 0.25 * (1.0 - np.exp(-dist_x[wake] * 6.0))
#     #apply_solid_bc(u, 0.0)
#     #apply_solid_bc(v, 0.0)
#     # ─────────────────────────────────────────────────────────────
# # D. NAVIER–STOKES MOMENTUM PREDICTOR
# # Eq.2 without pressure (projection method)
# # ─────────────────────────────────────────────────────────────

#     # Initialize velocity only first time
#     # try:
#     #     u
#     #     v
#     # except NameError:
#     #     u = np.full((NY, NX), U_inf)
#     #     v = np.zeros((NY, NX))

#     # Effective viscosity (laminar + turbulent)
#     MU_EFF = MU + RHO * (k_turb / (omega_sst + 1e-12))
#     NU_EFF = MU_EFF / RHO

#     # ---------- Gradients ----------
#     dudx = np.gradient(u, dx, axis=1)
#     dudy = np.gradient(u, dy, axis=0)
#     dvdx = np.gradient(v, dx, axis=1)
#     dvdy = np.gradient(v, dy, axis=0)   
#     S2 = 2.0*(dudx**2 + dvdy**2) + (dudy + dvdx)**2
#     S = np.sqrt(S2 + 1e-12)
    
#     Omega_mag = np.sqrt((dvdx - dudy)**2 + 1e-12)

#     nu_t = KW_A1 * k_turb / np.maximum(
#         KW_A1 * omega_sst,
#         Omega_mag
#     )

#     nu_t = np.clip(nu_t, 0.0, 50.0 * NU) 
#     Pk = nu_t * S2 

#     Pk = np.minimum(Pk, 10.0 * KW_BETA_STAR * k_turb * omega_sst)
#     # ---------- Convection ----------
#     conv_u = u * dudx + v * dudy
#     conv_v = u * dvdx + v * dvdy

#     # ---------- Diffusion ----------
#     lap_u = (
#         np.gradient(np.gradient(u, dx, axis=1), dx, axis=1) +
#         np.gradient(np.gradient(u, dy, axis=0), dy, axis=0)
#     )

#     lap_v = (
#         np.gradient(np.gradient(v, dx, axis=1), dx, axis=1) +
#         np.gradient(np.gradient(v, dy, axis=0), dy, axis=0)
#     )

#     # diff_u = NU_EFF * lap_u
#     # diff_v = NU_EFF * lap_v
#     diff_u = (
#     np.gradient(NU_EFF * dudx, dx, axis=1) +
#     np.gradient(NU_EFF * dudy, dy, axis=0)
#     )

#     diff_v = (
#         np.gradient(NU_EFF * dvdx, dx, axis=1) +
#         np.gradient(NU_EFF * dvdy, dy, axis=0)
#     )


#     # ---------- Buoyancy ----------
#     #F_b = G_ACC * beta_f(T_field) * (T_field - T_IN)
#     F_b = G_ACC * BETA * dT

#     # ---------- Predictor step ----------
#     u_star = u + SIM_DT * (-conv_u + diff_u)
#     v_star = v + SIM_DT * (-conv_v + diff_v + F_b) 
#     # ------------------------------------------------------------
#     # Continuity RHS  (∇·u*)
#     # ------------------------------------------------------------
#     div_u = (
#         np.gradient(u_star, dx, axis=1) +
#         np.gradient(v_star, dy, axis=0)
#     )

#     rhs = (RHO / SIM_DT) * div_u
#     rhs[SOLID] = 0.0 
#     # ------------------------------------------------------------
#     # Pressure Poisson Equation (Jacobi iterations)
#     # ------------------------------------------------------------
#     for _ in range(40):

#         p[1:-1,1:-1] = (
#             (p[1:-1,2:] + p[1:-1,:-2]) * dy**2 +
#             (p[2:,1:-1] + p[:-2,1:-1]) * dx**2 -
#             rhs[1:-1,1:-1] * dx**2 * dy**2
#         ) / (2 * (dx**2 + dy**2))

#         # Neumann BC (zero gradient walls)
#         p[:,0]  = p[:,1]
#         p[:,-1] = p[:,-2]
#         p[0,:]  = p[1,:]
#         p[-1,:] = p[-2,:]

#         apply_solid_bc(p, 0.0) 
       
       
#     dpdx = np.gradient(p, dx, axis=1)
#     dpdy = np.gradient(p, dy, axis=0) 

#     u = u_star - SIM_DT * dpdx / RHO
#     v = v_star - SIM_DT * dpdy / RHO 

#     apply_solid_bc(u, 0.0)
#     apply_solid_bc(v, 0.0)
#     div_check = np.mean(np.abs(
#         np.gradient(u, dx, axis=1) +
#         np.gradient(v, dy, axis=0)
#     ))   
#     if state["step"] % 50 == 0:
#       print("Divergence:", div_check)  
    
#     # No-slip inside solids
#     u_star[SOLID] = 0.0
#     v_star[SOLID] = 0.0 

#     div_u = (
#     np.gradient(u_star, dx, axis=1) +
#     np.gradient(v_star, dy, axis=0)
#     )
#     dudx_star = np.gradient(u_star, dx, axis=1)
#     dvdy_star = np.gradient(v_star, dy, axis=0)

#     div_u = dudx_star + dvdy_star

# #     rhs = (RHO / SIM_DT) * div_u 

# #     # ─────────────────────────────────────────────
# # # Solve ∇²p = rhs   (Jacobi iterations)
# # # ─────────────────────────────────────────────
# #     for _ in range(40):   # 30–80 typical
# #         p[1:-1,1:-1] = (
# #             (p[1:-1,2:] + p[1:-1,:-2]) * dy**2 +
# #             (p[2:,1:-1] + p[:-2,1:-1]) * dx**2 -
# #             rhs[1:-1,1:-1] * dx**2 * dy**2
# #         ) / (2*(dx**2 + dy**2))

#         # Neumann BCs (zero normal gradient)
#     p[:,0]  = p[:,1]
#     p[:,-1] = p[:,-2]
#     p[0,:]  = p[1,:]
#     p[-1,:] = p[-2,:]

#         # solid region pressure smoothing
#     p[SOLID] = np.mean(p[FLUID])
#     # ─────────────────────────────────────────────────────────────
# # 
# # ─────────────────────────────────────────────────────────────
#     # E. PRESSURE POISSON SOLVER
#     # ∇²p = (ρ/Δt) ∇·u*
#     # ─────────────────────────────────────────────────────────────

#     # divergence of predictor velocity
#     # div_u = (
#     #     np.gradient(u_star, dx, axis=1) +
#     #     np.gradient(v_star, dy, axis=0)
#     # )

#     # #rhs = (RHO / SIM_DT) * div_u
#     # rhs = (RHO / SIM_DT) * div_u

#     # # Jacobi iterations
#     # for _ in range(60):
#     #     p[1:-1,1:-1] = (
#     #         (p[1:-1,2:] + p[1:-1,:-2]) * dy**2 +
#     #         (p[2:,1:-1] + p[:-2,1:-1]) * dx**2 -
#     #         rhs[1:-1,1:-1] * dx**2 * dy**2
#     #     ) / (2*(dx**2 + dy**2))

#     #     # Neumann BC (zero normal gradient)
#     #     p[:,0]  = p[:,1]
#     #     p[:,-1] = p[:,-2]
#     #     p[0,:]  = p[1,:]
#     #     p[-1,:] = p[-2,:]

#     # # pressure gradients
#     # dpdx = np.gradient(p, dx, axis=1)
#     # dpdy = np.gradient(p, dy, axis=0)

#     # # velocity correction
#     # u = u_star - (SIM_DT / RHO) * dpdx
#     # v = v_star - (SIM_DT / RHO) * dpdy

#     # # enforce solid BC again
#     # u[SOLID] = 0.0
#     # # v[SOLID] = 0.0 
#   # ─────────────────────────────────────────────────────────────
# # E. PRESSURE POISSON SOLVER (Variable-density projection)
# # ∇·((1/ρ)∇p) = (1/Δt) ∇·u*
# # ─────────────────────────────────────────────────────────────

#     div_u = (
#         np.gradient(u_star, dx, axis=1) +
#         np.gradient(v_star, dy, axis=0)
#     )

#     rhs = div_u / SIM_DT

#     inv_rho = 1.0 / (RHO + 1e-12)

#     # Jacobi iterations
#     for _ in range(60):

#         p_new = p.copy()

#         p_new[1:-1,1:-1] = (
#             (
#                 inv_rho[1:-1,2:] * p[1:-1,2:] +
#                 inv_rho[1:-1,:-2] * p[1:-1,:-2]
#             ) * dy**2
#             +
#             (
#                 inv_rho[2:,1:-1] * p[2:,1:-1] +
#                 inv_rho[:-2,1:-1] * p[:-2,1:-1]
#             ) * dx**2
#             - rhs[1:-1,1:-1] * dx**2 * dy**2
#         ) / (
#             (inv_rho[1:-1,2:] + inv_rho[1:-1,:-2]) * dy**2 +
#             (inv_rho[2:,1:-1] + inv_rho[:-2,1:-1]) * dx**2
#             + 1e-12
#         )

#         # Neumann BC (zero normal gradient)
#         p_new[:,0]  = p_new[:,1]
#         p_new[:,-1] = p_new[:,-2]
#         p_new[0,:]  = p_new[1,:]
#         p_new[-1,:] = p_new[-2,:]

#         # reference pressure (removes drift)
#         #p_new Pk-= np.mean(p_new)

#         p[:] = p_new    
#     apply_solid_bc(u,0.0)
#     apply_solid_bc(v,0.0)

#     u[:,0] = U_inf
#     v[:,0] = 0.0     

#     # inlet
#     u[:,0] = U_inf
#     v[:,0] = 0.0

#     # outlet
#     u[:,-1] = u[:,-2]
#     v[:,-1] = v[:,-2]
#     p[:,-1] = 0.0
#     # ─────────────────────────────────────────────
#     # Velocity correction (projection)
#     # ─────────────────────────────────────────────
#     dpdx = np.gradient(p, dx, axis=1)
#     dpdy = np.gradient(p, dy, axis=0)

#     # u = u_star - SIM_DT * dpdx / RHO
#     # v = v_star - SIM_DT * dpdy / RHO
#     # ─────────────────────────────────────────────────────────────
#     # F. VELOCITY CORRECTION
#     # u = u* − Δt/ρ ∇p
#     # ─────────────────────────────────────────────────────────────

#     dpdx = np.gradient(p, dx, axis=1)
#     dpdy = np.gradient(p, dy, axis=0)

#     u = u_star - SIM_DT * dpdx / (RHO + 1e-12)
#     v = v_star - SIM_DT * dpdy / (RHO + 1e-12)

#     # enforce no-slip again
#     u[SOLID] = 0.0
#     v[SOLID] = 0.0  
#     apply_solid_bc(u, 0.0)
#     apply_solid_bc(v, 0.0)
#     # =====================================================
#     # INLET BOUNDARY CONDITION  (MANDATORY)
#     # =====================================================
#     u[:, 0] = U_inf     # fluid enters domain
#     v[:, 0] = 0.0  
#     # =====================================================
#     # OUTLET (zero-gradient / open boundary)
#     # =====================================================
#     u[:, -1] = u[:, -2]
#     v[:, -1] = v[:, -2]
#     p[:, -1] = 0.0      # reference pressure
#     # =====================================================
#     # WALLS (no-slip)
#     # =====================================================
#     u[0, :]  = 0.0
#     u[-1, :] = 0.0
#     v[0, :]  = 0.0
#     v[-1, :] = 0.0

#     # solids
#     u[SOLID] = 0.0
#     v[SOLID] = 0.0 
#     alpha_eff = (K + RHO*CP*(k_turb/(omega_sst+1e-12))/KW_PR_T) / (RHO*CP)

#     dTdx = np.gradient(T_field, dx, axis=1)
#     dTdy = np.gradient(T_field, dy, axis=0)

#     conv_T = u*dTdx + v*dTdy

#     lap_T = (
#         np.gradient(np.gradient(T_field, dx, axis=1), dx, axis=1) +
#         np.gradient(np.gradient(T_field, dy, axis=0), dy, axis=0)
#     )

#     #T_field += SIM_DT * (-conv_T + alpha_eff * lap_T)

 
#     # ─────────────────────────────────────────────────────────────
# # ENERGY EQUATION (Eq.3)
# # ρcp(∂T/∂t + u·∇T) = ∇·(keff∇T) + Φ
# # ───────────────────────────────────────────────────────────── 

#     # ------------------------------------------------------------
#     # ENERGY EQUATION (Fluid)
#     # ------------------------------------------------------------

#     alpha = K / (RHO * CP)

#     dTdx = np.gradient(T_field, dx, axis=1)
#     dTdy = np.gradient(T_field, dy, axis=0)

#     lapT = (
#         np.gradient(np.gradient(T_field, dx, axis=1), dx, axis=1) +
#         np.gradient(np.gradient(T_field, dy, axis=0), dy, axis=0)
#     )

#     T_field[FLUID] += SIM_DT * (
#         - u[FLUID]*dTdx[FLUID]
#         - v[FLUID]*dTdy[FLUID]
#         + alpha[FLUID]*lapT[FLUID]
#     ) 

#         # outlet (right boundary)
#     T_field[:, -1] = T_field[:, -2] 
    

# # turbulent thermal diffusivity
#     nu_t = k_turb / (omega_sst + 1e-12)
#     alpha_eff = (K / (RHO * CP)) + nu_t / KW_PR_T

#     # temperature gradients
#     dTdx = np.gradient(T_field, dx, axis=1)
#     dTdy = np.gradient(T_field, dy, axis=0)

#     # advection
#     adv_T = u * dTdx + v * dTdy

#     # diffusion
#     diff_T = (
#         np.gradient(alpha_eff * dTdx, dx, axis=1) +
#         np.gradient(alpha_eff * dTdy, dy, axis=0)
#     )

#     # server volumetric heating
#     Q_src = np.zeros_like(T_field)
#     for halo, (_, Q) in zip(SERVER_HALO, SERVERS):
#         Q_src[halo] += Q * Q_SCALE / (dx * dy * CH_DEPTH)

#     Q_src /= (RHO * CP + 1e-12)

#     # time update
#     T_field += SIM_DT * (-adv_T + diff_T + Q_src)

#     # enforce conjugate solid temperature
#     T_field[SOLID] = T_solid[SOLID]
#     T_field = np.clip(T_field, T_IN-5, T_IN+120)

#       # IMPORTANT — use persistent pressure field

# #     div_u = (
# #         np.gradient(u_star, dx, axis=1) +
# #         np.gradient(v_star, dy, axis=0)
# #     )

# #     rhs = (RHO / SIM_DT) * div_u

# #     for _ in range(60):

# #         p_new = p.copy()

# #         p_new[1:-1,1:-1] = (
# #             (p[1:-1,2:] + p[1:-1,:-2]) * dy**2 +
# #             (p[2:,1:-1] + p[:-2,1:-1]) * dx**2 -
# #             rhs[1:-1,1:-1] * dx**2 * dy**2
# #         ) / (2*(dx**2 + dy**2))

# #         # Neumann BC
# #         p_new[:,0]  = p_new[:,1]
# #         p_new[:,-1] = p_new[:,-2]
# #         p_new[0,:]  = p_new[1,:]
# #         p_new[-1,:] = p_new[-2,:]

# #         p_new[SOLID] = 0.0
# #         p = p_new


# # # ─────────────────────────────────────────────────────────────
# # # F. VELOCITY PROJECTION
# # # u = u* − dt/ρ ∇p
# # # ─────────────────────────────────────────────────────────────

# #     dpdx = np.gradient(p, dx, axis=1)
# #     dpdy = np.gradient(p, dy, axis=0)

# #     u = u_star - (SIM_DT / RHO) * dpdx
# #     v = v_star - (SIM_DT / RHO) * dpdy

# #     u[SOLID] = 0.0
# #     v[SOLID] = 0.0

#     # ────────────────────────────────────────────────────────────────────────
#     # D2.  k-ω SST TURBULENCE MODEL  (Eq.9 — Menter 1994)
#     #
#     #   ∂k/∂t  + u·∇k  = Pk − β*kω  + ∇·[(ν+σk νt)∇k]
#     #   ∂ω/∂t  + u·∇ω  = α(ω/k)Pk − βω² + ∇·[(ν+σω νt)∇ω] + CDkω
#     #   νt = a1·k / max(a1·ω, |Ω|·F2)
#     #   F1 blends inner k-ω layer ↔ outer k-ε layer via wall distance d
#     # ────────────────────────────────────────────────────────────────────────

#     # Strain-rate magnitude  S = √(2 Sij Sij)
#     dudx_s = np.gradient(u, dx, axis=1)
#     dudy_s = np.gradient(u, dy, axis=0)
#     dvdx_s = np.gradient(v, dx, axis=1)
#     dvdy_s = np.gradient(v, dy, axis=0)
#     S_mag  = np.sqrt(np.maximum(
#         2*(dudx_s**2 + dvdy_s**2) + (dudy_s + dvdx_s)**2, 0.0))
#     S_mag[SOLID] = 0.0

#     # Vorticity magnitude |Ω| for SST νt limiter
#     Omega_sst = np.abs(dvdx_s - dudy_s)
#     Omega_sst[SOLID] = 0.0

#     d_wall = WALL_DIST + 1e-10   # wall distance [m]

#     # Cross-diffusion term CDkω
#     dkdx  = np.gradient(k_turb,   dx, axis=1)
#     dkdy  = np.gradient(k_turb,   dy, axis=0)
#     dowdx = np.gradient(omega_sst, dx, axis=1)
#     dowdy = np.gradient(omega_sst, dy, axis=0)
#     CDkw  = np.maximum(
#         2.0 * KW_SIGMA_W2 / (omega_sst + 1e-10) * (dkdx*dowdx + dkdy*dowdy),
#         1e-20)

#     # Blending function F1  (1 = k-ω near wall;  0 = k-ε free stream)
#     sqrt_k = np.sqrt(np.maximum(k_turb, 0.0))
#     a1_a   = sqrt_k / (KW_BETA_STAR * omega_sst * d_wall)
#     a1_b   = 500.0 * nu_mean / (omega_sst * d_wall**2 + 1e-10)
#     a1_c   = 4.0 * KW_SIGMA_W2 * k_turb / (CDkw * d_wall**2 + 1e-10)
#     arg1   = np.minimum(np.maximum(a1_a, a1_b), a1_c)
#     F1     = np.tanh(np.clip(arg1, 0.0, 10.0)**4)
#     F1[SOLID] = 0.0

#     # Blending function F2  (for νt SST limiter)
#     arg2   = np.maximum(2.0 * a1_a, a1_b)
#     F2_sst = np.tanh(np.clip(arg2, 0.0, 10.0)**2)
#     F2_sst[SOLID] = 0.0

#     # Blended SST constants (inner/outer layer weighted by F1)
#     sigma_k  = F1 * KW_SIGMA_K1 + (1.0 - F1) * KW_SIGMA_K2
#     sigma_w  = F1 * KW_SIGMA_W1 + (1.0 - F1) * KW_SIGMA_W2
#     beta_bl  = F1 * KW_BETA1    + (1.0 - F1) * KW_BETA2
#     alpha_bl = F1 * KW_ALPHA1   + (1.0 - F1) * KW_ALPHA2

#     # Turbulent kinematic viscosity νt (SST limiter prevents over-production)
#     nu_t = (KW_A1 * k_turb /
#             np.maximum(KW_A1 * omega_sst, Omega_sst * F2_sst + 1e-10))
#     nu_t = np.clip(nu_t, 0.0, 50.0 * nu_mean)
#     nu_t[SOLID] = 0.0

#     # Effective viscosity  μeff = μ(T) + ρ(T)·νt
#     nu_eff = NU + nu_t

#     # Turbulence production Pk  (Bradshaw limiter: Pk ≤ 10β*kω)
#     P_k = nu_t * S_mag**2
#     P_k = np.minimum(P_k, 10.0 * KW_BETA_STAR * k_turb * omega_sst)
#     P_k[SOLID] = 0.0

#     # Helper: upwind advection of scalar f
#     def _upwind(f, uu, vv):
#         fx = np.gradient(f, dx, axis=1)
#         fy = np.gradient(f, dy, axis=0)
#         res = (np.where(uu >= 0, uu, 0.0) * fx +
#                np.where(uu <  0, uu, 0.0) * np.roll(fx, -1, axis=1) +
#                np.where(vv >= 0, vv, 0.0) * fy +
#                np.where(vv <  0, vv, 0.0) * np.roll(fy, -1, axis=0))
#         res[SOLID] = 0.0
#         return -res

#     # Helper: variable-coefficient diffusion ∇·(D ∇f)
#     def _diff(f, D):
#         res = (np.gradient(D * np.gradient(f, dx, axis=1), dx, axis=1) +
#                np.gradient(D * np.gradient(f, dy, axis=0), dy, axis=0))
#         res[SOLID] = 0.0
#         return res

#     # k equation  — semi-implicit destruction β*kω to ensure stability
#     k_rhs   = (_upwind(k_turb, u, v) +
#                 _diff(k_turb, NU + sigma_k * nu_t) + P_k) * SIM_DT
#     k_turb  = (k_turb + k_rhs) / (1.0 + KW_BETA_STAR * omega_sst * SIM_DT)
#     k_turb  = np.clip(k_turb, 1e-10, 10.0)
#     k_turb[SOLID] = 0.0

#     # ω equation  — semi-implicit destruction βω²
#     w_prod  = alpha_bl * (omega_sst / (k_turb + 1e-10)) * P_k
#     w_cd    = (1.0 - F1) * CDkw            # cross-diffusion in outer layer
#     w_rhs   = (_upwind(omega_sst, u, v) +
#                 _diff(omega_sst, NU + sigma_w * nu_t) +
#                 w_prod + w_cd) * SIM_DT
#     omega_sst = (omega_sst + w_rhs) / (1.0 + beta_bl * omega_sst * SIM_DT)
#     # Dirichlet wall BC: ω_wall = 60ν/(β1·Δy²)
#     omg_wbc   = 60.0 * nu_mean / (KW_BETA1 * min(dx, dy)**2)
#     omega_sst = np.clip(omega_sst, omg_wbc * 0.01, 1e7)
#     omega_sst[SOLID] = omg_wbc

#     # Turbulence diagnostics
#     turb_intensity = float(
#         np.nanmean(np.sqrt(2.0/3.0 * np.maximum(k_turb[FLUID], 0.0)))
#     ) / (U_inf + 1e-10)
#     nu_t_ratio = float(np.nanmean(nu_t[FLUID])) / (nu_mean + 1e-10)

#     # ────────────────────────────────────────────────────────────────────────
#     # D3.  TURBULENT THERMAL DIFFUSION  (Eq.3 enhanced by k-ω SST)
#     #      keff = k(T) + ρ·cp·νt/Prt
#     # ────────────────────────────────────────────────────────────────────────
#     alpha_t     = nu_t / KW_PR_T   # turbulent thermal diffusivity [m²/s]
#     T_turb_diff = _diff(T_field, RHO * CP * alpha_t) / (RHO * CP + 1e-9)
#     T_field    += T_turb_diff * SIM_DT * 0.5
#     T_field[SOLID] = T_solid[SOLID]
#     T_field = np.clip(T_field, T_IN - 2.0, T_IN + 120.0)

#     # ────────────────────────────────────────────────────────────────────────
#     # E.  BOUSSINESQ BUOYANCY  ρ(T)·g·β(T)·ΔT  (Eq.2 body force)
#     # ────────────────────────────────────────────────────────────────────────
#     # dT     = T_field - T_IN
#     # v_buoy = RHO * G_ACC * BETA * dT / (rho_mean + 1e-9) * SIM_DT * 0.3
#     # v_buoy[SOLID] = 0.0
#     # v_buoy = np.clip(v_buoy, -0.05, 0.05)
#     # v += v_buoy

#     # ────────────────────────────────────────────────────────────────────────
#     # F.  VORTICITY TRANSPORT  (Eq.2 curl — now uses νeff = ν + νt)
#     #
#     #   ∂ω/∂t = −u·∇ω + ∇·(νeff ∇ω) − g·β(T)·∂T/∂x
#     # ────────────────────────────────────────────────────────────────────────
#     dTdx = np.gradient(T_field, dx, axis=1);  dTdx[SOLID] = 0.0
#     baroclinic = -G_ACC * BETA * dTdx
#     baroclinic[SOLID] = 0.0

#     domega_dx = np.gradient(omega_field, dx, axis=1)
#     domega_dy = np.gradient(omega_field, dy, axis=0)
#     adv = -(np.where(u >= 0, u, 0.0) * domega_dx +
#             np.where(u <  0, u, 0.0) * np.roll(domega_dx, -1, axis=1) +
#             np.where(v >= 0, v, 0.0) * domega_dy +
#             np.where(v <  0, v, 0.0) * np.roll(domega_dy, -1, axis=0))
#     adv[SOLID] = 0.0

#     # νeff diffusion — includes turbulent contribution from k-ω SST
#     nu_omg_dx = nu_eff * np.gradient(omega_field, dx, axis=1)
#     nu_omg_dy = nu_eff * np.gradient(omega_field, dy, axis=0)
#     diff_omg  = (np.gradient(nu_omg_dx, dx, axis=1) +
#                  np.gradient(nu_omg_dy, dy, axis=0))
#     diff_omg[SOLID] = 0.0

#     omega_field += (adv + diff_omg + baroclinic) * SIM_DT
#     omega_field  = np.clip(omega_field, -OMEGA_MAX, OMEGA_MAX)
#     apply_solid_bc(omega_field, 0.0)

#     # ────────────────────────────────────────────────────────────────────────
#     # G.  STREAM-FUNCTION → EDDY VELOCITIES
#     # ────────────────────────────────────────────────────────────────────────
#     psi        = np.cumsum(-omega_field * dy, axis=0)
#     eddy_scale = min(0.05 * U_inf / (np.max(np.abs(psi)) + 1e-9), 0.08)
#     u_eddy     =  np.gradient(psi, dy, axis=0) * eddy_scale
#     v_eddy     = -np.gradient(psi, dx, axis=1) * eddy_scale
#     apply_solid_bc(u_eddy, 0.0)
#     apply_solid_bc(v_eddy, 0.0)

#     u_total = np.clip(u + u_eddy, -2.0, 2.0);  u_total[SOLID] = 0.0
#     v_total = np.clip(v + v_eddy, -1.0, 1.0);  v_total[SOLID] = 0.0
#     vel_mag = np.sqrt(u_total**2 + v_total**2); vel_mag[SOLID] = np.nan

#     # ────────────────────────────────────────────────────────────────────────
#     # H.  DARCY-WEISBACH PRESSURE DROP  (Eq.7)
#     # ────────────────────────────────────────────────────────────────────────
#     u_mean  = max(float(np.nanmean(np.abs(u_total[FLUID]))), 1e-6)
#     Re_bulk = rho_mean * u_mean * DH / mu_mean
#     Re_bulk = np.clip(Re_bulk, 1.0, 1e6)
#     f_DW    = (64.0 / Re_bulk if Re_bulk < 2300
#                else 0.316 * Re_bulk**(-0.25))
#     delta_P = (f_DW * (L_CH / DH) + K_FIT) * 0.5 * rho_mean * u_mean**2
#     delta_P = np.clip(delta_P, 0.0, 5e4)

#     P_field = np.tile(np.linspace(delta_P, 0, NX), (NY, 1)).astype(float)
#     for (sx, sy), _ in SERVERS:
#         P_field += 200.0 * np.exp(-((X-sx+0.05)**2 + (Y-sy)**2) / 0.002)
#     P_field[SOLID] = np.nan

#     # ────────────────────────────────────────────────────────────────────────
#     # I.  DITTUS-BOELTER Nu  (Eq.5) — Pr(T) field, μb/μw correction
#     # ────────────────────────────────────────────────────────────────────────
#     Pr_bulk   = float(np.nanmean(PR[FLUID]))
#     mu_b      = float(np.nanmean(MU[FLUID]))
#     mu_w      = (float(np.nanmean(MU[SURFACE])) if SURFACE.any()
#                  else mu_f(T_IN + 15.0))
#     visc_corr = np.clip((mu_b / mu_w)**0.14, 0.5, 2.0)
#     Nu_avg    = np.clip(0.023 * Re_bulk**0.8 * Pr_bulk**0.4 * visc_corr,
#                         3.66, 1e4)
#     h_conv    = Nu_avg * float(np.nanmean(K[FLUID])) / DH   # [W/m²K]

#     # ────────────────────────────────────────────────────────────────────────
#     # J.  CONJUGATE SOLID  (Eq.4  ∇·(ks∇Ts)=0)
#     # ────────────────────────────────────────────────────────────────────────
#     for (sx, sy), Q in SERVERS:
#         area_s = S_SIZE[0] * S_SIZE[1]
#         q_flux = Q / (area_s * 2.0 * K_SOLID)
#         s_mask = SOLID.copy()
#         Ts_new = T_solid.copy()
#         for _ in range(5):
#             Ts_new = np.where(
#                 s_mask,
#                 (np.roll(Ts_new,  1, axis=0) + np.roll(Ts_new, -1, axis=0) +
#                  np.roll(Ts_new,  1, axis=1) + np.roll(Ts_new, -1, axis=1)) / 4.0
#                 + q_flux,
#                 T_solid)
#         T_solid = np.where(s_mask, Ts_new, T_solid)
#         T_solid[s_mask] = np.clip(T_solid[s_mask],
#                                   T_IN + Q/500*6, T_IN + Q/500*16)

#     # ────────────────────────────────────────────────────────────────────────
#     # K.  CDU BULK ODE  (Eq.6)
#     # ────────────────────────────────────────────────────────────────────────
#     cp_b   = float(cp_f(Tb_cdu))
#     dTb_dt = (m_dot * cp_b * (T_IN - Tb_cdu)
#               + Q_IT - UA_CDU * (Tb_cdu - T_COOL)) / (M_FLUID * cp_b + 1e-9)
#     Tb_cdu = float(np.clip(Tb_cdu + dTb_dt * SIM_DT, T_IN - 5.0, T_IN + 40.0))

#     # ────────────────────────────────────────────────────────────────────────
#     # L.  OUTLET / SERVER TEMPERATURES  +  THERMAL DIAGNOSTICS
#     # ────────────────────────────────────────────────────────────────────────
#     out_msk = FLUID[:, -3:]
#     T_out   = (float(np.nanmean(T_field[:, -3:][out_msk]))
#                if out_msk.any() else T_IN)
#     delta_T = T_out - T_IN

#     Ts_list   = []
#     stat_list = []
#     for hm in SERVER_HALO:
#         Ts = float(np.nanmean(T_field[hm])) if hm.any() else T_IN
#         Ts_list.append(Ts)
#         stat_list.append("BURN" if Ts > BURN_THRESH else "FINE")

#     # Server solid temperature extremes
#     solid_vals  = T_solid[SOLID]
#     T_max_srv   = float(np.nanmax(solid_vals))  if solid_vals.size else T_IN
#     T_min_srv   = float(np.nanmin(solid_vals))  if solid_vals.size else T_IN

#     # Fluid temperature extremes
#     fluid_vals  = T_field[FLUID]
#     T_min_fld   = float(np.nanmin(fluid_vals))  if fluid_vals.size else T_IN

#     # Thermal efficiency η = ṁ·cp·(Tout−Tin) / Q_IT × 100 %
#     #   numerator  = energy actually absorbed by the coolant
#     #   denominator= total heat injected by servers
#     thermal_eff = np.clip(
#         m_dot * float(np.nanmean(CP[FLUID])) * delta_T / (Q_IT + 1e-9) * 100.0,
#         0.0, 100.0)

#     # ────────────────────────────────────────────────────────────────────────
#     # M.  PUE  (Eq.8)   Ppump = V̇ΔP/ηp   Pchiller = (PIT+Ppump)/COP_eff
#     #     COP degrades with temperature lift (Carnot penalty):
#     #     COP_eff = COP_CHILL / (1 + 0.025·max(Tout−Tin, 0))
#     #     This couples PUE to the live thermal field — gives real variation.
#     # ────────────────────────────────────────────────────────────────────────
#     COP_eff   = max(COP_CHILL / (1.0 + 0.025 * max(T_out - T_IN, 0.0)), 1.2)
#     P_pump    = np.clip(V_DOT * delta_P / ETA_PUMP, 0.0, 5000.0)
#     P_chiller = (Q_IT + P_pump) / COP_eff
#     pue_inst  = np.clip((Q_IT + P_pump + P_chiller) / (Q_IT + 1e-9),
#                         1.0, 3.0)

#     # ────────────────────────────────────────────────────────────────────────
#     # N.  DIAGNOSTICS
#     # ────────────────────────────────────────────────────────────────────────
#     alpha_th = float(np.nanmean(K[FLUID])) / (rho_mean * cp_mean + 1e-9)
#     Ra       = np.clip(                                            # noqa: F841
#         G_ACC * float(np.nanmean(BETA[FLUID])) * float(np.nanmax(dT)) * W_CH**3
#         / (nu_mean * alpha_th + 1e-12), 0.0, 1e12)

#     rho_span  = float(np.nanmax(RHO[FLUID])  - np.nanmin(RHO[FLUID]))
#     mu_span   = float(np.nanmax(MU[FLUID])   - np.nanmin(MU[FLUID]))
#     beta_span = float(np.nanmax(BETA[FLUID]) - np.nanmin(BETA[FLUID]))

#     return (T_field, P_field, omega_field.copy(), vel_mag,
#             u_total, v_total, u_eddy, v_eddy,
#             dTdx, pue_inst, Ra, dT, T_out, delta_T,
#             Ts_list, stat_list, delta_P, Nu_avg, Re_bulk, Tb_cdu,
#             rho_span, mu_span, beta_span,
#             RHO, MU, BETA,
#             nu_t, turb_intensity, nu_t_ratio,
#             T_max_srv, T_min_srv, T_min_fld, thermal_eff, h_conv)

import numpy as np

# def compute_frame():
#     global p, u, v, T_field
#     global omega_field, T_solid, Tb_cdu, k_turb, omega_sst

#     # Scale inputs
#     Q_IT  = (Q1_W + Q2_W) * Q_SCALE
#     m_dot = rho_f(T_IN) * V_DOT
#     U_inf = V_DOT / (W_CH * CH_DEPTH)

#     # =========================================================================
#     # 0. HELPER FUNCTIONS (Stable Advection & Diffusion)
#     # =========================================================================
#     def upwind_advection(f, u_vel, v_vel):
#         """1st-order upwind scheme for stable convection."""
#         f_px = np.roll(f, -1, axis=1)
#         f_nx = np.roll(f, 1, axis=1)
#         f_py = np.roll(f, -1, axis=0)
#         f_ny = np.roll(f, 1, axis=0)

#         dfdx_fw = (f_px - f) / dx
#         dfdx_bw = (f - f_nx) / dx
#         dfdy_fw = (f_py - f) / dy
#         dfdy_bw = (f - f_ny) / dy

#         adv_x = np.where(u_vel > 0, u_vel * dfdx_bw, u_vel * dfdx_fw)
#         adv_y = np.where(v_vel > 0, v_vel * dfdy_bw, v_vel * dfdy_fw)
        
#         res = adv_x + adv_y
#         res[SOLID] = 0.0
#         return res

#     def compute_diffusion(f, diff_coeff):
#         """Variable coefficient diffusion: ∇·(Γ ∇f)"""
#         dfdx = np.gradient(f, dx, axis=1)
#         dfdy = np.gradient(f, dy, axis=0)
#         res = (np.gradient(diff_coeff * dfdx, dx, axis=1) +
#                np.gradient(diff_coeff * dfdy, dy, axis=0))
#         res[SOLID] = 0.0
#         return res

#     # =========================================================================
#     # 1. LOCAL PROPERTY FIELDS (T-dependent)
#     # =========================================================================
#     RHO  = rho_f(T_field)
#     MU   = mu_f(T_field)
#     CP   = cp_f(T_field)
#     K    = k_f(T_field)
#     BETA = beta_f(T_field)
#     NU   = MU / (RHO + 1e-12)
#     PR   = MU * CP / (K + 1e-12)

#     rho_mean = float(np.nanmean(RHO[FLUID]))
#     nu_mean  = float(np.nanmean(NU[FLUID]))
#     mu_mean  = float(np.nanmean(MU[FLUID]))
#     cp_mean  = float(np.nanmean(CP[FLUID]))

#     dT = T_field - T_IN

#     # =========================================================================
#     # 2. k-ω SST TURBULENCE MODEL
#     # =========================================================================
#     dudx = np.gradient(u, dx, axis=1)
#     dudy = np.gradient(u, dy, axis=0)
#     dvdx = np.gradient(v, dx, axis=1)
#     dvdy = np.gradient(v, dy, axis=0)
    
#     # Strain rate and Vorticity magnitude
#     S2 = 2.0*(dudx**2 + dvdy**2) + (dudy + dvdx)**2
#     S_mag = np.sqrt(S2 + 1e-12)
#     Omega_mag = np.sqrt((dvdx - dudy)**2 + 1e-12)
    
#     S_mag[SOLID] = 0.0
#     Omega_mag[SOLID] = 0.0

#     d_wall = WALL_DIST + 1e-10

#     # Blending functions (F1, F2) and Cross-diffusion
#     dkdx = np.gradient(k_turb, dx, axis=1)
#     dkdy = np.gradient(k_turb, dy, axis=0)
#     dowdx = np.gradient(omega_sst, dx, axis=1)
#     dowdy = np.gradient(omega_sst, dy, axis=0)
    
#     CDkw = np.maximum(2.0 * KW_SIGMA_W2 / (omega_sst + 1e-10) * (dkdx*dowdx + dkdy*dowdy), 1e-20)

#     sqrt_k = np.sqrt(np.maximum(k_turb, 0.0))
#     a1_a = sqrt_k / (KW_BETA_STAR * omega_sst * d_wall)
#     a1_b = 500.0 * nu_mean / (omega_sst * d_wall**2 + 1e-10)
#     a1_c = 4.0 * KW_SIGMA_W2 * k_turb / (CDkw * d_wall**2 + 1e-10)
    
#     F1 = np.tanh(np.clip(np.minimum(np.maximum(a1_a, a1_b), a1_c), 0.0, 10.0)**4)
#     F2 = np.tanh(np.clip(np.maximum(2.0 * a1_a, a1_b), 0.0, 10.0)**2)
#     F1[SOLID] = F2[SOLID] = 0.0

#     # Blended Constants
#     sigma_k = F1 * KW_SIGMA_K1 + (1.0 - F1) * KW_SIGMA_K2
#     sigma_w = F1 * KW_SIGMA_W1 + (1.0 - F1) * KW_SIGMA_W2
#     beta_bl = F1 * KW_BETA1    + (1.0 - F1) * KW_BETA2
#     alpha_bl= F1 * KW_ALPHA1   + (1.0 - F1) * KW_ALPHA2

#     # Eddy Viscosity (with limiter)
#     nu_t = KW_A1 * k_turb / np.maximum(KW_A1 * omega_sst, Omega_mag * F2 + 1e-10)
#     nu_t = np.clip(nu_t, 0.0, 50.0 * nu_mean)
#     nu_t[SOLID] = 0.0
#     nu_eff = NU + nu_t

#     # Production (with Bradshaw limiter)
#     P_k = np.minimum(nu_t * S2, 10.0 * KW_BETA_STAR * k_turb * omega_sst)
#     P_k[SOLID] = 0.0

#     # Evolve k and ω
#     k_adv = upwind_advection(k_turb, u, v)
#     k_diff = compute_diffusion(k_turb, NU + sigma_k * nu_t)
#     k_turb = (k_turb + SIM_DT * (-k_adv + k_diff + P_k)) / (1.0 + KW_BETA_STAR * omega_sst * SIM_DT)
#     k_turb = np.clip(k_turb, 1e-10, 10.0)
#     k_turb[SOLID] = 0.0

#     w_prod = alpha_bl * (omega_sst / (k_turb + 1e-10)) * P_k
#     w_adv = upwind_advection(omega_sst, u, v)
#     w_diff = compute_diffusion(omega_sst, NU + sigma_w * nu_t)
#     omega_sst = (omega_sst + SIM_DT * (-w_adv + w_diff + w_prod + (1.0 - F1)*CDkw)) / (1.0 + beta_bl * omega_sst * SIM_DT)
    
#     omg_wbc = 60.0 * nu_mean / (KW_BETA1 * min(dx, dy)**2)
#     omega_sst = np.clip(omega_sst, omg_wbc * 0.01, 1e7)
#     omega_sst[SOLID] = omg_wbc

#     # Diagnostics
#     turb_intensity = float(np.nanmean(np.sqrt(2.0/3.0 * np.maximum(k_turb[FLUID], 0.0)))) / (U_inf + 1e-10)
#     nu_t_ratio = float(np.nanmean(nu_t[FLUID])) / (nu_mean + 1e-10)

#     # =========================================================================
#     # 3. MOMENTUM PREDICTOR (Navier-Stokes)
#     # =========================================================================
#     conv_u = upwind_advection(u, u, v)
#     conv_v = upwind_advection(v, u, v)

#     diff_u = compute_diffusion(u, nu_eff)
#     diff_v = compute_diffusion(v, nu_eff)

#     F_b = G_ACC * BETA * dT # Boussinesq Buoyancy

#     u_star = u + SIM_DT * (-conv_u + diff_u)
#     v_star = v + SIM_DT * (-conv_v + diff_v + F_b)

#     u_star[SOLID] = 0.0
#     v_star[SOLID] = 0.0

#     # =========================================================================
#     # 4. PRESSURE POISSON EQUATION (PPE)
#     # =========================================================================
#     div_u_star = np.gradient(u_star, dx, axis=1) + np.gradient(v_star, dy, axis=0)
#     rhs = div_u_star / SIM_DT
#     inv_rho = 1.0 / (RHO + 1e-12)

#     # Jacobi Iteration for Variable Density Projection
#     for _ in range(50):
#         p_new = p.copy()
        
#         rho_x_fwd = inv_rho[1:-1,2:] + inv_rho[1:-1,1:-1]
#         rho_x_bwd = inv_rho[1:-1,:-2] + inv_rho[1:-1,1:-1]
#         rho_y_fwd = inv_rho[2:,1:-1] + inv_rho[1:-1,1:-1]
#         rho_y_bwd = inv_rho[:-2,1:-1] + inv_rho[1:-1,1:-1]

#         p_new[1:-1,1:-1] = (
#             (rho_x_fwd * p[1:-1,2:] + rho_x_bwd * p[1:-1,:-2]) * dy**2 +
#             (rho_y_fwd * p[2:,1:-1] + rho_y_bwd * p[:-2,1:-1]) * dx**2 -
#             2.0 * rhs[1:-1,1:-1] * dx**2 * dy**2
#         ) / ((rho_x_fwd + rho_x_bwd) * dy**2 + (rho_y_fwd + rho_y_bwd) * dx**2 + 1e-12)

#         # Neumann BCs (Zero normal gradient)
#         p_new[:,0]  = p_new[:,1]
#         p_new[:,-1] = 0.0 # Dirichlet at outlet to anchor pressure
#         p_new[0,:]  = p_new[1,:]
#         p_new[-1,:] = p_new[-2,:]
        
#         p[:] = p_new

#     # =========================================================================
#     # 5. VELOCITY CORRECTOR
#     # =========================================================================
#     dpdx = np.gradient(p, dx, axis=1)
#     dpdy = np.gradient(p, dy, axis=0)

#     u = u_star - SIM_DT * dpdx * inv_rho
#     v = v_star - SIM_DT * dpdy * inv_rho

#     # Enforce Wall/Solid BCs strictly
#     u[SOLID] = 0.0
#     v[SOLID] = 0.0
#     u[:, 0] = U_inf;  v[:, 0] = 0.0  # Inlet
#     u[:, -1] = u[:, -2]; v[:, -1] = v[:, -2]  # Outlet
#     u[0, :] = 0.0; u[-1, :] = 0.0    # Top/Bottom walls
#     v[0, :] = 0.0; v[-1, :] = 0.0

#     # =========================================================================
#     # 6. ENERGY EQUATION (Fluid Heat Transfer)
#     # =========================================================================
#     # Viscous Dissipation Φ
#     Phi = MU * S2
#     Phi[SOLID] = 0.0
    
#     # Effective Thermal Diffusivity α_eff = α + α_t
#     alpha_eff = (K / (RHO * CP + 1e-12)) + (nu_t / KW_PR_T)

#     adv_T = upwind_advection(T_field, u, v)
#     diff_T = compute_diffusion(T_field, alpha_eff)
    
#     # Server Heat Source 
#     Q_src = np.zeros_like(T_field)
#     for halo, (_, Q) in zip(SERVER_HALO, SERVERS):
#         Q_src[halo] += Q * Q_SCALE / (dx * dy * CH_DEPTH)
#     Q_src /= (RHO * CP + 1e-12)

#     T_field += SIM_DT * (-adv_T + diff_T + Q_src + Phi / (RHO * CP + 1e-12))
    
#     # Boundary Conditions for Temperature
#     T_field[:, 0] = T_IN             # Inlet
#     T_field[:, -1] = T_field[:, -2]  # Outlet
#     T_field[SOLID] = T_solid[SOLID]  # Anchor solid nodes
#     T_field = np.clip(T_field, T_IN - 5.0, T_IN + 120.0)

#     dTdx = np.gradient(T_field, dx, axis=1)

#     # =========================================================================
#     # 7. CONJUGATE HEAT TRANSFER (Solids)
#     # =========================================================================
#     for (sx, sy), Q in SERVERS:
#         q_flux = Q / (S_SIZE[0] * S_SIZE[1] * 2.0 * K_SOLID)
#         s_mask = SOLID.copy()
#         Ts_new = T_solid.copy()
#         for _ in range(5):
#             Ts_new = np.where(s_mask,
#                 (np.roll(Ts_new, 1, axis=0) + np.roll(Ts_new, -1, axis=0) +
#                  np.roll(Ts_new, 1, axis=1) + np.roll(Ts_new, -1, axis=1)) / 4.0 + q_flux,
#                 T_solid)
#         T_solid = np.where(s_mask, Ts_new, T_solid)
#         T_solid[s_mask] = np.clip(T_solid[s_mask], T_IN, T_IN + 120.0)

#     # =========================================================================
#     # 8. SYNTHETIC TURBULENCE / VORTICITY VISUALIZATION
#     # =========================================================================
#     baroclinic = -G_ACC * BETA * dTdx
#     baroclinic[SOLID] = 0.0

#     adv_omg = upwind_advection(omega_field, u, v)
#     diff_omg = compute_diffusion(omega_field, nu_eff)

#     omega_field += SIM_DT * (-adv_omg + diff_omg + baroclinic)
#     omega_field = np.clip(omega_field, -OMEGA_MAX, OMEGA_MAX)
#     omega_field[SOLID] = 0.0

#     psi = np.cumsum(-omega_field * dy, axis=0)
#     eddy_scale = min(0.05 * U_inf / (np.max(np.abs(psi)) + 1e-9), 0.08)
#     u_eddy = np.gradient(psi, dy, axis=0) * eddy_scale
#     v_eddy = -np.gradient(psi, dx, axis=1) * eddy_scale
#     u_eddy[SOLID] = 0.0; v_eddy[SOLID] = 0.0

#     u_total = np.clip(u + u_eddy, -2.0, 2.0); u_total[SOLID] = 0.0
#     v_total = np.clip(v + v_eddy, -1.0, 1.0); v_total[SOLID] = 0.0
#     vel_mag = np.sqrt(u_total**2 + v_total**2); vel_mag[SOLID] = np.nan

#     # =========================================================================
#     # 9. ENGINEERING DIAGNOSTICS & PUE
#     # =========================================================================
#     u_mean = max(float(np.nanmean(np.abs(u_total[FLUID]))), 1e-6)
#     Re_bulk = rho_mean * u_mean * DH / mu_mean
#     Re_bulk = np.clip(Re_bulk, 1.0, 1e6)
    
#     f_DW = 64.0 / Re_bulk if Re_bulk < 2300 else 0.316 * Re_bulk**(-0.25)
#     delta_P = (f_DW * (L_CH / DH) + K_FIT) * 0.5 * rho_mean * u_mean**2
#     delta_P = np.clip(delta_P, 0.0, 5e4)

#     P_field = np.tile(np.linspace(delta_P, 0, NX), (NY, 1)).astype(float)
#     P_field[SOLID] = np.nan

#     Pr_bulk = float(np.nanmean(PR[FLUID]))
#     mu_w = float(np.nanmean(MU[SURFACE])) if SURFACE.any() else mu_f(T_IN + 15.0)
#     visc_corr = np.clip((mu_mean / mu_w)**0.14, 0.5, 2.0)
#     Nu_avg = np.clip(0.023 * Re_bulk**0.8 * Pr_bulk**0.4 * visc_corr, 3.66, 1e4)
#     h_conv = Nu_avg * float(np.nanmean(K[FLUID])) / DH

#     # CDU Heat Exchanger
#     cp_b = float(cp_f(Tb_cdu))
#     dTb_dt = (m_dot * cp_b * (T_IN - Tb_cdu) + Q_IT - UA_CDU * (Tb_cdu - T_COOL)) / (M_FLUID * cp_b + 1e-9)
#     Tb_cdu = float(np.clip(Tb_cdu + dTb_dt * SIM_DT, T_IN - 5.0, T_IN + 40.0))

#     # Thermal Diagnostics
#     out_msk = FLUID[:, -3:]
#     T_out = float(np.nanmean(T_field[:, -3:][out_msk])) if out_msk.any() else T_IN
#     delta_T = T_out - T_IN

#     Ts_list, stat_list = [], []
#     for hm in SERVER_HALO:
#         Ts = float(np.nanmean(T_field[hm])) if hm.any() else T_IN
#         Ts_list.append(Ts)
#         stat_list.append("BURN" if Ts > BURN_THRESH else "FINE")

#     solid_vals = T_solid[SOLID]
#     T_max_srv = float(np.nanmax(solid_vals)) if solid_vals.size else T_IN
#     T_min_srv = float(np.nanmin(solid_vals)) if solid_vals.size else T_IN
    
#     fluid_vals = T_field[FLUID]
#     T_min_fld = float(np.nanmin(fluid_vals)) if fluid_vals.size else T_IN

#     thermal_eff = np.clip(m_dot * float(np.nanmean(CP[FLUID])) * delta_T / (Q_IT + 1e-9) * 100.0, 0.0, 100.0)

#     # Power Usage Effectiveness (PUE)
#     COP_eff = max(COP_CHILL / (1.0 + 0.025 * max(T_out - T_IN, 0.0)), 1.2)
#     P_pump = np.clip(V_DOT * delta_P / ETA_PUMP, 0.0, 5000.0)
#     P_chiller = (Q_IT + P_pump) / COP_eff
#     pue_inst = np.clip((Q_IT + P_pump + P_chiller) / (Q_IT + 1e-9), 1.0, 3.0)

#     alpha_th = float(np.nanmean(K[FLUID])) / (rho_mean * cp_mean + 1e-9)
#     Ra = np.clip(G_ACC * float(np.nanmean(BETA[FLUID])) * float(np.nanmax(dT)) * W_CH**3 / (nu_mean * alpha_th + 1e-12), 0.0, 1e12)

#     rho_span = float(np.nanmax(RHO[FLUID]) - np.nanmin(RHO[FLUID]))
#     mu_span = float(np.nanmax(MU[FLUID]) - np.nanmin(MU[FLUID]))
#     beta_span = float(np.nanmax(BETA[FLUID]) - np.nanmin(BETA[FLUID]))

#     return (T_field, P_field, omega_field.copy(), vel_mag,
#             u_total, v_total, u_eddy, v_eddy,
#             dTdx, pue_inst, Ra, dT, T_out, delta_T,
#             Ts_list, stat_list, delta_P, Nu_avg, Re_bulk, Tb_cdu,
#             rho_span, mu_span, beta_span,
#             RHO, MU, BETA,
#             nu_t, turb_intensity, nu_t_ratio,
#             T_max_srv, T_min_srv, T_min_fld, thermal_eff, h_conv) 

import numpy as np

def compute_frame():
    global p, u, v, T_field
    global omega_field, T_solid, Tb_cdu, k_turb, omega_sst

    # Scale inputs
    Q_IT  = (Q1_W + Q2_W) * Q_SCALE
    m_dot = float(np.nanmean(rho_f(T_IN))) * V_DOT
    U_inf = V_DOT / (W_CH * CH_DEPTH)

    # =========================================================================
    # 0. HELPER FUNCTIONS (Stable Upwinding)
    # =========================================================================
    def upwind_advection(f, u_vel, v_vel):
        """1st-order upwind scheme for stable convection (u·∇f)"""
        f_px = np.roll(f, -1, axis=1)
        f_nx = np.roll(f, 1, axis=1)
        f_py = np.roll(f, -1, axis=0)
        f_ny = np.roll(f, 1, axis=0)

        dfdx_fw = (f_px - f) / dx
        dfdx_bw = (f - f_nx) / dx
        dfdy_fw = (f_py - f) / dy
        dfdy_bw = (f - f_ny) / dy

        adv_x = np.where(u_vel > 0, u_vel * dfdx_bw, u_vel * dfdx_fw)
        adv_y = np.where(v_vel > 0, v_vel * dfdy_bw, v_vel * dfdy_fw)
        
        return adv_x + adv_y

    # =========================================================================
    # 1. LOCAL PROPERTY FIELDS (T-dependent)
    # =========================================================================
    RHO  = rho_f(T_field)
    MU   = mu_f(T_field)
    CP   = cp_f(T_field)
    K    = k_f(T_field)
    BETA = beta_f(T_field)
    NU   = MU / (RHO + 1e-12)
    PR   = MU * CP / (K + 1e-12)

    rho_mean = float(np.nanmean(RHO[FLUID]))
    nu_mean  = float(np.nanmean(NU[FLUID]))
    mu_mean  = float(np.nanmean(MU[FLUID]))
    cp_mean  = float(np.nanmean(CP[FLUID]))

    dT = T_field - T_IN

    # =========================================================================
    # 2. k-ω SST TURBULENCE MODEL
    # =========================================================================
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    
    # Strain rate (S2) and Vorticity magnitude
    S2 = 2.0*(dudx**2 + dvdy**2) + (dudy + dvdx)**2
    S_mag = np.sqrt(S2 + 1e-12)
    Omega_mag = np.sqrt((dvdx - dudy)**2 + 1e-12)
    
    S_mag[SOLID] = 0.0
    Omega_mag[SOLID] = 0.0
    d_wall = WALL_DIST + 1e-10

    # Cross-diffusion & Blending
    dkdx, dkdy = np.gradient(k_turb, dx, axis=1), np.gradient(k_turb, dy, axis=0)
    dowdx, dowdy = np.gradient(omega_sst, dx, axis=1), np.gradient(omega_sst, dy, axis=0)
    
    CDkw = np.maximum(2.0 * KW_SIGMA_W2 / (omega_sst + 1e-10) * (dkdx*dowdx + dkdy*dowdy), 1e-20)
    a1_a = np.sqrt(np.maximum(k_turb, 0.0)) / (KW_BETA_STAR * omega_sst * d_wall)
    a1_b = 500.0 * nu_mean / (omega_sst * d_wall**2 + 1e-10)
    a1_c = 4.0 * KW_SIGMA_W2 * k_turb / (CDkw * d_wall**2 + 1e-10)
    
    F1 = np.tanh(np.clip(np.minimum(np.maximum(a1_a, a1_b), a1_c), 0.0, 10.0)**4)
    F2 = np.tanh(np.clip(np.maximum(2.0 * a1_a, a1_b), 0.0, 10.0)**2)
    F1[SOLID] = F2[SOLID] = 0.0

    sigma_k = F1 * KW_SIGMA_K1 + (1.0 - F1) * KW_SIGMA_K2
    sigma_w = F1 * KW_SIGMA_W1 + (1.0 - F1) * KW_SIGMA_W2
    beta_bl = F1 * KW_BETA1    + (1.0 - F1) * KW_BETA2
    alpha_bl= F1 * KW_ALPHA1   + (1.0 - F1) * KW_ALPHA2

    # Eddy Viscosity (νt)
    nu_t = KW_A1 * k_turb / np.maximum(KW_A1 * omega_sst, Omega_mag * F2 + 1e-10)
    nu_t = np.clip(nu_t, 0.0, 50.0 * nu_mean)
    nu_t[SOLID] = 0.0
    nu_eff = NU + nu_t

    # k-ω Production and Solvers
    P_k = np.minimum(nu_t * S2, 10.0 * KW_BETA_STAR * k_turb * omega_sst)
    P_k[SOLID] = 0.0

    k_adv = upwind_advection(k_turb, u, v)
    k_diff = np.gradient((NU + sigma_k * nu_t) * dkdx, dx, axis=1) + \
             np.gradient((NU + sigma_k * nu_t) * dkdy, dy, axis=0)
    
    k_turb = (k_turb + SIM_DT * (-k_adv + k_diff + P_k)) / (1.0 + KW_BETA_STAR * omega_sst * SIM_DT)
    k_turb = np.clip(k_turb, 1e-10, 10.0)
    k_turb[SOLID] = 0.0

    w_prod = alpha_bl * (omega_sst / (k_turb + 1e-10)) * P_k
    w_adv = upwind_advection(omega_sst, u, v)
    w_diff = np.gradient((NU + sigma_w * nu_t) * dowdx, dx, axis=1) + \
             np.gradient((NU + sigma_w * nu_t) * dowdy, dy, axis=0)
             
    omega_sst = (omega_sst + SIM_DT * (-w_adv + w_diff + w_prod + (1.0 - F1)*CDkw)) / (1.0 + beta_bl * omega_sst * SIM_DT)
    
    omg_wbc = 60.0 * nu_mean / (KW_BETA1 * min(dx, dy)**2)
    omega_sst = np.clip(omega_sst, omg_wbc * 0.01, 1e7)
    omega_sst[SOLID] = omg_wbc

    turb_intensity = float(np.nanmean(np.sqrt(2.0/3.0 * np.maximum(k_turb[FLUID], 0.0)))) / (U_inf + 1e-10)
    nu_t_ratio = float(np.nanmean(nu_t[FLUID])) / (nu_mean + 1e-10)

    # =========================================================================
    # 3. MOMENTUM PREDICTOR (Eq. 2 - Navier-Stokes)
    # =========================================================================
    conv_u = upwind_advection(u, u, v)
    conv_v = upwind_advection(v, u, v)

    diff_u = np.gradient(nu_eff * dudx, dx, axis=1) + np.gradient(nu_eff * dudy, dy, axis=0)
    diff_v = np.gradient(nu_eff * dvdx, dx, axis=1) + np.gradient(nu_eff * dvdy, dy, axis=0)

    F_b = G_ACC * BETA * dT # Boussinesq Buoyancy

    u_star = u + SIM_DT * (-conv_u + diff_u)
    v_star = v + SIM_DT * (-conv_v + diff_v + F_b)

    u_star[SOLID] = 0.0
    v_star[SOLID] = 0.0

    # =========================================================================
    # 4. PRESSURE POISSON EQUATION (Eq. 1 - Continuity)
    #    ∇²p = (ρ/Δt) ∇·u* (Incompressible formulation as per image)
    # =========================================================================
    div_u_star = np.gradient(u_star, dx, axis=1) + np.gradient(v_star, dy, axis=0)
    rhs = (RHO / SIM_DT) * div_u_star

    for _ in range(50):
        p_new = p.copy()
        p_new[1:-1,1:-1] = (
            (p[1:-1,2:] + p[1:-1,:-2]) * dy**2 +
            (p[2:,1:-1] + p[:-2,1:-1]) * dx**2 -
            rhs[1:-1,1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))

        # Boundaries
        p_new[:,0]  = p_new[:,1]   # Inlet (Neumann)
        p_new[:,-1] = 0.0          # Outlet (Dirichlet anchor)
        p_new[0,:]  = p_new[1,:]   # Top Wall
        p_new[-1,:] = p_new[-2,:]  # Bottom Wall
        p[:] = p_new

    # =========================================================================
    # 5. VELOCITY CORRECTOR
    # =========================================================================
    dpdx = np.gradient(p, dx, axis=1)
    dpdy = np.gradient(p, dy, axis=0)

    u = u_star - (SIM_DT / RHO) * dpdx
    v = v_star - (SIM_DT / RHO) * dpdy

    # Enforce Wall/Solid Boundaries strictly
    u[SOLID] = 0.0; v[SOLID] = 0.0
    u[:, 0] = U_inf; v[:, 0] = 0.0            # Inlet
    u[:, -1] = u[:, -2]; v[:, -1] = v[:, -2]  # Outlet
    u[0, :] = 0.0; u[-1, :] = 0.0             # Top/Bottom walls
    v[0, :] = 0.0; v[-1, :] = 0.0

    # =========================================================================
    # 6. MONOLITHIC ENERGY EQUATION (Eq. 3 & Eq. 4 Combined)
    #    Solves CHT implicitly to ensure exact heat flux at the boundary.
    # =========================================================================
    # Unified Properties
    RhoCp = RHO * CP
    
    K_eff = K.copy()
    K_eff[FLUID] += (nu_t[FLUID] * RHO[FLUID] * CP[FLUID] / KW_PR_T)
    K_eff[SOLID] = K_SOLID

    # Conduction (∇·(k ∇T))
    dTdx = np.gradient(T_field, dx, axis=1)
    dTdy = np.gradient(T_field, dy, axis=0)
    diff_T = np.gradient(K_eff * dTdx, dx, axis=1) + np.gradient(K_eff * dTdy, dy, axis=0)

    # Convection (u·∇T)
    adv_T = upwind_advection(T_field, u, v)
    adv_T[SOLID] = 0.0

    # Volumetric Heat Source (Servers)
    Q_src = np.zeros_like(T_field)
    for halo, (_, Q) in zip(SERVER_HALO, SERVERS):
        Q_src[halo] += Q * Q_SCALE / (dx * dy * CH_DEPTH)

    # Viscous Dissipation (Φ)
    Phi = MU * S2
    Phi[SOLID] = 0.0

    # Explicit Time Update
    T_field += SIM_DT * (-adv_T + (diff_T + Q_src + Phi) / RhoCp)
    
    # Boundary Conditions
    T_field[:, 0] = T_IN             
    T_field[:, -1] = T_field[:, -2]  
    T_field = np.clip(T_field, T_IN - 5.0, T_IN + 150.0)

    # Sync T_solid purely for external diagnostics
    T_solid[SOLID] = T_field[SOLID]

    # =========================================================================
    # 7. VISUALIZATION VORTICITY
    # =========================================================================
    baroclinic = -G_ACC * BETA * dTdx
    baroclinic[SOLID] = 0.0

    adv_omg = upwind_advection(omega_field, u, v)
    diff_omg = np.gradient(nu_eff * np.gradient(omega_field, dx, axis=1), dx, axis=1) + \
               np.gradient(nu_eff * np.gradient(omega_field, dy, axis=0), dy, axis=0)

    omega_field += SIM_DT * (-adv_omg + diff_omg + baroclinic)
    omega_field = np.clip(omega_field, -OMEGA_MAX, OMEGA_MAX)
    omega_field[SOLID] = 0.0

    psi = np.cumsum(-omega_field * dy, axis=0)
    eddy_scale = min(0.05 * U_inf / (np.max(np.abs(psi)) + 1e-9), 0.08)
    u_eddy = np.gradient(psi, dy, axis=0) * eddy_scale
    v_eddy = -np.gradient(psi, dx, axis=1) * eddy_scale
    u_eddy[SOLID] = 0.0; v_eddy[SOLID] = 0.0

    u_total = np.clip(u + u_eddy, -2.0, 2.0); u_total[SOLID] = 0.0
    v_total = np.clip(v + v_eddy, -1.0, 1.0); v_total[SOLID] = 0.0
    vel_mag = np.sqrt(u_total**2 + v_total**2); vel_mag[SOLID] = np.nan

    # =========================================================================
    # 8. ENGINEERING DIAGNOSTICS (Eq. 5, 6, 7, 8)
    # =========================================================================
    u_mean = max(float(np.nanmean(np.abs(u_total[FLUID]))), 1e-6)
    Re_bulk = rho_mean * u_mean * DH / mu_mean
    Re_bulk = np.clip(Re_bulk, 1.0, 1e6)
    
    # Darcy-Weisbach (Eq. 7)
    f_DW = 64.0 / Re_bulk if Re_bulk < 2300 else 0.316 * Re_bulk**(-0.25)
    delta_P = (f_DW * (L_CH / DH) + K_FIT) * 0.5 * rho_mean * u_mean**2
    delta_P = np.clip(delta_P, 0.0, 5e4)

    P_field = np.tile(np.linspace(delta_P, 0, NX), (NY, 1)).astype(float)
    P_field[SOLID] = np.nan

    # Heat Transfer Correlation (Eq. 5)
    Pr_bulk = float(np.nanmean(PR[FLUID]))
    mu_w = float(np.nanmean(MU[SURFACE])) if SURFACE.any() else mu_f(T_IN + 15.0)
    visc_corr = np.clip((mu_mean / mu_w)**0.14, 0.5, 2.0)
    Nu_avg = np.clip(0.023 * Re_bulk**0.8 * Pr_bulk**0.4 * visc_corr, 3.66, 1e4)
    h_conv = Nu_avg * float(np.nanmean(K[FLUID])) / DH

    # CDU Loop (Eq. 6)
    cp_b = float(cp_f(Tb_cdu))
    dTb_dt = (m_dot * cp_b * (T_IN - Tb_cdu) + Q_IT - UA_CDU * (Tb_cdu - T_COOL)) / (M_FLUID * cp_b + 1e-9)
    Tb_cdu = float(np.clip(Tb_cdu + dTb_dt * SIM_DT, T_IN - 5.0, T_IN + 40.0))

    # Thermal Diagnostics
    out_msk = FLUID[:, -3:]
    T_out = float(np.nanmean(T_field[:, -3:][out_msk])) if out_msk.any() else T_IN
    delta_T = T_out - T_IN

    Ts_list, stat_list = [], []
    for hm in SERVER_HALO:
        Ts = float(np.nanmax(T_field[hm])) if hm.any() else T_IN
        Ts_list.append(Ts)
        stat_list.append("BURN" if Ts > BURN_THRESH else "FINE")

    solid_vals = T_field[SOLID]
    T_max_srv = float(np.nanmax(solid_vals)) if solid_vals.size else T_IN
    T_min_srv = float(np.nanmin(solid_vals)) if solid_vals.size else T_IN
    
    fluid_vals = T_field[FLUID]
    T_min_fld = float(np.nanmin(fluid_vals)) if fluid_vals.size else T_IN

    # PUE Optimization (Eq. 8, 9, 10)
    COP_eff = max(COP_CHILL / (1.0 + 0.025 * max(T_out - T_IN, 0.0)), 1.2)
    P_pump = np.clip(V_DOT * delta_P / ETA_PUMP, 0.0, 5000.0)
    P_chiller = (Q_IT + P_pump) / COP_eff
    pue_inst = np.clip((Q_IT + P_pump + P_chiller) / (Q_IT + 1e-9), 1.0, 3.0)

    # Miscellaneous physics
    thermal_eff = np.clip(m_dot * float(np.nanmean(CP[FLUID])) * delta_T / (Q_IT + 1e-9) * 100.0, 0.0, 100.0)
    alpha_th = float(np.nanmean(K[FLUID])) / (rho_mean * cp_mean + 1e-9)
    Ra = np.clip(G_ACC * float(np.nanmean(BETA[FLUID])) * float(np.nanmax(dT)) * W_CH**3 / (nu_mean * alpha_th + 1e-12), 0.0, 1e12)
    rho_span = float(np.nanmax(RHO[FLUID]) - np.nanmin(RHO[FLUID]))
    mu_span = float(np.nanmax(MU[FLUID]) - np.nanmin(MU[FLUID]))
    beta_span = float(np.nanmax(BETA[FLUID]) - np.nanmin(BETA[FLUID]))

    return (T_field, P_field, omega_field.copy(), vel_mag,
            u_total, v_total, u_eddy, v_eddy,
            dTdx, pue_inst, Ra, dT, T_out, delta_T,
            Ts_list, stat_list, delta_P, Nu_avg, Re_bulk, Tb_cdu,
            rho_span, mu_span, beta_span,
            RHO, MU, BETA,
            nu_t, turb_intensity, nu_t_ratio,
            T_max_srv, T_min_srv, T_min_fld, thermal_eff, h_conv)

# ═════════════════════════════════════════════════════════════════════════════
# 7.  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
SKIP_Q = 8

def _decorate(axes_list):
    for ax in axes_list:
        for (sx, sy), _ in SERVERS:
            ax.add_patch(Rectangle(
                (sx - S_SIZE[0]/2, sy - S_SIZE[1]/2),
                S_SIZE[0], S_SIZE[1],
                fill=True, facecolor='#1a1a2e',
                edgecolor='white', lw=1.8, alpha=0.95, zorder=5))
        ax.set_xlim(0, L_CH);  ax.set_ylim(0, W_CH)
        ax.tick_params(labelsize=6)


# ═════════════════════════════════════════════════════════════════════════════
# 8.  ANIMATION UPDATE LOOP
# ═════════════════════════════════════════════════════════════════════════════
def update(frame):
    global _ros_last, Q_SCALE
    s = state
    s['t']    += SIM_DT
    s['step'] += 1

    # ── Time-varying server workload  (±20%, 60-second duty cycle) ────────
    Q_SCALE = 1.0 + 0.20 * np.sin(2.0 * np.pi * s['t'] / 60.0)

    (T_field, P_field, omega, vel_mag,
     u_tot, v_tot, u_eddy, v_eddy,
     dTdx, pue_inst, Ra, dT,
     T_out, delta_T, Ts_list, stat_list,
     delta_P, Nu_avg, Re_bulk, Tb_now,
     rho_span, mu_span, beta_span,
     RHO_f, MU_f, BETA_f,
     nu_t_field, turb_intensity, nu_t_ratio,
     T_max_srv, T_min_srv, T_min_fld, thermal_eff, h_conv) = compute_frame()

    pue_history.append(pue_inst)
    pue_avg = float(np.mean(pue_history))
    T_max   = float(np.nanmax(T_field[FLUID]))

    buf['t'].append(s['t']);         buf['pue'].append(pue_inst)
    buf['pue_avg'].append(pue_avg);  buf['t_in'].append(T_IN)
    buf['t_out'].append(T_out);      buf['delta_t'].append(delta_T)
    buf['ts1'].append(Ts_list[0]);   buf['ts2'].append(Ts_list[1])
    t_arr = list(buf['t'])

    for ax in [ax_temp, ax_turb, ax_eddy, ax_vort, ax_vel,
               ax_ros, ax_temps, ax_pue]:
        ax.cla()

    omega_lim  = max(float(np.nanmax(np.abs(omega[FLUID]))), 1e-6)
    T_plot     = np.where(FLUID, T_field, np.nan)
    omega_plot = np.where(FLUID, omega,   np.nan)
    dTdx_plot  = np.where(FLUID, dTdx,   np.nan)

    # ── 1. Thermal field ──────────────────────────────────────────────────
    ax_temp.set_title(
        f"Thermal Field  T∈[{T_IN:.0f},{np.nanmax(T_plot):.1f}]°C"
        f"   ρ-span={rho_span:.1f} kg/m³",
        color='#ff8c00', fontsize=7.5)
    ax_temp.contourf(X, Y, T_plot, levels=50, cmap='magma')
    ax_temp.quiver(X[::SKIP_Q,::SKIP_Q], Y[::SKIP_Q,::SKIP_Q],
                   u_tot[::SKIP_Q,::SKIP_Q], v_tot[::SKIP_Q,::SKIP_Q],
                   color='white', alpha=0.28, scale=3.5, width=0.003, zorder=6)
    ax_temp.contour(X, Y, T_plot,
                    levels=[T_IN+10, T_IN+20, T_IN+30],
                    colors=['yellow'], linewidths=0.6, alpha=0.5, zorder=6)

    # ── 2. k-ω SST: Turbulent Viscosity Ratio νt/ν ───────────────────────
    nu_mol_mean  = float(np.nanmean(MU_f[FLUID] / RHO_f[FLUID]))
    nut_plot     = np.where(FLUID, nu_t_field, np.nan)
    nut_ratio_2d = nut_plot / (nu_mol_mean + 1e-10)
    ax_turb.set_title(
        f"k-ω SST  νt/ν(avg)={nu_t_ratio:.1f}  I={turb_intensity*100:.2f}%"
        f"   Re={Re_bulk:.0f}  ΔP={delta_P:.1f}Pa",
        color='#00e5ff', fontsize=7.0)
    ax_turb.contourf(X, Y, nut_ratio_2d, levels=30, cmap='plasma')
    try:
        ax_turb.contour(X, Y, nut_ratio_2d,
                        levels=[1.0, 5.0, 20.0],
                        colors=['cyan'], linewidths=0.6, alpha=0.55, zorder=6)
    except Exception:
        pass

    # ── 3. Buoyancy eddy (β(T) baroclinic) ───────────────────────────────
    ax_eddy.set_title(
        f"Eddy ω  β(T)-baroclinic  μ-span={mu_span*1e3:.3f}e-3 Pa·s",
        color='#ff55ff', fontsize=7.0)
    ax_eddy.contourf(X, Y, omega_plot, levels=40, cmap='coolwarm',
                     vmin=-omega_lim, vmax=omega_lim)
    try:
        u_e = np.where(FLUID, u_eddy, 0.0)
        v_e = np.where(FLUID, v_eddy, 0.0)
        ax_eddy.streamplot(x, y, u_e, v_e,
                           color=np.sqrt(u_e**2+v_e**2)+1e-12,
                           cmap='plasma', linewidth=0.9, density=1.3,
                           arrowsize=0.7, arrowstyle='->', zorder=6)
    except Exception:
        pass

    # ── 4. Vorticity + baroclinic source ─────────────────────────────────
    ax_vort.set_title("Vorticity ω  +  dT/dx baroclinic source (lime)",
                      color='#ff4444', fontsize=7.5)
    ax_vort.imshow(omega_plot, extent=[0, L_CH, 0, W_CH],
                   cmap='RdBu_r', origin='lower', aspect='auto',
                   vmin=-omega_lim, vmax=omega_lim)
    ax_vort.contour(X, Y, dTdx_plot, levels=6,
                    colors=['lime'], linewidths=0.5, alpha=0.45, zorder=6)

    # ── 5. Velocity + Nu ─────────────────────────────────────────────────
    ax_vel.set_title(
        f"|u| Velocity  Nu={Nu_avg:.1f}  h={Nu_avg*K0/DH:.1f} W/m²K"
        f"   Pr(T̄)={float(np.nanmean(MU_f[FLUID]*cp_f(T_field[FLUID])/k_f(T_field[FLUID]))):.2f}",
        color='#ffff00', fontsize=6.8)
    ax_vel.contourf(X, Y, vel_mag, levels=30, cmap='inferno')
    ax_vel.quiver(X[::SKIP_Q,::SKIP_Q], Y[::SKIP_Q,::SKIP_Q],
                  u_tot[::SKIP_Q,::SKIP_Q], v_tot[::SKIP_Q,::SKIP_Q],
                  color='white', alpha=0.4, scale=4, width=0.004, zorder=6)

    _decorate([ax_temp, ax_turb, ax_eddy, ax_vort, ax_vel])

    # ── 6. ROS Status Panel ───────────────────────────────────────────────
    ax_ros.axis('off')
    ax_ros.set_facecolor('#0d0d0d')
    for sp in ax_ros.spines.values():
        sp.set_edgecolor('#333333')
    ax_ros.set_title("ROS  /immersion_cooling", color='#aaaaaa', fontsize=7.5)

    pue_avg_val = pue_avg if len(pue_history) > 0 else pue_inst
    s1c = '#ff4444' if stat_list[0] == 'BURN' else '#00ff88'
    s2c = '#ff4444' if stat_list[1] == 'BURN' else '#00ff88'

    rows = [
        ("sim_time",   f"{state['t']:.2f} s",                  '#ffffff'),
        ("pump_lpm",   f"{V_LPM:.3f}",                         '#00e5ff'),
        ("Tin   (C)",  f"{T_IN:.2f}",                          '#aaddff'),
        ("Tout  (C)",  f"{T_out:.2f}",                         '#ffaa44'),
        ("dT    (K)",  f"{delta_T:.2f}",                       '#ffff44'),
        ("Tmax fluid", f"{T_max:.2f}",                         '#ff8c00'),
        ("Tmin fluid", f"{T_min_fld:.2f}",                     '#88ddff'),
        ("Tmax srv",   f"{T_max_srv:.2f}",                     '#ff6644'),
        ("Tmin srv",   f"{T_min_srv:.2f}",                     '#aaffaa'),
        ("η therm %",  f"{thermal_eff:.2f}",                   '#ffffaa'),
        ("Tb_cdu(C)",  f"{Tb_now:.2f}",                        '#88ccff'),
        ("dP   (Pa)",  f"{delta_P:.2f}",                       '#ccccff'),
        ("Re   bulk",  f"{Re_bulk:.1f}",                       '#dddddd'),
        ("Nu   avg",   f"{Nu_avg:.2f}",                        '#dddddd'),
        ("PUE inst",   f"{pue_inst:.5f}",                      '#00ff88'),
        ("PUE run.avg",f"{pue_avg_val:.5f}",                   '#88ff88'),
        ("Turb I%",    f"{turb_intensity*100:.3f}",            '#ff88ff'),
        ("vt/v",       f"{nu_t_ratio:.2f}",                    '#cc88ff'),
        ("Ts1   (C)",  f"{Ts_list[0]:.2f} [{stat_list[0]}]",   s1c),
        ("Ts2   (C)",  f"{Ts_list[1]:.2f} [{stat_list[1]}]",   s2c),
    ]

    y0 = 0.97
    for key, val, col in rows:
        ax_ros.text(0.03, y0, f"{key:<13}", transform=ax_ros.transAxes,
                    fontsize=5.9, color='#666666', family='monospace')
        ax_ros.text(0.54, y0, val, transform=ax_ros.transAxes,
                    fontsize=5.9, color=col, family='monospace', fontweight='bold')
        y0 -= 0.049

    # ── 7. Thermal time-series ────────────────────────────────────────────
    ax_temps.set_facecolor('#0d0d0d')
    ax_temps.set_title(
        "Thermal Monitor  —  Tin / Tout / dT / Ts1 / Ts2   (°C & K)",
        color='#ff8c00', fontsize=8.5, fontweight='bold')

    if len(t_arr) > 1:
        ax_temps.plot(t_arr, list(buf['t_in']),
                      color='#aaddff', lw=1.6, label=f"Tin  {T_IN:.1f}C")
        ax_temps.plot(t_arr, list(buf['t_out']),
                      color='#ffaa44', lw=1.6, label=f"Tout {T_out:.1f}C")
        ax_temps.plot(t_arr, list(buf['delta_t']),
                      color='#ffff44', lw=1.4, ls='--', label=f"dT   {delta_T:.1f}K")
        ax_temps.plot(t_arr, list(buf['ts1']),
                      color='#ff6688', lw=2.0, label=f"Ts1  {Ts_list[0]:.1f}C")
        ax_temps.plot(t_arr, list(buf['ts2']),
                      color='#ff44aa', lw=2.0, label=f"Ts2  {Ts_list[1]:.1f}C")

    xlim = [t_arr[0], t_arr[-1]] if len(t_arr) > 1 else [0, 1]
    ax_temps.axhline(BURN_THRESH, color='#ff0000', lw=2.0, ls='-.',
                     label=f"BURN {BURN_THRESH:.0f}C", zorder=6)
    ax_temps.fill_betweenx([BURN_THRESH, BURN_THRESH+35],
                           xlim[0], xlim[1],
                           color='#ff0000', alpha=0.07, zorder=3)
    ax_temps.text(0.01, BURN_THRESH + 1.5, f"  !! BURN > {BURN_THRESH:.0f}C",
                  color='#ff4444', fontsize=7.5, fontweight='bold',
                  family='monospace',
                  transform=ax_temps.get_yaxis_transform(), zorder=7)
    ax_temps.set_xlabel("Sim Time (s)", color='#888888', fontsize=7)
    ax_temps.set_ylabel("Temperature (C) / dT (K)", color='#888888', fontsize=7)
    ax_temps.tick_params(labelsize=6.5, colors='#888888')
    ax_temps.legend(loc='upper left', fontsize=6.5, ncol=3,
                    facecolor='#111111', edgecolor='#444444')
    for sp in ax_temps.spines.values():
        sp.set_edgecolor('#333333')

    # ── 8. PUE time-series ────────────────────────────────────────────────
    ax_pue.set_facecolor('#0d0d0d')
    ax_pue.set_title(
        f"PUE Monitor  —  COP_eff=COP/(1+0.025·ΔT)  workload={Q_SCALE:.2f}×",
        color='#00ff88', fontsize=7.0, fontweight='bold')
    
    if len(t_arr) > 1:
        ax_pue.plot(t_arr, list(buf['pue']),
                    color='#00ff88', lw=1.5, alpha=0.7,
                    label=f"PUE inst     {pue_inst:.5f}")
        ax_pue.plot(t_arr, list(buf['pue_avg']),
                    color='#ffffff', lw=2.2, ls='--',
                    label=f"PUE run.avg  {pue_avg:.5f}")

    # --- FIX APPLIED HERE ---
    from matplotlib.ticker import ScalarFormatter
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False) 
    ax_pue.yaxis.set_major_formatter(y_formatter)
    # ------------------------

    ax_pue.set_xlabel("Sim Time (s)", color='#888888', fontsize=7)
    ax_pue.set_ylabel("PUE", color='#888888', fontsize=7)
    ax_pue.tick_params(labelsize=6.5, colors='#888888')
    ax_pue.legend(loc='upper right', fontsize=7,
                  facecolor='#111111', edgecolor='#444444')
    for sp in ax_pue.spines.values():
        sp.set_edgecolor('#333333')

    # ── 9. ROS console @ 10 Hz ───────────────────────────────────────────
    now_w = _time.monotonic()
    if (now_w - _ros_last) >= 1.0 / ROS_HZ:
        _ros_last = now_w
        G, R, YL, C, M, W, B, DM, RS = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[95m', '\033[97m', '\033[94m', '\033[90m', '\033[0m'
        s1_burn, s2_burn = stat_list[0] == "BURN", stat_list[1] == "BURN"
        ts1_col, ts2_col = R if s1_burn else G, R if s2_burn else G
        sep = f"{DM}{'─'*62}{RS}"

        print(sep)
        print(f"{DM} t={s['t']:>8.2f}s   /immersion_cooling   k-ω SST active{RS}")
        print(sep)
        print(f"  {C}PUE inst        {RS}: {W}{pue_inst:.5f}{RS}")
        print(f"  {C}PUE run.avg     {RS}: {W}{pue_avg:.5f}{RS}")
        print(f"  {YL}Tin             {RS}: {W}{T_IN:.2f} °C{RS}")
        print(f"  {YL}Tout            {RS}: {W}{T_out:.2f} °C{RS}")
        print(f"  {YL}ΔT              {RS}: {W}{delta_T:.2f} K{RS}")
        print(f"  {YL}Tmax fluid      {RS}: {W}{T_max:.2f} °C{RS}")
        print(f"  {B}Tmin fluid      {RS}: {W}{T_min_fld:.2f} °C{RS}")
        print(f"  {R}Tmax server     {RS}: {W}{T_max_srv:.2f} °C{RS}")
        print(f"  {G}Tmin server     {RS}: {W}{T_min_srv:.2f} °C{RS}")
        print(f"  {ts1_col}Ts1             {RS}: {W}{Ts_list[0]:.2f} °C  [{ts1_col}{stat_list[0]}{RS}]")
        print(f"  {ts2_col}Ts2             {RS}: {W}{Ts_list[1]:.2f} °C  [{ts2_col}{stat_list[1]}{RS}]")
        print(f"  {YL}Thermal eff η   {RS}: {W}{thermal_eff:.2f} %{RS}  {DM}[ṁcp·ΔT / Q_IT]{RS}")
        print(f"  {M}Turb intensity  {RS}: {W}{turb_intensity*100:.3f} %{RS}")
        print(f"  {M}νt / ν          {RS}: {W}{nu_t_ratio:.2f}{RS}")
        print(sep)

# ═════════════════════════════════════════════════════════════════════════════
# 9.  RUN
# ═════════════════════════════════════════════════════════════════════════════
ani = FuncAnimation(fig, update, interval=30, cache_frame_data=False)
plt.show()
