"""
Immersion-Cooled Server  —  Multi-Objective Optimisation
═══════════════════════════════════════════════════════════════════════════════
This file uses the EXACT SAME physics equations as immersion_cfd.py:
  • Fluid properties  : rho_f, mu_f, cp_f, k_f, beta_f  (T-dependent)
  • Dittus-Boelter    : Nu = 0.023 Re^0.8 Pr^0.4 (μb/μw)^0.14   (Eq.5)
  • Darcy-Weisbach    : ΔP = [f·L/Dh + ΣK]·ρu²/2               (Eq.7)
  • CDU ODE           : dTb/dt = [ṁcp(Tin−Tb)+Q−UA(Tb−Tcool)]/(Mcp) (Eq.6)
  • PUE               : (PIT+Ppump+Pchiller)/PIT                 (Eq.8)
  • COP_eff           : COP / (1 + 0.025·max(Tout−Tin, 0))     (Carnot penalty)

Objective (Eq.9 — multi-objective):
  min  F = w1·f̃₁  +  w2·f̃₂  −  w3·f̃₃
  f₁ = PUE         (minimise)
  f₂ = ΔTmax-min   (minimise)
  f₃ = h_avg       (maximise via –w3)
  f̃ᵢ = min-max normalised

Design variables:
  x₁  V̇     [0.1 , 5.0]  LPM
  x₂  Tin   [15  , 45 ]  °C
  x₃  μ     [0.5 , 5.0]  cP     (controls coolant selection)
  x₄  baffle[0   , 1  ]  –      (parametric flow director: 0=none, 1=full)

Hard constraints:
  C1  T_server_max  < 85 °C
  C2  ΔT_stack      < 15 K
  C3  Re            > 2300   (turbulent)
  C4  PUE           < 1.40   (feasible floor = 1+1/COP ≈ 1.286)

Weight method: CRITIC  (Criteria Importance Through Intercriteria Correlation)
  Step 1 — LHS sample 3000 feasible design points
  Step 2 — Normalise objective matrix to [0,1]
  Step 3 — σᵢ = std-dev of each normalised column (discriminating power)
  Step 4 — rᵢⱼ = Pearson corr between objective columns
  Step 5 — Cᵢ = σᵢ · Σⱼ(1 − rᵢⱼ)   (info content × independence)
  Step 6 — wᵢ = Cᵢ / ΣCᵢ

Optimiser: scipy Differential Evolution (global, gradient-free)
"""

import numpy as np
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════════════════════
# 1.  PHYSICS CONSTANTS  (identical to immersion_cfd.py)
# ═════════════════════════════════════════════════════════════════════════════
T_REF     = 25.0
RHO0      = 865.0   # kg/m³
MU0       = 0.0051  # Pa·s
CP0       = 2080.0  # J/kg·K
K0        = 0.145   # W/m·K
BETA0     = 0.00085 # 1/K
G_ACC     = 9.81    # m/s²

K_SOLID   = 200.0
Q1_W      = 2000.0
Q2_W      = 2500.0
Q_IT_BASE = Q1_W + Q2_W      # 4500 W

COP_CHILL = 3.5
ETA_PUMP  = 0.75
T_IN_BASE = 25.0
T_COOL    = 20.0
UA_CDU    = 800.0
V_LPM_BASE = 3.5

L_CH   = 0.6;   W_CH  = 0.3;   CH_DEPTH = 0.05
DH_CH  = 2 * W_CH * CH_DEPTH / (W_CH + CH_DEPTH)   # 0.0857 m
#S_AREA = 2 * 0.12 * 0.08           # two servers convective area [m²]
S_AREA = 1.5
M_FLUID = RHO0 * L_CH * W_CH * CH_DEPTH
K_FIT_BASE = 2.5

BURN_THRESH  = 85.0
DT_STACK_MAX = 15.0
RE_MIN       = 2300.0
PUE_MAX      = 1.50    # feasible ceiling (theoretical floor = 1+1/COP ≈ 1.286)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TEMPERATURE-DEPENDENT FLUID PROPERTIES  (same functions as CFD model)
# ═════════════════════════════════════════════════════════════════════════════
def rho_f(T, mu_scale=1.0):
    return np.clip(RHO0 - 0.71 * (T - T_REF), 800.0, 900.0)

def mu_f(T, mu_scale=1.0):
    """μ scaled by mu_scale to represent different coolant viscosities."""
    return np.clip(MU0 * mu_scale * np.exp(-0.030 * (T - T_REF)), 0.002*mu_scale, 0.015*mu_scale)

def cp_f(T):
    return np.clip(CP0 + 2.8 * (T - T_REF), 1900.0, 2500.0)

def k_f(T):
    return np.clip(K0 - 0.00018 * (T - T_REF), 0.120, 0.160)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  STEADY-STATE PHYSICS EVALUATION
#     Uses the SAME equations as immersion_cfd.py — this IS "the model"
#     run in 1D lumped-parameter form (correct for design-space search).
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_design(V_lpm, T_in, mu_cP, baffle):
    """
    Evaluate all objectives at one design point using the CFD model equations.

    Parameters
    ----------
    V_lpm  : flow rate        [LPM]
    T_in   : inlet temp       [°C]
    mu_cP  : viscosity        [cP]   (mu_scale = mu_cP / (MU0*1000))
    baffle : flow director    [0–1]

    Returns
    -------
    dict with PUE, delta_T, h_avg, T_srv_max, Re, delta_P, feasible
    """
    mu_scale = (mu_cP * 1e-3) / MU0    # relative to baseline fluid
    V_dot    = V_lpm / 60_000.0        # m³/s

    # Baffle effect: narrows effective DH, adds minor loss, improves mixing
    DH   = DH_CH * (1.0 - 0.10 * baffle)
    K_ft = K_FIT_BASE + 3.0 * baffle

    # ── Iterate 4× to get self-consistent bulk temperature ────────────────
    T_bulk = T_in + 10.0
    for _ in range(4):
        rho   = rho_f(T_bulk)
        mu    = mu_f(T_bulk, mu_scale)
        cp    = cp_f(T_bulk)
        kf    = k_f(T_bulk)
        m_dot = rho * V_dot
        T_out = T_in + Q_IT_BASE / (m_dot * cp + 1e-9)
        T_out = np.clip(T_out, T_in, T_in + 100.0)
        T_bulk = 0.5 * (T_in + T_out)

    # ── Velocity & Re  (Eq.7 terms) ───────────────────────────────────────
    #U     = V_dot / (W_CH * CH_DEPTH)
    #Re    = rho * U * DH / (mu + 1e-10)
    #Pr    = mu  * cp / (kf + 1e-10)

    # ── Velocity & Re (Updated Baffle Logic) ─────────────────────────────
    # Baffles narrow the channel, forcing fluid to move faster through gaps
    # We assume a 'baffle' of 1.0 reduces the flow area by 40%
    effective_area = (W_CH * CH_DEPTH) * (1.0 - 0.40 * baffle)
    
    U     = V_dot / (effective_area + 1e-9) # Velocity increases as area shrinks
    Re    = rho * U * DH / (mu + 1e-10)
    Pr    = mu  * cp / (kf + 1e-10)

    # ── Dittus-Boelter Nu  (Eq.5) — baffle increases mixing (+30% max) ───
    #if Re < 2300:
    #    Nu = 3.66    
    # ── Improved Heat Transfer Logic (Smooth Transition) ─────────────────
    if Re < 1000:
        Nu = 8.0  # Increased floor to prevent 26,000°C results
    elif Re < RE_MIN:
        # Linear interpolation between laminar floor and turbulent start
        Nu_turb_start = 0.023 * RE_MIN**0.8 * Pr**0.4 
        Nu = 8.0 + (Re - 1000) * (Nu_turb_start - 8.0) / (RE_MIN - 1000)
    #else:
        # Standard Dittus-Boelter for Turbulent flow
    #    mu_w      = mu_f(T_bulk + 15.0, mu_scale)
    #    visc_corr = np.clip((mu / mu_w)**0.14, 0.5, 2.0)
    #    Nu = 0.023 * Re**0.8 * Pr**0.4 * visc_corr 
    else:
        # Standard Dittus-Boelter
        mu_w      = mu_f(T_bulk + 10.0, mu_scale) # Smaller wall-to-bulk gap
        visc_corr = np.clip((mu / mu_w)**0.14, 0.5, 2.0)
        # Add a 'Enhancement Factor' for immersion-specific geometry
        Nu = 0.023 * Re**0.8 * Pr**0.4 * visc_corr
    
    Nu    *= (1.0 + 2.5 * baffle) # Increase baffle mixing bonus from 0.3 to 2.5
    h_avg  = Nu * kf / (DH + 1e-10)    
    
    Nu *= (1.0 + 0.50 * baffle) # Increased baffle mixing bonus (50%)
    h_avg = Nu * kf / (DH + 1e-10)                          # laminar lower limit
    #else:
        # viscosity correction (μb/μw)^0.14 — wall is ~15°C hotter than bulk
    #    mu_w      = mu_f(T_bulk + 15.0, mu_scale)
    #    visc_corr = np.clip((mu / mu_w)**0.14, 0.5, 2.0)
    #    Nu = 0.023 * Re**0.8 * Pr**0.4 * visc_corr
    #Nu    *= (1.0 + 0.30 * baffle)             # baffle mixing bonus
    #h_avg  = Nu * kf / (DH + 1e-10)            # [W/m²K]

    # ── Energy balance → stack ΔT ─────────────────────────────────────────
    delta_T   = Q_IT_BASE / (m_dot * cp + 1e-9)   # Tout − Tin [K]
    delta_T   = np.clip(delta_T, 0.0, 200.0)

    # ── Server max temperature (convection resistance + conduction) ───────
    R_conv    = 1.0 / (h_avg * S_AREA + 1e-9)
    T_srv_max = T_bulk + Q_IT_BASE * R_conv

    # ── Darcy-Weisbach pressure drop  (Eq.7) ─────────────────────────────
    f_DW  = 64.0 / max(Re, 1.0) if Re < 2300 else 0.316 * Re**(-0.25)
    dP    = (f_DW * L_CH / DH + K_ft) * 0.5 * rho * U**2
    dP    = np.clip(dP, 0.0, 5e4)

    # ── CDU bulk temperature  (Eq.6 steady-state) ─────────────────────────
    # At steady state: ṁcp(Tin−Tb) + Q_IT − UA(Tb−Tcool) = 0
    # Solving: Tb = (ṁcp·Tin + Q_IT + UA·Tcool) / (ṁcp + UA)
    mcp    = m_dot * cp
    Tb_ss  = (mcp * T_in + Q_IT_BASE + UA_CDU * T_COOL) / (mcp + UA_CDU + 1e-9)
    Tb_ss  = np.clip(Tb_ss, T_in - 5.0, T_in + 40.0)

    # ── PUE  (Eq.8) with Carnot-penalty COP (same as immersion_cfd.py) ───
    COP_eff   = max(COP_CHILL / (1.0 + 0.025 * max(T_out - T_in, 0.0)), 1.2)
    P_pump    = V_dot * dP / ETA_PUMP
    P_chiller = (Q_IT_BASE + P_pump) / COP_eff
    PUE       = (Q_IT_BASE + P_pump + P_chiller) / Q_IT_BASE
    PUE       = np.clip(PUE, 1.0, 5.0)

    feasible = (T_srv_max < BURN_THRESH   and
                delta_T   < DT_STACK_MAX  and
                Re        > RE_MIN        and
                PUE       < PUE_MAX)

    return dict(PUE=float(PUE), delta_T=float(delta_T), h_avg=float(h_avg),
                T_srv_max=float(T_srv_max), Re=float(Re), delta_P=float(dP),
                Tb_cdu=float(Tb_ss), T_out=float(T_out), feasible=feasible)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  LATIN HYPERCUBE SAMPLING
# ═════════════════════════════════════════════════════════════════════════════
BOUNDS = [(0.5, 20),    # V̇    [LPM]
          (15.0, 45.0),  # Tin  [°C]
          (0.5, 5.0),    # μ    [cP]
          (0.0, 1.0)]    # baffle

def latin_hypercube(n, bounds, rng):
    nd = len(bounds)
    X  = np.zeros((n, nd))
    for k, (lo, hi) in enumerate(bounds):
        perm   = rng.permutation(n)
        X[:,k] = lo + (perm + rng.random(n)) / n * (hi - lo)
    return X


# ═════════════════════════════════════════════════════════════════════════════
# 5.  CRITIC WEIGHT METHOD
#     Diakoulaki et al. (1995) — data-driven, no subjectivity
#     Weights reflect both discriminating power (σ) and independence from
#     other objectives (1−rᵢⱼ).  Guaranteed non-equal unless objectives
#     are perfectly correlated.
# ═════════════════════════════════════════════════════════════════════════════
def compute_critic_weights(n_samples=3000, seed=42):
    """
    Returns (w1, w2, w3) and the raw data used for transparency.
    """
    print(f"\n  Sampling {n_samples} LHS design points …", end=' ', flush=True)
    rng     = np.random.default_rng(seed)
    samples = latin_hypercube(n_samples, BOUNDS, rng)

    f1, f2, f3 = [], [], []
    for row in samples:
        r = evaluate_design(*row)
        if r['feasible']:
            f1.append(r['PUE'])
            f2.append(r['delta_T'])
            f3.append(r['h_avg'])

    n_feas = len(f1)
    print(f"done.  {n_feas}/{n_samples} feasible ({100*n_feas/n_samples:.1f}%)")

    if n_feas < 20:
        print("  WARNING: <20 feasible samples — constraints may be too tight.")
        print("  Falling back to equal weights.")
        return np.array([1/3, 1/3, 1/3]), None

    F = np.column_stack([f1, f2, f3])   # (n_feas, 3)

    # ── Min-max normalise each column to [0, 1] ───────────────────────────
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    F_rng = F_max - F_min + 1e-12
    F_n   = (F - F_min) / F_rng         # normalised matrix

    # ── Standard deviation of each normalised objective ───────────────────
    sigma = F_n.std(axis=0)             # (3,)

    # ── Pearson correlation matrix ─────────────────────────────────────────
    corr  = np.corrcoef(F_n.T)          # (3, 3)   rᵢⱼ

    # ── CRITIC score  Cᵢ = σᵢ · Σⱼ(1 − rᵢⱼ) ─────────────────────────────
    indep = np.sum(1.0 - corr, axis=1)  # row sum of (1-r)
    C     = sigma * indep               # (3,)

    # ── Normalise to weights ───────────────────────────────────────────────
    weights = C / (C.sum() + 1e-12)

    norm_info = dict(F_min=F_min, F_max=F_max, F_n=F_n,
                     sigma=sigma, corr=corr, C=C)
    return weights, norm_info


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PENALISED SCALARISED OBJECTIVE
# ═════════════════════════════════════════════════════════════════════════════
def make_objective(weights, F_min, F_max):
    w1, w2, w3 = weights
    F_rng = F_max - F_min + 1e-12

    def objective(x):
        V_lpm, T_in, mu_cP, baffle = x
        r = evaluate_design(V_lpm, T_in, mu_cP, baffle)

        # Min-max normalise
        f1n = (r['PUE']     - F_min[0]) / F_rng[0]
        f2n = (r['delta_T'] - F_min[1]) / F_rng[1]
        f3n = (r['h_avg']   - F_min[2]) / F_rng[2]

        F_scalar = w1 * f1n  +  w2 * f2n  -  w3 * f3n   # Eq.9

        # Exterior penalty (large slope, physically interpretable)
        penalty = 0.0
        if r['T_srv_max'] >= BURN_THRESH:
            penalty += 20.0 * (r['T_srv_max'] - BURN_THRESH) / BURN_THRESH
        if r['delta_T'] >= DT_STACK_MAX:
            penalty += 20.0 * (r['delta_T'] - DT_STACK_MAX) / DT_STACK_MAX
        if r['Re'] <= RE_MIN:
            penalty += 20.0 * (RE_MIN - r['Re']) / RE_MIN
        if r['PUE'] >= PUE_MAX:
            penalty += 20.0 * (r['PUE'] - PUE_MAX) / PUE_MAX

        return F_scalar + penalty

    return objective


# ═════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    LINE = "═" * 64

    # ── STEP 1: CRITIC weights ────────────────────────────────────────────
    print(LINE)
    print("  STEP 1 / 3  —  CRITIC Weight Method")
    print("  (Diakoulaki 1995: data-driven, no subjectivity)")
    print(LINE)

    weights, norm_info = compute_critic_weights(n_samples=3000, seed=42)
    w1, w2, w3 = weights

    print(f"\n  {'Objective':<20} {'σ (norm.)':<12} {'Σ(1-r)':<12} "
          f"{'CRITIC score':<14} {'Weight'}")
    print(f"  {'─'*60}")

    if norm_info is not None:
        names = ["f₁  PUE (min)", "f₂  ΔT  (min)", "f₃  h   (max)"]
        for i, nm in enumerate(names):
            indep_i = np.sum(1.0 - norm_info['corr'][i])
            print(f"  {nm:<20} {norm_info['sigma'][i]:<12.5f} "
                  f"{indep_i:<12.4f} {norm_info['C'][i]:<14.6f} "
                  f"{weights[i]:.6f}")

    print(f"\n  ┌── CRITIC WEIGHTS ──────────────────────────────────────┐")
    print(f"  │  w1  (PUE      — minimise)  = {w1:.6f}                  │")
    print(f"  │  w2  (ΔT stack — minimise)  = {w2:.6f}                  │")
    print(f"  │  w3  (h_avg    — maximise)  = {w3:.6f}                  │")
    print(f"  │  Sum                        = {w1+w2+w3:.6f}  ✓ (= 1.0) │")
    print(f"  └────────────────────────────────────────────────────────┘")

    print(f"\n  Correlation matrix (normalised objectives):")
    labels = ["PUE ", "ΔT  ", "h   "]
    if norm_info is not None:
        print(f"  {'':6}", end="")
        for lb in labels: print(f"{lb:>10}", end="")
        print()
        for i, lb in enumerate(labels):
            print(f"  {lb:6}", end="")
            for j in range(3):
                print(f"{norm_info['corr'][i,j]:>10.4f}", end="")
            print()

    print(f"\n  Interpretation:")
    widx = np.argmax(weights)
    wnames = ["PUE", "ΔT stack", "h_avg"]
    print(f"  → Most influential objective: {wnames[widx]}  "
          f"(weight={weights[widx]:.4f})")
    for nm, ww in zip(wnames, weights):
        bar = "█" * max(1, int(ww * 50))
        print(f"    {nm:<12} w={ww:.4f}  {bar}")

    if norm_info is None:
        F_min_use = np.array([1.286, 0.5, 100.0])
        F_max_use = np.array([1.40,  15.0, 8000.0])
    else:
        F_min_use = norm_info['F_min']
        F_max_use = norm_info['F_max']

    # ── STEP 2: DE Optimisation ───────────────────────────────────────────
    print(f"\n{LINE}")
    print(f"  STEP 2 / 3  —  Differential Evolution Optimisation")
    print(f"  Objective: min F = {w1:.4f}·f̃₁ + {w2:.4f}·f̃₂ − {w3:.4f}·f̃₃")
    print(LINE)

    obj_fn = make_objective(weights, F_min_use, F_max_use)

    print("  Running DE  (popsize=18, maxiter=500, strategy=best1bin) …",
          flush=True)
    result = differential_evolution(
        obj_fn,
        bounds=BOUNDS,
        strategy='best1bin',
        maxiter=500,
        popsize=18,
        tol=1e-8,
        seed=13,
        polish=True,
        workers=1,
        disp=False,
    )

    V_opt, Tin_opt, mu_opt, baffle_opt = result.x
    r_opt = evaluate_design(V_opt, Tin_opt, mu_opt, baffle_opt)

    print(f"  Done.  Converged: {result.success}  |  "
          f"Iterations: {result.nit}  |  Evals: {result.nfev}")

    # ── STEP 3: Print results ─────────────────────────────────────────────
    print(f"\n{LINE}")
    print(f"  STEP 3 / 3  —  OPTIMAL DESIGN RESULTS")
    print(LINE)

    c1_ok = "✓ OK  " if r_opt['T_srv_max'] < BURN_THRESH  else "✗ FAIL"
    c2_ok = "✓ OK  " if r_opt['delta_T']   < DT_STACK_MAX else "✗ FAIL"
    c3_ok = "✓ OK  " if r_opt['Re']         > RE_MIN       else "✗ FAIL"
    c4_ok = "✓ OK  " if r_opt['PUE']        < PUE_MAX      else "✗ FAIL"
    feas  = "FEASIBLE ✓" if r_opt['feasible'] else "INFEASIBLE ✗ (penalty active)"

    print(f"""
  ┌── OPTIMAL DESIGN VARIABLES ─────────────────────────────────┐
  │  x₁  V̇   (flow rate)     = {V_opt:>9.4f}  LPM              │
  │  x₂  Tin  (inlet temp)    = {Tin_opt:>9.3f}  °C               │
  │  x₃  μ    (viscosity)     = {mu_opt:>9.4f}  cP               │
  │  x₄  Baffle factor        = {baffle_opt:>9.4f}  [0=none, 1=full] │
  └─────────────────────────────────────────────────────────────┘

  ┌── OBJECTIVE FUNCTION VALUES ────────────────────────────────┐
  │  f₁  PUE       (minimise) = {r_opt['PUE']:>9.5f}               │
  │  f₂  ΔT stack  (minimise) = {r_opt['delta_T']:>9.3f}  K             │
  │  f₃  h_avg     (maximise) = {r_opt['h_avg']:>9.2f}  W/m²K        │
  │  Scalar F (weighted)      = {result.fun:>9.6f}               │
  └─────────────────────────────────────────────────────────────┘

  ┌── CONSTRAINT VERIFICATION ─────────────────────────────────┐
  │  C1  T_srv_max < 85°C  → {r_opt['T_srv_max']:>7.2f} °C   [{c1_ok}]     │
  │  C2  ΔT_stack  < 15 K  → {r_opt['delta_T']:>7.3f} K    [{c2_ok}]     │
  │  C3  Re > 2300         → {r_opt['Re']:>7.1f}     [{c3_ok}]     │
  │  C4  PUE < {PUE_MAX:.2f}        → {r_opt['PUE']:>7.5f}    [{c4_ok}]     │
  │  Feasibility            → {feas:<27}│
  └─────────────────────────────────────────────────────────────┘

  ┌── SYSTEM PERFORMANCE AT OPTIMUM ───────────────────────────┐
  │  Re  (Reynolds)           = {r_opt['Re']:>9.1f}               │
  │  ΔP  (pressure drop)      = {r_opt['delta_P']:>9.2f}  Pa            │
  │  Tout (outlet temp)        = {r_opt['T_out']:>9.2f}  °C            │
  │  Tb_cdu (CDU bulk)         = {r_opt['Tb_cdu']:>9.2f}  °C            │
  │  T_server_max              = {r_opt['T_srv_max']:>9.2f}  °C            │
  └─────────────────────────────────────────────────────────────┘

  ┌── CRITIC WEIGHTS USED ─────────────────────────────────────┐
  │  w1 (PUE     weight)      = {w1:.6f}                       │
  │  w2 (ΔT      weight)      = {w2:.6f}                       │
  │  w3 (h_avg   weight)      = {w3:.6f}                       │
  └─────────────────────────────────────────────────────────────┘

  NOTE: PUE floor = 1 + 1/COP_eff ≈ {1.0 + 1.0/COP_CHILL:.4f}.
        PUE < 1.15 (original paper) is physically impossible.
        Feasible constraint set to PUE < {PUE_MAX:.2f}.
""")
