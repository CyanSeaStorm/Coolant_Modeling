"""
==============================================================
STABLE CFD-LITE IMMERSION COOLING SIMULATION
CONTINUOUS VERSION — RUNNING AVG PUE ADDED
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DOMAIN
# ============================================================

Nx, Ny = 120, 60
Lx, Ly = 1.2, 0.6

dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# ============================================================
# FLUID (METHYL MYRISTATE)
# ============================================================

rho = 850.0
cp = 2100.0
k = 0.15
nu = 3.5e-6
alpha = k / (rho * cp)

beta = 7e-4
g = 9.81

Tinlet = 300.0

# ============================================================
# FLOW
# ============================================================

U_in = 0.08
flow_rate = 0.5
CFL = 0.4

# ============================================================
# FIELDS
# ============================================================

u = np.zeros((Ny, Nx))
v = np.zeros((Ny, Nx))
p = np.zeros((Ny, Nx))
T = Tinlet * np.ones((Ny, Nx))

u[:, 0] = U_in

# ============================================================
# SERVERS
# ============================================================

server_mask = np.zeros_like(T, dtype=bool)

s1 = (X > 0.35) & (X < 0.45) & (Y > 0.2) & (Y < 0.35)
s2 = (X > 0.75) & (X < 0.85) & (Y > 0.2) & (Y < 0.35)

server_mask[s1] = True
server_mask[s2] = True

Q = np.zeros_like(T)
Q[server_mask] = 2e6 


#OPTIMIZATION 


# ============================================================
# NUMERICAL OPERATORS
# ============================================================

def lap(f):
    return (
        (np.roll(f, 1, 1) + np.roll(f, -1, 1) - 2 * f) / dx**2 +
        (np.roll(f, 1, 0) + np.roll(f, -1, 0) - 2 * f) / dy**2
    )

def upwind_x(phi, u):
    return np.where(
        u > 0,
        (phi - np.roll(phi, 1, 1)) / dx,
        (np.roll(phi, -1, 1) - phi) / dx
    )

def upwind_y(phi, v):
    return np.where(
        v > 0,
        (phi - np.roll(phi, 1, 0)) / dy,
        (np.roll(phi, -1, 0) - phi) / dy
    )

# ============================================================
# DASHBOARD
# ============================================================

plt.ion()
fig = plt.figure(figsize=(14, 9))

ax1 = plt.subplot2grid((3, 2), (0, 0))
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

Tin_hist, Tout_hist, dT_hist = [], [], []
PUE_hist = []
PUE_avg_hist = []

running_PUE_sum = 0.0

# ============================================================
# CONTINUOUS SIMULATION
# ============================================================

step = 0

while True:

    # ---------- CFL timestep ----------
    umax = np.max(np.abs(u)) + 1e-6
    vmax = np.max(np.abs(v)) + 1e-6
    dt = CFL * min(dx / umax, dy / vmax)

    # ---------- momentum ----------
    u_star = u + dt * (-u*upwind_x(u,u) - v*upwind_y(u,v) + nu*lap(u))

    buoy = g * beta * (T - Tinlet)

    v_star = v + dt * (-u*upwind_x(v,u) - v*upwind_y(v,v)
                       + nu*lap(v) + buoy)

    u_star[server_mask] = 0
    v_star[server_mask] = 0

    # ---------- pressure projection ----------
    div = (
        (np.roll(u_star,-1,1)-np.roll(u_star,1,1))/(2*dx)
        +(np.roll(v_star,-1,0)-np.roll(v_star,1,0))/(2*dy)
    )

    for _ in range(25):
        p = 0.75*p + 0.25*0.25*(
            np.roll(p,1,0)+np.roll(p,-1,0)+
            np.roll(p,1,1)+np.roll(p,-1,1)
            - rho*dx*dy/dt*div
        )

    u = u_star - dt/rho*(np.roll(p,-1,1)-np.roll(p,1,1))/(2*dx)
    v = v_star - dt/rho*(np.roll(p,-1,0)-np.roll(p,1,0))/(2*dy)

    u[server_mask] = 0
    v[server_mask] = 0

    # velocity limiter
    vel = np.sqrt(u**2 + v**2)
    mask = vel > 1.5
    u[mask] *= 1.5/vel[mask]
    v[mask] *= 1.5/vel[mask]

    # ---------- energy ----------
    T += dt*(-u*upwind_x(T,u) - v*upwind_y(T,v)
             + alpha*lap(T) + Q/(rho*cp))

    T[:,0] = Tinlet
    T = np.clip(T, 280, 450)

    # =====================================================
    # MEASUREMENTS
    # =====================================================

    Tin = np.mean(T[:,2])
    Tout = np.mean(T[:,-3])
    dT = Tout - Tin

    P_IT = 1200
    pump = 80*flow_rate
    chiller = max(0, rho*cp*flow_rate*dT*2e-5)
    PUE = (P_IT + pump + chiller)/P_IT

    # running average (O(1) update)
    running_PUE_sum += PUE
    PUE_avg = running_PUE_sum/(step+1)

    Tin_hist.append(Tin)
    Tout_hist.append(Tout)
    dT_hist.append(dT)
    PUE_hist.append(PUE)
    PUE_avg_hist.append(PUE_avg)

    if step % 20 == 0:
        print(
            f"[{step:06d}] "
            f"PUE={PUE:.3f} | "
            f"AvgPUE={PUE_avg:.3f} | "
            f"Tin={Tin:.2f}K | "
            f"Tout={Tout:.2f}K | "
            f"dT={dT:.2f}K"
        )

    # =====================================================
    # LIVE DASHBOARD
    # =====================================================

    if step % 30 == 0:

        ax1.clear()
        ax1.imshow(T, origin="lower",
                   extent=[0,Lx,0,Ly],
                   cmap="coolwarm",
                   vmin=290, vmax=360)

        ax1.contour(X,Y,server_mask,
                    levels=[0.5],
                    colors="black",
                    linewidths=2)

        ax1.set_title("Temperature Field")

        # velocity
        ax2.clear()
        skip = 5
        ax2.quiver(X[::skip,::skip],
                   Y[::skip,::skip],
                   u[::skip,::skip],
                   v[::skip,::skip],
                   scale=2.5)
        ax2.set_title("Velocity Field")

        # PUE + running avg
        ax3.clear()
        ax3.plot(PUE_hist, label="Instantaneous PUE")
        ax3.plot(PUE_avg_hist, linestyle="--",
                 label="Running Avg PUE")
        ax3.legend()
        ax3.set_title("PUE Evolution")

        # thermal history
        ax4.clear()
        ax4.plot(Tin_hist,label="Tin")
        ax4.plot(Tout_hist,label="Tout")
        ax4.plot(dT_hist,label="ΔT")
        ax4.legend()
        ax4.set_title("Thermal History")

        plt.pause(0.001)

    step += 1 
# ============================================================
# OPTIMIZER INTERFACE FUNCTION
# ============================================================

def run_simulation(flow_rate_input=0.5,
                   Tinlet_input=300.0,
                   steps=1200,
                   visualize=False):
    """
    Runs CFD for finite steps (used by optimizer).

    Returns performance metrics dictionary.
    """

    # ---------------- RESET DOMAIN ----------------
    global u, v, p, T
    global flow_rate, Tinlet
    global Tin_hist, Tout_hist, dT_hist
    global PUE_hist, PUE_avg_hist

    flow_rate = flow_rate_input
    Tinlet = Tinlet_input

    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    p = np.zeros((Ny, Nx))
    T = Tinlet * np.ones((Ny, Nx))

    u[:, 0] = U_in

    Tin_hist, Tout_hist, dT_hist = [], [], []
    PUE_hist = []
    PUE_avg_hist = []

    running_sum = 0.0

    # ---------------- TIME LOOP ----------------
    for step in range(steps):

        umax = np.max(np.abs(u)) + 1e-6
        vmax = np.max(np.abs(v)) + 1e-6
        dt = CFL * min(dx/umax, dy/vmax)

        # ----- momentum -----
        u_star = u + dt*(-u*upwind_x(u,u)
                         -v*upwind_y(u,v)
                         +nu*lap(u))

        buoy = g*beta*(T-Tinlet)

        v_star = v + dt*(-u*upwind_x(v,u)
                         -v*upwind_y(v,v)
                         +nu*lap(v)
                         +buoy)

        u_star[server_mask] = 0
        v_star[server_mask] = 0

        # ----- pressure -----
        div = (
            (np.roll(u_star,-1,1)-np.roll(u_star,1,1))/(2*dx)
            +(np.roll(v_star,-1,0)-np.roll(v_star,1,0))/(2*dy)
        )

        for _ in range(20):
            p[:] = 0.75*p + 0.25*0.25*(
                np.roll(p,1,0)+np.roll(p,-1,0)+
                np.roll(p,1,1)+np.roll(p,-1,1)
                - rho*dx*dy/dt*div
            )

        u[:] = u_star - dt/rho*(np.roll(p,-1,1)-np.roll(p,1,1))/(2*dx)
        v[:] = v_star - dt/rho*(np.roll(p,-1,0)-np.roll(p,1,0))/(2*dy)

        u[server_mask] = 0
        v[server_mask] = 0

        # velocity limiter
        vel = np.sqrt(u**2+v**2)
        mask = vel > 1.5
        u[mask] *= 1.5/vel[mask]
        v[mask] *= 1.5/vel[mask]

        # ----- energy -----
        T[:] += dt*(-u*upwind_x(T,u)
                    -v*upwind_y(T,v)
                    +alpha*lap(T)
                    +Q/(rho*cp))

        T[:,0] = Tinlet
        T[:] = np.clip(T,280,450)

        # ----- measurements -----
        Tin = np.mean(T[:,2])
        Tout = np.mean(T[:,-3])
        dT = Tout - Tin

        P_IT = 1200
        pump = 80*flow_rate
        chiller = max(0, rho*cp*flow_rate*dT*2e-5)
        PUE = (P_IT+pump+chiller)/P_IT

        running_sum += PUE
        PUE_avg = running_sum/(step+1)

        Tin_hist.append(Tin)
        Tout_hist.append(Tout)
        dT_hist.append(dT)
        PUE_hist.append(PUE)
        PUE_avg_hist.append(PUE_avg)

    # ---------------- RETURN METRICS ----------------

    metrics = {
        "PUE": float(np.mean(PUE_hist[-100:])),
        "PUE_avg": float(PUE_avg_hist[-1]),
        "deltaT": float(np.mean(dT_hist[-100:])),
        "Tmax": float(np.max(T)),
        "velocity_mean": float(np.mean(np.sqrt(u**2+v**2)))
    }

    return metrics    