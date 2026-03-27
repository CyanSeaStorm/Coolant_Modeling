"""
==============================================================
STABLE CFD-LITE IMMERSION COOLING SIMULATION
OPTIMIZER-COMPATIBLE VERSION
==============================================================

Acts as:
1) Importable CFD solver  (used by optimizer.py)
2) Standalone simulator   (for testing)

Main API:
    run_simulation(flow_rate, Tinlet, steps)
"""

import numpy as np

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
# FLUID — METHYL MYRISTATE
# ============================================================

rho = 850.0        # kg/m³
cp  = 2100.0       # J/kgK
k   = 0.15         # W/mK
nu  = 3.5e-6       # kinematic viscosity
alpha = k/(rho*cp)

beta = 7e-4
g = 9.81

U_in = 0.08
CFL = 0.4

# ============================================================
# SERVERS (IMMERSED HEAT SOURCES)
# ============================================================

server_mask = np.zeros((Ny, Nx), dtype=bool)

s1 = (X > 0.35) & (X < 0.45) & (Y > 0.2) & (Y < 0.35)
s2 = (X > 0.75) & (X < 0.85) & (Y > 0.2) & (Y < 0.35)

server_mask[s1] = True
server_mask[s2] = True

Q_base = np.zeros((Ny, Nx))
Q_base[server_mask] = 2e6  # W/m³ heat generation

# ============================================================
# NUMERICAL OPERATORS
# ============================================================

def lap(f):
    return (
        (np.roll(f,1,1)+np.roll(f,-1,1)-2*f)/dx**2 +
        (np.roll(f,1,0)+np.roll(f,-1,0)-2*f)/dy**2
    )

def upwind_x(phi,u):
    return np.where(
        u>0,
        (phi-np.roll(phi,1,1))/dx,
        (np.roll(phi,-1,1)-phi)/dx
    )

def upwind_y(phi,v):
    return np.where(
        v>0,
        (phi-np.roll(phi,1,0))/dy,
        (np.roll(phi,-1,0)-phi)/dy
    )

# ============================================================
# MAIN CFD FUNCTION (USED BY OPTIMIZER)
# ============================================================

def run_simulation(flow_rate=0.5, Tinlet=300.0, steps=1500):
    """
    Runs CFD-lite immersion cooling simulation.

    Returns dictionary:
        {
            "PUE",
            "Tmax",
            "deltaT",
            "h_avg"
        }
    """

    # ---------- fields ----------
    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    p = np.zeros((Ny, Nx))
    T = Tinlet*np.ones((Ny, Nx))

    u[:,0] = U_in * flow_rate
    Q = Q_base.copy()

    # =====================================================
    # TIME LOOP
    # =====================================================

    for step in range(steps):

        # ----- CFL timestep -----
        umax = np.max(np.abs(u)) + 1e-6
        vmax = np.max(np.abs(v)) + 1e-6
        dt = CFL * min(dx/umax, dy/vmax)

        # ----- momentum -----
        u_star = u + dt*(-u*upwind_x(u,u)
                         -v*upwind_y(u,v)
                         +nu*lap(u))

        buoy = g*beta*(T - Tinlet)

        v_star = v + dt*(-u*upwind_x(v,u)
                         -v*upwind_y(v,v)
                         +nu*lap(v)
                         +buoy)

        u_star[server_mask] = 0
        v_star[server_mask] = 0

        # ----- pressure projection -----
        div = (
            (np.roll(u_star,-1,1)-np.roll(u_star,1,1))/(2*dx)
            +(np.roll(v_star,-1,0)-np.roll(v_star,1,0))/(2*dy)
        )

        for _ in range(20):
            p[:] = 0.25*(
                np.roll(p,1,0)+np.roll(p,-1,0)+
                np.roll(p,1,1)+np.roll(p,-1,1)
                - rho*dx*dy/dt*div
            )

        u[:] = u_star - dt/rho*(np.roll(p,-1,1)-np.roll(p,1,1))/(2*dx)
        v[:] = v_star - dt/rho*(np.roll(p,-1,0)-np.roll(p,1,0))/(2*dy)

        u[server_mask] = 0
        v[server_mask] = 0

        # ----- velocity limiter -----
        vel = np.sqrt(u**2 + v**2)
        mask = vel > 1.5
        u[mask] *= 1.5/vel[mask]
        v[mask] *= 1.5/vel[mask]

        # ----- energy equation -----
        T[:] += dt*(-u*upwind_x(T,u)
                    -v*upwind_y(T,v)
                    +alpha*lap(T)
                    +Q/(rho*cp))

        T[:,0] = Tinlet
        T[:] = np.clip(T,280,450)

    # =====================================================
    # METRICS
    # =====================================================

    Tin  = np.mean(T[:,2])
    Tout = np.mean(T[:,-3])
    dT   = Tout - Tin

    P_IT = 1200
    pump = 80*flow_rate
    chiller = max(0, rho*cp*flow_rate*dT*2e-5)

    PUE = (P_IT + pump + chiller)/P_IT
    Tmax = np.max(T)

    # heat transfer proxy
    h_avg = k*dT/Lx

    return {
        "PUE": PUE,
        "Tmax": Tmax,
        "deltaT": dT,
        "h_avg": h_avg
    }

# ============================================================
# STANDALONE EXECUTION (SAFE)
# ============================================================

if __name__ == "__main__":

    print("\nRunning standalone CFD test...\n")

    metrics = run_simulation(
        flow_rate=0.5,
        Tinlet=300.0,
        steps=2000
    )

    print("Simulation finished.\n")
    print(metrics)