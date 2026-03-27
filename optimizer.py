"""
==============================================================
IMMERSION COOLING OPTIMIZER
Research Version (CFD + PUE Optimization)
NOW WITH RUNTIME + ETA DISPLAY
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------------
# IMPORT CFD SOLVER
# ------------------------------------------------------------
from coolant import run_simulation


# ============================================================
# DESIGN SPACE
# ============================================================

FLOW_RANGE = np.linspace(0.1, 1.0, 10)
TIN_RANGE  = np.linspace(295, 310, 6)

TMAX_LIMIT = 358.0
DT_LIMIT   = 15.0

# objective weights
w1 = 1.0
w2 = 0.2
w3 = 0.1


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(metrics):
    PUE = metrics["PUE"]
    dT  = metrics["deltaT"]
    h   = metrics["h_avg"]
    return w1*PUE + w2*dT - w3*h


# ============================================================
# OPTIMIZATION LOOP
# ============================================================

results = []

best_obj = np.inf
best_design = None

total_runs = len(FLOW_RANGE) * len(TIN_RANGE)
run_id = 0

print("\n===== STARTING OPTIMIZATION =====\n")

# ---- timing start ----
start_time = time.time()

for flow in FLOW_RANGE:
    for Tin in TIN_RANGE:

        run_id += 1

        # ---------------- ETA calculation ----------------
        elapsed = time.time() - start_time

        if run_id > 1:
            avg_time = elapsed / (run_id - 1)
            remaining = avg_time * (total_runs - run_id + 1)
        else:
            avg_time = 0
            remaining = 0

        print("\n--------------------------------------------------")
        print(f"Run {run_id}/{total_runs}")
        print(f"Flow={flow:.2f} | Tin={Tin:.1f}K")
        print(f"Elapsed Time : {elapsed/60:.2f} min")
        print(f"Estimated Remaining : {remaining/60:.2f} min")
        print("--------------------------------------------------")

        # ---------------- CFD RUN ----------------
        metrics = run_simulation(
            flow_rate=flow,
            Tinlet=Tin,
            steps=1200
        )

        # ---------------- constraints ----------------
        if metrics["Tmax"] > TMAX_LIMIT:
            print("❌ Rejected: Tmax constraint violated")
            continue

        if metrics["deltaT"] > DT_LIMIT:
            print("❌ Rejected: ΔT constraint violated")
            continue

        obj = objective(metrics)

        results.append([
            flow,
            Tin,
            metrics["PUE"],
            metrics["Tmax"],
            metrics["deltaT"],
            metrics["h_avg"],
            obj
        ])

        print(
            f"✓ PUE={metrics['PUE']:.3f} | "
            f"Tmax={metrics['Tmax']:.2f}K | "
            f"dT={metrics['deltaT']:.2f} | "
            f"Obj={obj:.4f}"
        )

        if obj < best_obj:
            best_obj = obj
            best_design = (flow, Tin)


# ============================================================
# RESULTS PROCESSING
# ============================================================

total_time = time.time() - start_time

results = np.array(results)

if len(results) == 0:
    print("No feasible designs found.")
    exit()

flow_vals = results[:,0]
Tin_vals  = results[:,1]
PUE_vals  = results[:,2]
obj_vals  = results[:,-1]

print("\n==============================")
print("BEST DESIGN FOUND")
print("==============================")
print(f"Flow Rate : {best_design[0]:.3f}")
print(f"Tinlet    : {best_design[1]:.2f} K")
print(f"Objective : {best_obj:.4f}")
print(f"Total Runtime : {total_time/60:.2f} minutes")


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(flow_vals, PUE_vals)
plt.xlabel("Flow Rate")
plt.ylabel("PUE")
plt.title("PUE vs Flow Rate")

plt.subplot(1,2,2)
plt.scatter(flow_vals, obj_vals)
plt.xlabel("Flow Rate")
plt.ylabel("Objective")
plt.title("Objective Landscape")

plt.tight_layout()
plt.show()