import analysis_utils as au
import quick_metadata_plots as mplts
from pathlib import Path
import os
import math
import statistics
import padeopsIO as pio
# from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import pyGCS as pg
import quick_metadata_plots as mplts

data_path = Path(au.DATA_PATH)
sim_folder_1 = os.path.join(au.DATA_PATH, "F_0000_SU_X_Files")
sim_folder_2 = os.path.join(au.DATA_PATH, "F_0002_SU_PI_X_Files")

spatial = ["coarse", "medium", "fine"]
temporal = ["small", "large"]

def plot_requested_turb_power(ax, folder, runs, labels, zoom = None):
    for i, e in enumerate(runs):  # 
        run_folder = os.path.join(folder, "Sim_000" + str(e))
        sim = pio.BudgetIO(run_folder, padeops = True, runid = 1)
        dt = sim.input_nml["input"]["dt"]
        trans_tau = int(math.ceil(50 / dt) + 1)
        # TODO: ask Kirby if default should be "all", rather than None
        # as it was confusing and when it is saved as a file first, they all printed I think?
        power = sim.read_turb_power("all", turb=1)[trans_tau:]
        Cp = [au.power_to_Cp(p) for p in power]
        time = [50 + dt * n for n in range(len(Cp))]
        if zoom is not None:
            time, Cp = au.x_zoom_plot(zoom, time, Cp)
        ax.plot(time, Cp, label = labels[i], lw=0.7)
    return

def plot_theoretical_turb_power(ax, CT, folder, run, zoom = None):
    run_folder = os.path.join(folder, "Sim_000" + str(run))
    sim = pio.BudgetIO(run_folder, padeops = True, runid = 1)
    dt = sim.input_nml["input"]["dt"]
    trans_tau = int(math.ceil(50 / dt) + 1)
    power = sim.read_turb_power("all", turb=1)[trans_tau:]
    time = [50 + dt * n for n in range(len(power))]
    a = au.analytical_a(CT)
    Cp = [au.a_to_Cp(a)] * len(time)
    if zoom is not None:
        time, Cp = au.x_zoom_plot(zoom, time, Cp)
    ax.plot(time, Cp, label = "Analytical Cp", lw=0.7)
    return


# zoom = (150, 160)
# Plot static turbine Cp calcualted from output power
# fig, ax = plt.subplots(figsize=(9, 6))
# plot_requested_turb_power(ax, sim_folder_1, [0, 3, 6], [s + ", " + temporal[0] for s in spatial], zoom = zoom)
# plot_requested_turb_power(ax, sim_folder_2, [0, 2], [spatial[0] + ", " + temporal[1], spatial[2] + ", " + temporal[1]], zoom = zoom)
# plot_theoretical_turb_power(ax, 1.0, sim_folder_1, 0, zoom = zoom)
# plt.title("Static Turbine Cp vs Time")
# plt.legend(loc="lower right")
# plt.ylabel("Cp")
# plt.xlabel("Simulation Time")
# plt.savefig(os.path.join(sim_folder_2, 'static_suite_power.png'))
# Plot static turbine Cp calculated from output velocity
# fig, ax = plt.subplots(figsize=(9, 6))
# plot_theoretical_turb_power(ax, sim_folder_1, [0, 3, 6], [s + ", " + temporal[0] for s in spatial], zoom = zoom)
# plot_theoretical_turb_power(ax, sim_folder_2, [0, 2], [spatial[0] + ", " + temporal[1], spatial[2] + ", " + temporal[1]], zoom = zoom)
# plt.title("Static Turbine U vs Time")
# plt.legend(loc="lower right")
# plt.ylabel("U")
# plt.xlabel("Simulation Time")
# plt.savefig(os.path.join(sim_folder_2, 'static_suite_velocity.png'))

# Plot pitching turbine
# fig, ax = plt.subplots(figsize=(9, 6))
# plot_requested_turb_power(ax, sim_folder_1, [2, 5, 8], [s + ", " + temporal[0] for s in spatial], zoom = zoom)
# plot_requested_turb_power(ax, sim_folder_2, [1, 3], [spatial[0] + ", " + temporal[1], spatial[2] + ", " + temporal[1]], zoom = zoom)
# plt.title("Pitching Turbine (f = 1, degree = 5) Cp vs Time")
# plt.legend(loc="lower right")
# plt.xlabel("Simulation Time")
# plt.ylabel("Cp")
# plt.savefig(os.path.join(sim_folder_2, 'pitching_suite_power.png'))
run_folder = "/scratch/10264/sgering/Data/F_0002_SU_PI_X_Files/Sim_0001"
save_folder = os.path.join(run_folder, "u_plots")
# save_folder = mplts.plot_instantaneous_field(run_folder, 1, tidx = "all", field = "u")
mplts.film_instantaneous_field(save_folder)