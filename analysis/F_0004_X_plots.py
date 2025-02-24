import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts
import padeopsIO as pio
import math
import numpy as np

data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0004_X_Files")
rows, fields = mplts.get_sim_varied_params(sim_folder)
ids, CT, nx, ny, nz, dt, filterWidth, useCorrection = zip(*rows)

# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("$C_p$ vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($C_T^' = 1$)", y = 0.99)
# ax1.set_title("\nNon-Const $\Delta$, Corrected")
# ax2.set_title("\n$\Delta = 0.08$, Corrected")
# ax3.set_title("\n$\Delta = 0.08$, NOT Corrected")
# ax1.set_xlabel("Simulation Time")
# ax2.set_xlabel("Simulation Time")
# ax3.set_xlabel("Simulation Time")
# plt.ylabel("Cp")
# zoom = (50, 75)

# rounded_filter_width = [round(float(width), 3) for width in filterWidth]
# # With filter varying with grid
# finished_sims_idx = [0, 1, 2]  # note the finest mesh it taking a long time to get off the queue
# mplts.plot_requested_turb_power(ax1, sim_folder, finished_sims_idx, [f"({nx[i]}, {ny[i]}, {nz[i]}), {rounded_filter_width[i]}" for i in finished_sims_idx], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
# ax1.legend(loc = "upper center", title = "(nx, ny, nz), $\Delta$", fancybox = True)
# # constant filter, correction on
# finished_sims_idx = [4, 5, 6]  # note the finest mesh it taking a long time to get off the queue
# mplts.plot_requested_turb_power(ax2, sim_folder, finished_sims_idx, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in finished_sims_idx], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax2, 1.0, sim_folder, 0, zoom = zoom)
# # constant filter, NOT corrected
# finished_sims_idx = [8, 9, 10]  # note the finest mesh it taking a long time to get off the queue
# mplts.plot_requested_turb_power(ax3, sim_folder, finished_sims_idx, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in finished_sims_idx], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax3, 1.0, sim_folder, 0, zoom = zoom)
# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'static_grid_convergence_power_filter_effects.png'))

# # New Figure
# fig, ax1 = plt.subplots(ncols=1, sharey = True)
# fig.set_figwidth(5)
# fig.suptitle("$C_p$ vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($C_T^' = 1$)", y = 0.99)
# ax1.set_title("Nx = 128, Ny = Nz = 64")
# # coarse run (128, 64, 64) with varrying filter widths + corrections
# finished_sims_idx = [12, 13, 14, 15, 16]  # note the finest mesh it taking a long time to get off the queue
# mplts.plot_requested_turb_power(ax1, sim_folder, finished_sims_idx, [f"{rounded_filter_width[i]}" for i in finished_sims_idx], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
# ax1.legend(loc = "upper right", title = "$\Delta$", fancybox = True)
# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'static_grid_convergence_power_filter_effects_coarse_example.png'))

# New Figure
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey = True)
fig.set_figwidth(9)
fig.suptitle("$C_p$ vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 0.08$)", y = 0.99)
ax1.set_title("\n$C_T'$ = 1.0")
ax2.set_title("\n$C_T'$ = 4.0")
ax3.set_title("\n$C_T'$ = 8.0")
ax1.set_xlabel("Simulation Time")
ax2.set_xlabel("Simulation Time")
ax3.set_xlabel("Simulation Time")
plt.ylabel("Cp")
zoom = (50, 150)

ct1_sims = [4, 5, 6]  # note the finest mesh it taking a long time to get off the queue
ct2_sims = [17, 18, 19]
ct3_sims = [21, 22, 23]

mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims], zoom = zoom)
mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)

mplts.plot_requested_turb_power(ax2, sim_folder, ct2_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct2_sims], zoom = zoom)
mplts.plot_theoretical_turb_power(ax2, 1.0, sim_folder, 0, zoom = zoom)

mplts.plot_requested_turb_power(ax3, sim_folder, ct3_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct3_sims], zoom = zoom)
mplts.plot_theoretical_turb_power(ax3, 1.0, sim_folder, 0, zoom = zoom)
ax3.legend(loc = "upper right", title = "$(nx, ny, nz)$", fancybox = True)

fig.subplots_adjust(top=0.8)
plt.savefig(os.path.join(sim_folder, 'static_grid_convergence_power_varied_CT.png'))


