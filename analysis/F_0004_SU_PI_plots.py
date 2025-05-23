import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts
import padeopsIO as pio
import math
import numpy as np

data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0004_SU_PI_Files")
rows, fields = mplts.get_sim_varied_params(sim_folder)
id, cT, nx, ny, nz, dt, filterWidth, useCorrection, surge_amplitude, pitch_amplitude = zip(*rows)
zoom = (80, 100)

# New Figure
# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("$C_p$ vs Simulation Time for Moving Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 0.08$)", y = 0.99)
# ax1.set_title("\n $C_T'$ = 1.0, Surging (A = 1.0, f = 0.5)")
# ax2.set_title("\n $C_T'$ = 4.0. Surging (A = 1.0, f = 0.5)")
# ax1.set_xlabel("Simulation Time")
# ax2.set_xlabel("Simulation Time")
# ax1.set_ylabel("$C_p$")
zoom = (80, 85)

# ax1_sims = [2, 4, 6]  # surging, CT = 1.0
# ax2_sims = [10, 12, 14]  # surging, CT = 4.0

# mplts.plot_requested_turb_power(ax1, sim_folder, ax1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ax1_sims], zoom = zoom, alpha = 0.7, linestyle = "dashed")
# mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
# ax1.legend(loc = "upper right", title = "$(nx, ny, nz)$", fancybox = True)

# mplts.plot_requested_turb_power(ax2, sim_folder, ax2_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ax2_sims], zoom = zoom, alpha = 0.7, linestyle = "dashed")
# # mplts.plot_theoretical_turb_power(ax2, 1.0, sim_folder, 0, zoom = zoom)
# # ax2.legend(loc = "lower left", title = "$(nx, ny, nz)$", fancybox = True)

# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'surging_grid_convergence_power_varied_CT_zoomed.png'))

# fig, (ax3, ax4) = plt.subplots(ncols=2, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("$C_p$ vs Simulation Time for Moving Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 0.08$)", y = 0.99)
# ax3.set_title("\n $C_T'$ = 1.0, Pitching (A = 1.0, $\circ$ = 5.0)")
# ax4.set_title("\n $C_T'$ = 4.0, Pitching (A = 1.0, $\circ$ = 5.0)")
# ax3.set_xlabel("Simulation Time")
# ax4.set_xlabel("Simulation Time")
# ax3.set_ylabel("$C_p$")

# ax3_sims = [3, 5, 7]  # pitching, CT = 1.0
# ax4_sims = [11, 13, 15]  # pitching, CT = 4.0

# mplts.plot_requested_turb_power(ax3, sim_folder, ax3_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ax3_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax3, 1.0, sim_folder, 0, zoom = zoom)
# ax3.legend(loc = "lower right", title = "$(nx, ny, nz)$", fancybox = True)

# mplts.plot_requested_turb_power(ax4, sim_folder, ax4_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ax4_sims], zoom = zoom)
# # mplts.plot_theoretical_turb_power(ax4, 1.0, sim_folder, 0, zoom = zoom)
# ax3.legend(loc = "lower left", title = "$(nx, ny, nz)$", fancybox = True)

# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'pitching_grid_convergence_power_varied_CT_zoomed.png'))


# plotting (128, 64, 64) Cp with different filter widths to check odd behavior
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
fig.set_figwidth(9)
fig.suptitle("$C_p$ vs Simulation Time for Pitching Turbine\nwith Lx = 25D and Ly = Lz = 10D (nx = 128 and ny = nz = 64)", y = 0.99)
coarse_pitch_sims_correction = [22, 16, 17, 18]
coarse_pitch_sims_no_correction = [23, 19, 20, 21]

ax1.set_title("With Correction")
h = math.sqrt((25 / 128)**2 + 2 * (10 / 64)**2)
labels = ["0.08"] + [f"{float(filterWidth[i]) / h}h" for i in coarse_pitch_sims_correction[1:]]
mplts.plot_requested_turb_power(ax1, sim_folder, coarse_pitch_sims_correction, labels, zoom = zoom)
mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
ax1.legend(loc = "upper center", title = "Filter Width", fancybox = True)

ax2.set_title("No Correction")
labels = ["0.08"] + [f"{float(filterWidth[i]) / h}h" for i in coarse_pitch_sims_no_correction[1:]]
mplts.plot_requested_turb_power(ax2, sim_folder, coarse_pitch_sims_no_correction, [f"$\Delta$ = {round(float(filterWidth[i]), 3)}" for i in coarse_pitch_sims_no_correction], zoom = zoom)
mplts.plot_theoretical_turb_power(ax2, 1.0, sim_folder, 0, zoom = zoom)

fig.subplots_adjust(top=0.8)
plt.savefig(os.path.join(sim_folder, 'pitching_coarse_varied_filter_check.png'))

# plotting the (128, 64, 64) u across the turbine with different filter widths to check odd behavior
# field = "u"
# for runid in [16]:
#     save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,  suptitle = f"Pitching Rotor Velocity: $C_T$ = {cT[runid]}, (nx, ny, nz) = ({nx[runid]}, {ny[runid]}, {nz[runid]})")
#     video_name = field + str(runid) + "_across_rotor.mp4"
#     mplts.film_instantaneous_field(save_folder, video_name = video_name)

# plot turbine tilt
# pitching_log = "/scratch/10264/sgering/Data/F_0004_SU_PI_Files/Sim_0007/grid_resolution_test_sg_0007.o1628466"
# pitch_tilt = pio.query_logfile(pitching_log, search_terms = ["tilt"])["tilt"]
# n = len(pitch_tilt)
# pitch_tilt = pitch_tilt[round(n * 0.5):round(n * 0.6)]
# fig, ax = plt.subplots()
# ax.plot(pitch_tilt)
# ax.set_title("Turbine Tilt for Domain (nx, ny, nz) = (512,256,256) and $C_T = 1$")
# ax.set_xlabel("Simulation Time")
# ax.set_ylabel("Tilt (degs.)")
# plt.savefig(os.path.join(sim_folder, 'pitching_tilt_over_time.png'))