import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts

data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0006_X_SU_PI_Files")
rows, fields = mplts.get_sim_varied_params(sim_folder)
id, dt, cT, surge_freq, surge_amplitude, pitch_amplitude, nx, ny, nz, filterWidth = zip(*rows)


# Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (512, 256, 256) for pitching turbines
# field = "u"
# runid = 19
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Stationary Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_stationary_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

# field = "u"
# runid = 22
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Surging Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_surging_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

# field = "u"
# runid = 25
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Pitching Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_pitching_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

# Plots for Stationary Turbines
# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("$C_p$ vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 5h/2D$)", y = 0.99)
# ax1.set_title("\n$C_T'$ = 1.0")
# ax2.set_title("\n$C_T'$ = 4.0")
# ax3.set_title("\n$C_T'$ = 6.0")
# ax1.set_xlabel("Simulation Time")
# ax2.set_xlabel("Simulation Time")
# ax3.set_xlabel("Simulation Time")
# ax1.set_ylabel("$C_p$")
# zoom = (50, 250)

# ct1_sims = [0, 1, 2]
# ct4_sims = [9, 10, 11]
# ct6_sims = [18, 19, 20]

# mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
# ax1.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

# mplts.plot_requested_turb_power(ax2, sim_folder, ct4_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct4_sims], zoom = zoom)
# # mplts.plot_theoretical_turb_power(ax2, 4.0, sim_folder, 9, zoom = zoom)

# mplts.plot_requested_turb_power(ax3, sim_folder, ct6_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct6_sims], zoom = zoom)
# # mplts.plot_theoretical_turb_power(ax3, 6.0, sim_folder, 18, zoom = zoom)

# fig, (ax2, ax3) = plt.subplots(ncols=2, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("Induced Velocity vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 5h/2D$)", y = 0.99)
# ax2.set_title("\n$C_T'$ = 4.0")
# ax3.set_title("\n$C_T'$ = 6.0")
# # ax1.set_xlabel("Simulation Time")
# ax2.set_xlabel("Simulation Time")
# ax3.set_xlabel("Simulation Time")
# ax2.set_ylabel("Induced Velocity ($u_\infty - u_{disk}$)")
# zoom = (50, 250)

# # ct1_sims = [0, 1, 2]
# ct4_sims = [9, 10, 11]
# ct6_sims = [18, 19, 20]

# # mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims], zoom = zoom)
# # mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
# # ax1.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

# mplts.plot_requested_turb_u_velocity(ax2, sim_folder, ct4_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct4_sims], zoom = zoom)
# # mplts.plot_theoretical_turb_power(ax2, 4.0, sim_folder, 9, zoom = zoom)

# mplts.plot_requested_turb_u_velocity(ax3, sim_folder, ct6_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct6_sims], zoom = zoom)
# # mplts.plot_theoretical_turb_power(ax3, 6.0, sim_folder, 18, zoom = zoom)

# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'static_induced_velocity_varied_CT_varied_filter_zoomed_out.png'))

# # Plots for Pitching Turbines
# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("$C_p$ vs Simulation Time for Pitching Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 5h/2D$)", y = 0.99)
# ax1.set_title("\n$C_T'$ = 1.0")
# ax2.set_title("\n$C_T'$ = 4.0")
# ax3.set_title("\n$C_T'$ = 6.0")
# ax1.set_xlabel("Simulation Time")
# ax2.set_xlabel("Simulation Time")
# ax3.set_xlabel("Simulation Time")
# ax1.set_ylabel("$C_p$")
# zoom = (150, 156)

# ct1_sims = [6, 7, 8]
# ct4_sims = [15, 16, 17]
# ct6_sims = [24, 25, 26]

# mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax1, 1, sim_folder, 6, zoom = zoom)
# ax1.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

# mplts.plot_requested_turb_power(ax2, sim_folder, ct4_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct4_sims], zoom = zoom)
# # mplts.plot_theoretical_turb_power(ax2, 4.0, sim_folder, 9, zoom = zoom)

# mplts.plot_requested_turb_power(ax3, sim_folder, ct6_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct6_sims], zoom = zoom)
# # mplts.plot_theoretical_turb_power(ax3, 6.0, sim_folder, 18, zoom = zoom)

# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'pitching_grid_convergence_power_varied_CT_varied_filter_zoomed_in.png'))


# Plots for Surging Turbines
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey = True)
fig.set_figwidth(9)
fig.suptitle("$C_p$ vs Simulation Time for Surging Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 5h/2D$)", y = 0.99)
ax1.set_title("\n$C_T'$ = 1.0")
ax2.set_title("\n$C_T'$ = 4.0")
ax3.set_title("\n$C_T'$ = 6.0")
ax1.set_xlabel("Simulation Time")
ax2.set_xlabel("Simulation Time")
ax3.set_xlabel("Simulation Time")
ax1.set_ylabel("$C_p$")
zoom = (150, 156)

ct1_sims = [3] #, 4, 5]
ct4_sims = [12, 13]# , 14]
ct6_sims = [21, 22]# , 23]

mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims], zoom = zoom)
mplts.plot_theoretical_turb_power(ax1, 1, sim_folder, 3, zoom = zoom)

mplts.plot_requested_turb_power(ax2, sim_folder, ct4_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct4_sims], zoom = zoom)
ax2.legend(loc = "center", title = "$(nx, ny, nz)$", fancybox = True)
# mplts.plot_theoretical_turb_power(ax2, 4.0, sim_folder, 9, zoom = zoom)

mplts.plot_requested_turb_power(ax3, sim_folder, ct6_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct6_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax3, 6.0, sim_folder, 18, zoom = zoom)

fig.subplots_adjust(top=0.8)
plt.savefig(os.path.join(sim_folder, 'surging_grid_convergence_power_varied_CT_varied_filter_zoomed_in.png'))