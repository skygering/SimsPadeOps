import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts

data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0005_X_SU_PI_Files")
rows, fields = mplts.get_sim_varied_params(sim_folder)
id, dt, cT, surge_freq, surge_amplitude, pitch_amplitude, nx, ny, nz, filterWidth = zip(*rows)

# Plot the CT' = 1 and 4 plots for Stationary turbines
# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("$C_p$ vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = h/D$)", y = 0.99)
# ax1.set_title("\n$C_T'$ = 1.0")
# ax2.set_title("\n$C_T'$ = 4.0")
# ax1.set_xlabel("Simulation Time")
# ax2.set_xlabel("Simulation Time")
# ax1.set_ylabel("$C_p$")
# zoom = (50, 250)

# ct1_sims = [0, 1, 2, 3]
# ct4_sims = [12, 13, 14, 15]

# mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
# ax1.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

# mplts.plot_requested_turb_power(ax2, sim_folder, ct4_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct4_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax2, 1.0, sim_folder, 0, zoom = zoom)

# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'static_grid_convergence_power_varied_CT_varied_filter_zoomed_out.png'))

# # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (64, 32, 32) for stationary turbines
# field = "u"
# runid = 12
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Stationary Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_stationary_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

# # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (256, 128, 128) for stationary turbines
# field = "u"
# runid = 14
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Stationary Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_stationary_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

# # Plot the CT' = 1 and 4 plots for Pitching turbines
# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("$C_p$ vs Simulation Time for Pitching Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = h/D$)", y = 0.99)
# ax1.set_title("\n$C_T'$ = 1.0")
# ax2.set_title("\n$C_T'$ = 4.0")
# ax1.set_xlabel("Simulation Time")
# ax2.set_xlabel("Simulation Time")
# ax1.set_ylabel("$C_p$")
# zoom = (50, 250)

# ct1_sims = [8, 9, 10, 11]
# ct4_sims = [20, 21, 22, 23]

# mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
# ax1.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

# mplts.plot_requested_turb_power(ax2, sim_folder, ct4_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct4_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax2, 1.0, sim_folder, 0, zoom = zoom)

# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'pitching_grid_convergence_power_varied_CT_varied_filter_zoomed_out.png'))

# # # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (64, 32, 32) for pitching turbines
field = "u"
runid = 9
save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
                                             set_plot_lims = True, suptitle = f"U Velocity Across Pitching Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
video_name = field + str(runid) + "_pitching_across_rotor.mp4"
mplts.film_instantaneous_field(save_folder, video_name = video_name)
# # # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (256, 128, 128) for pitching turbines
# # field = "u"
# # runid = 22
# # save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
# #                                              set_plot_lims = True, suptitle = f"U Velocity Across Pitching Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# # video_name = field + str(runid) + "_pitching_across_rotor.mp4"
# # mplts.film_instantaneous_field(save_folder, video_name = video_name)

# # # Plot the CT' = 1 and 4 plots for Surging turbines
# fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
# fig.set_figwidth(9)
# fig.suptitle("$C_p$ vs Simulation Time for Suring Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = h/D$)", y = 0.99)
# ax1.set_title("\n$C_T'$ = 1.0")
# ax2.set_title("\n$C_T'$ = 4.0")
# ax1.set_xlabel("Simulation Time")
# ax2.set_xlabel("Simulation Time")
# ax1.set_ylabel("$C_p$")
# zoom = (50, 250)

# ct1_sims = [4, 5, 6, 7]
# ct4_sims = [16, 17, 18, 19]

# mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
# ax1.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

# mplts.plot_requested_turb_power(ax2, sim_folder, ct4_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct4_sims], zoom = zoom)
# mplts.plot_theoretical_turb_power(ax2, 1.0, sim_folder, 0, zoom = zoom)

# fig.subplots_adjust(top=0.8)
# plt.savefig(os.path.join(sim_folder, 'surging_grid_convergence_power_varied_CT_varied_filter_zoomed_out.png'))

# Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (128, 64, 64) for surging turbines
# field = "u"
# runid = 17
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Surging Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_surging_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)
# # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (256, 128, 128) for surging turbines
# field = "u"
# runid = 18
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Surging Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_suring_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

# # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (512, 256, 256) for stationary turbines
# field = "u"
# runid = 15
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Stationary Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_stationary_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

# # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (512, 256, 256) for surging turbines
# field = "u"
# runid = 19
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Surging Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_surging_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

# # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (512, 256, 256) for pitching turbines
# field = "u"
# runid = 23
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, suptitle = f"U Velocity Across Pitching Rotor ($C_T'$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_pitching_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)