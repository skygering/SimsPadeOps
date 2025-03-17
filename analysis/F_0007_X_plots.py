import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts

data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0007_X_Files")
rows, fields = mplts.get_sim_varied_params(sim_folder)
id, cT, nx, ny, nz, dt, filterWidth = zip(*rows)


# # Plot the u velocity across the rotor for CT' = 4, (nx, ny, nz) = (512, 256, 256) for pitching turbines
# field = "u"
# runid = 6
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0, suptitle = f"U Velocity Across Stationary Rotor ($C_T$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_stationary_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
fig.set_figwidth(9)
fig.suptitle("$C_p$ vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 5h/2D$)", y = 0.99)
ax1.set_title("\n$C_T$ = 1.0")
ax2.set_title("\n$C_T$ = 1.2")
ax1.set_xlabel("Simulation Time")
ax2.set_xlabel("Simulation Time")
ax1.set_ylabel("$C_p$")
# zoom = (50, 250)

ct1_sims = [0, 1, 2]
ct2_sims = [6, 7, 8]

mplts.plot_requested_turb_power(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims])
# mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
ax1.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

mplts.plot_requested_turb_power(ax2, sim_folder, ct2_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct2_sims])
ax2.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

fig.subplots_adjust(top=0.8)
plt.savefig(os.path.join(sim_folder, 'static_grid_convergence_power_varied_CT_NOT_PRIME_varied_filter.png'))

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
fig.set_figwidth(9)
fig.suptitle("Induced Velocity vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($\Delta = 5h/2D$)", y = 0.99)
ax1.set_title("\n$C_T$ = 1.0")
ax2.set_title("\n$C_T$ = 1.2")
ax1.set_xlabel("Simulation Time")
ax2.set_xlabel("Simulation Time")
ax1.set_ylabel("Induced Velocity")
# zoom = (50, 250)

ct1_sims = [0, 1, 2]
ct2_sims = [6, 7, 8]

mplts.plot_requested_turb_u_velocity(ax1, sim_folder, ct1_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct1_sims])
ax1.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

mplts.plot_requested_turb_u_velocity(ax2, sim_folder, ct2_sims, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in ct2_sims])
ax2.legend(loc = "upper center", title = "$(nx, ny, nz)$", fancybox = True)

fig.subplots_adjust(top=0.8)
plt.savefig(os.path.join(sim_folder, 'static_induced_velocity_varied_CT_NOT_PRIME_varied_filter.png'))

# Plot the u velocity across the rotor for CT = 1.2, (nx, ny, nz) = (256, 128, 128) for pitching turbines
# field = "u"
# runid = 7
# save_folder = mplts.plot_instantaneous_field(sim_folder, runid, tidx = "all", field = "u", xlim = 0,
#                                              set_plot_lims = True, plot_ylim = [-0.25, 1.25], suptitle = f"U Velocity Across Stationary Rotor ($C_T$ = {cT[runid]}, (nx, ny, nz) = {nx[runid]}, {ny[runid]}, {nz[runid]})")
# video_name = field + str(runid) + "_stationary_across_rotor.mp4"
# mplts.film_instantaneous_field(save_folder, video_name = video_name)

