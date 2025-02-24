import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts
import padeopsIO as pio
import math

data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0004_X_Files")

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey = True)
fig.suptitle("$C_p$ vs Simulation Time for Stationary Turbine\nwith Lx = 25D and Ly = Lz = 10D ($C_T^' = 1$)", y = 0.99)
ax1.set_title("\nNon-Constant $\Delta$")
ax2.set_title("\n$\Delta = 0.08$")
ax1.set_xlabel("Simulation Time")
ax2.set_xlabel("Simulation Time")
plt.ylabel("Cp")
zoom = (50, 150)

rows, fields = mplts.get_sim_varied_params(sim_folder)
ids, CT, nx, ny, nz, dt, filterWidth = zip(*rows)
# With filter varying with grid
finished_sims_idx = [0, 1, 2]  # note the finest mesh it taking a long time to get off the queue
mplts.plot_requested_turb_power(ax1, sim_folder, finished_sims_idx, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in finished_sims_idx], zoom = zoom)
mplts.plot_theoretical_turb_power(ax1, 1.0, sim_folder, 0, zoom = zoom)
ax1.legend(loc = "upper center", title = "(nx, ny, nz)", fancybox = True)
# constant filter
finished_sims_idx = [4, 5, 6]  # note the finest mesh it taking a long time to get off the queue
mplts.plot_requested_turb_power(ax2, sim_folder, finished_sims_idx, [f"({nx[i]}, {ny[i]}, {nz[i]})" for i in finished_sims_idx], zoom = zoom)
mplts.plot_theoretical_turb_power(ax2, 1.0, sim_folder, 0, zoom = zoom)

fig.subplots_adjust(top=0.8)
plt.savefig(os.path.join(sim_folder, 'static_grid_convergence_power_filter_effects.png'))

