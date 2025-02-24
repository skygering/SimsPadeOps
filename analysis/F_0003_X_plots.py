import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts
import padeopsIO as pio

data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0003_X_Files")

fig, ax = plt.subplots()
fig.suptitle("$C_p$ vs Simulation Time for Stationary Turbine\nwith Constant Grid Spacing ($C_T^' = 1$)")
zoom = (50, 250)

rows, fields = mplts.get_sim_varied_params(sim_folder)
ids, Ly, Lz, ny, nz = zip(*rows)
sims_idx = [2, 0, 3]  # note Ly = Lz = 5 seems to have lower power
mplts.plot_requested_turb_power(ax, sim_folder, sims_idx, [Ly[i] for i in sims_idx], zoom = zoom)
mplts.plot_theoretical_turb_power(ax, 1.0, sim_folder, 0, zoom = zoom)

ax.legend(loc = "center", title = "Ly and Lz [D]", fancybox = True)
plt.xlabel("Simulation Time")
plt.ylabel("Cp")
plt.savefig(os.path.join(sim_folder, 'static_diff_blockage_power.png'))

