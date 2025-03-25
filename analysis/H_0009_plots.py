import analysis_utils as au
import quick_metadata_plots as mplts
from pathlib import Path
import os

data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0009_X_SU_PI_HIT_Files")

log0 = "TI_test_sg_0000.o1721392"
log1 = "TI_test_sg_0001.o1721393"

mplts.plot_TI_vals(os.path.join(sim_folder + "/Sim_0000"), log0)
mplts.plot_TI_vals(os.path.join(sim_folder + "/Sim_0001"), log1)

fact0 = au.get_TI_fact(os.path.join(sim_folder + "/Sim_0000"), log0, 2500)
fact1 = au.get_TI_fact(os.path.join(sim_folder + "/Sim_0001"), log1, 3000)

print(fact0)
print(fact1)