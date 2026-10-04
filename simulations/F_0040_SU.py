import itertools
import jinja_sim_utils as ju
from pathlib import Path
import pandas as pd

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")
df = pd.read_csv("/scratch/10264/sgering/SimsPadeOps/simulations/F_0029_SU.csv")

# 1) Filter out rows where Av > 0.4  (keep Av <= 0.4)
df = df[df["Av"] <= 0.4].copy()
df = df[df["f"] >= 1.5]

# IMPORTANT: Reset index to avoid KeyError and SettingWithCopyWarning
df.reset_index(drop=True, inplace=True)

# 2) Swap f values:
df["f"] = df["f"].replace({
    1.5: 4,
    2: 8,
})

# Re-assign your series after filtering/remapping
CT_prime = df["CT_prime"]
f, Av = df["f"], df["Av"]
nx, ny, nz = df["nx"], df["ny"], df["nz"]
Lx, Ly, Lz = df["Lx"], df["Ly"], df["Lz"]
filterWidth, useCorrection = df["filterWidth"], df["useCorrection"]

f4_dt = 0.0025
f4_time = 77.5
f4_hrs = 9.0
f8_dt = 0.00125
f8_time = 76.25
f8_hrs = 12.0

dt, tstop, n_hrs = df["dt"].copy(), df["runtime"].copy(), df["n_hrs"].copy()
for i, f_val in enumerate(f):
    if f_val == 4:
        dt.iloc[i] = f4_dt
        tstop.iloc[i] = f4_time
        n_hrs.iloc[i] = f4_hrs
    elif f_val == 8:
        dt.iloc[i] = f8_dt
        tstop.iloc[i] = f8_time
        n_hrs.iloc[i] = f8_hrs

# Update back to dataframe
df["dt"] = dt
df["runtime"] = tstop
df["n_hrs"] = n_hrs

curr_script_name = Path(__file__).with_suffix('').name
single_inputs = dict(
    sim = dict(
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        CFL = -1,
        do_time_budgets = False,
        do_multi_phase_budgets = False,
        t_dataDump = 100,
    ),
    turb = dict(
        yLoc = Ly.iloc[0] / 2,
        zLoc = Lz.iloc[0] / 2,
        pitch_amplitude = 0,
    ),
    run = dict(
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "high_f_surge",
        build_folder = "build_opti_phase",
        queue = "spr",
    )
)

varied_inputs = itertools.zip_longest(CT_prime, Av, f, nx, ny, nz, Lx, Ly, Lz, dt, tstop, n_hrs, filterWidth, useCorrection)
varied_header = ["cT", "surge_amplitude", "surge_freq", "nx", "ny", "nz", "Lx", "Ly", "Lz", "dt", "tstop", "n_hrs", "filterWidth", "useCorrection"]

# for v in varied_inputs: 
#     print(v)                                

# # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

# ju.make_batched_sbatch_files( # batch all sims that take 6 hours
#     ju.DATA_PATH + curr_script_name + "_Files",
#     max_per_batch=12,
#     output_glob="*.out",
#     avg_hours = 6,
#     timeout_hours = 12,
#     sbatch_prefix="run_surge_6_hrs",
#     max_walltime_hours=18,
#     min_sim = 4,
#     max_sim = 8
# )

# ju.make_batched_sbatch_files( # batch all sims that take 5 hours
#     ju.DATA_PATH + curr_script_name + "_Files",
#     max_per_batch=12,
#     output_glob="*.out",
#     avg_hours = 5,
#     timeout_hours = 10,
#     sbatch_prefix="run_surge_5_hrs",
#     max_walltime_hours=18,
#     min_sim = 9,
#     max_sim = 28
# )

# ju.make_batched_sbatch_files( # batch all sims that take 4 hours
#     ju.DATA_PATH + curr_script_name + "_Files",
#     max_per_batch=12,
#     output_glob="*.out",
#     avg_hours = 4,
#     timeout_hours = 8,
#     sbatch_prefix="run_surge_4_hrs",
#     max_walltime_hours=18,
#     min_sim = 29,
#     max_sim = 34,
# )

