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
df = df[df["CT_prime"] == 1.33]

# 2) Swap CT_prime values:
#    1.33 -> 1
df["CT_prime"] = df["CT_prime"].replace({
    1.33: 1.0,
})

# (Optional) re-assign your series after filtering/remapping
CT_prime = df["CT_prime"]
f, Av = df["f"], df["Av"]
nx, ny, nz = df["nx"], df["ny"], df["nz"]
Lx, Ly, Lz = df["Lx"], df["Ly"], df["Lz"]
dt, tstop, n_hrs = df["dt"], df["runtime"], df["n_hrs"]
filterWidth, useCorrection = df["filterWidth"], df["useCorrection"]

curr_script_name = Path(__file__).with_suffix('').name
single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        # if not provided, default_inputs will be used
        CFL = -1,
        do_time_budgets = False,
        do_multi_phase_budgets = False,
        t_dataDump = 100,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        yLoc = Ly[0] / 2,
        zLoc = Lz[0] / 2,
        pitch_amplitude = 0,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "finalSurge",
        # if not provided, default_inputs will be used
        build_folder = "build_opti_phase",
        queue = "spr",
    )
)

varied_inputs = itertools.zip_longest(CT_prime, Av, f, nx, ny, nz, Lx, Ly, Lz, dt, tstop, n_hrs, filterWidth, useCorrection)
varied_header = ["cT", "surge_amplitude", "surge_freq", "nx", "ny", "nz", "Lx", "Ly", "Lz", "dt", "tstop", "n_hrs", "filterWidth", "useCorrection"]

# for v in varied_inputs: 
#     print(v)                                   

# # write needed simulation files
# ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
#     sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

ju.make_batched_sbatch_files( # batch all sims that take 6 hours
    ju.DATA_PATH + curr_script_name + "_Files",
    max_per_batch=12,
    output_glob="*.out",
    avg_hours = 6,
    timeout_hours = 12,
    sbatch_prefix="re_run_surge_6_hrs",
    max_walltime_hours=18,
    min_sim = 4,
    max_sim = 8
)

ju.make_batched_sbatch_files( # batch all sims that take 5 hours
    ju.DATA_PATH + curr_script_name + "_Files",
    max_per_batch=12,
    output_glob="*.out",
    avg_hours = 5,
    timeout_hours = 10,
    sbatch_prefix="re_run_surge_5_hrs",
    max_walltime_hours=20,
    min_sim = 9,
    max_sim = 28
)

ju.make_batched_sbatch_files( # batch all sims that take 4 hours
    ju.DATA_PATH + curr_script_name + "_Files",
    max_per_batch=12,
    output_glob="*.out",
    avg_hours = 4,
    timeout_hours = 8,
    sbatch_prefix="re_run_surge_4_hrs",
    max_walltime_hours=20,
    min_sim = 29,
    max_sim = 34,
)

