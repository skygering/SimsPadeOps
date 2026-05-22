
import itertools
import jinja_sim_utils as ju
from pathlib import Path
import pandas as pd

# This set of simulations is to test the sensitivity of the phase budgets to length of the run,
# the tolerance of the budget, and the amplitude

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

df = pd.read_csv("/scratch/10264/sgering/SimsPadeOps/simulations/F_0025_SU.csv")
CT_prime = df["CT_prime"]
f, Av = df["f"], df["Av"]
nx, ny, nz = df["nx"], df["ny"], df["nz"]
Lx, Ly, Lz = df["Lx"], df["Ly"], df["Lz"]
dt, tstop = df["dt"], df["runtime"]
filterWidth, useCorrection = df["filterWidth"], df["useCorrection"]
# file name based on current script name
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
        t_dataDump = 50,
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
        job_name = "filterWidthTest",
        # if not provided, default_inputs will be used
        build_folder = "build_opti",
        queue = "spr",
        n_hrs = 2,
    )
)

CT_prime = df["CT_prime"]
f, Av = df["f"], df["Av"]
nx, ny, nz = df["nx"], df["ny"], df["nz"]
Lx, Ly, Lz = df["Lx"], df["Ly"], df["Lz"]
dt, tstop = df["dt"], df["runtime"]
filterWidth, useCorrection = df["filterWidth"], df["useCorrection"]

varied_inputs = itertools.zip_longest(CT_prime, Av, f, nx, ny, nz, Lx, Ly, Lz, dt, tstop, filterWidth, useCorrection)
varied_header = ["cT", "surge_amplitude", "surge_freq", "nx", "ny", "nz", "Lx", "Ly", "Lz", "dt", "tstop", "filterWidth", "useCorrection"]

# for v in varied_inputs: 
#     print(v)                                   

# # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

