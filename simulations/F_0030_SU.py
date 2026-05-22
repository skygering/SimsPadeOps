import itertools
import jinja_sim_utils as ju
from pathlib import Path
import pandas as pd
import numpy as np

# this set of simulations is to run a phase-budget on 2-3 different cases for surge (as defined in F_0029)

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

df = pd.read_csv("/scratch/10264/sgering/SimsPadeOps/simulations/F_0029_SU.csv")
nx, ny, nz = np.unique(df["nx"])[0], np.unique(df["ny"])[0], np.unique(df["nz"])[0]
Lx, Ly, Lz = np.unique(df["Lx"])[0], np.unique(df["Ly"])[0], np.unique(df["Lz"])[0]
dt = np.unique(df["dt"])[0]
filterWidth = np.unique(df["filterWidth"])[0]

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name
single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        # if not provided, default_inputs will be used
        nx = nx,
        ny = ny,
        nz = nz,
        Lx = Lx,
        Ly = Ly,
        Lz = Lz,
        time_budget_start = 100,
        tidx_dump_time_budget = 10000,
        tidx_compute_time_budget = 1,
        CFL = -1,
        dt = dt,
        do_time_budgets = True,
        time_budgetType = 0,
        do_multi_phase_budgets = True,
        phases = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        phase_tol = 0.05,
        tstop = 300,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        yLoc = Ly / 2,
        zLoc = Lz / 2,
        pitch_amplitude = 0,
        cT = 2.00,
        useCorrection = True,
        filterWidth = filterWidth,

    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "phase_budgets",
        # if not provided, default_inputs will be used
        build_folder = "build_opti_phase",
        queue = "spr",
        n_hrs = 12,
    )
)

sa = [0.3]
sf = [0.5] # switch back to 0.5
varied_inputs = itertools.zip_longest(sa, sf)
varied_header = ["surge_amplitude", "surge_freq"]

# for v in varied_inputs: 
#     print(v)                                   

# # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

