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

Lz = Lz * 2
nz = nz * 2

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
        CFL = -1,
        dt = dt,
        do_time_budgets = False,
        do_multi_phase_budgets = False,
        tstop = 100,
        t_dataDump = 100,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        yLoc = Ly / 2,
        zLoc = Lz / 2,
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
        n_hrs = 18,
    )
)

# (0) case with pitch + surge (that matches pitch)
# (1) case with just surge (that matches pitch)
# (2) case with tilt + pitch + surge
# (3) case with tilt + surge
L = 1 # assume height ~ rotor diameter
sf = [0.5, 0.5, 0.5, 0.5]
pa = [5.0, 0.0, 5.0, 0.0]
tilt = [0.0, 0.0, 10, 10] # can include shaft tilt + persistent pitch
pa_rad = np.deg2rad(pa[0])
sa_val = L * pa_rad * 2 * np.pi * sf[0]
sa = [sa_val, sa_val, sa_val, sa_val]
varied_inputs = itertools.zip_longest(sa, sf, pa, tilt)
varied_header = ["surge_amplitude", "surge_freq", "pitch_amplitude", "tilt"]

# for v in varied_inputs: 
#     print(v)                                   

# # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

