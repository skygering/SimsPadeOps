import itertools
import jinja_sim_utils as ju
from pathlib import Path

# This set of simulations is for my final project for 2.29. 
# I will be testing several modal analysis methods on velocity data from
# floating offshore wind turbine wakes.

# I need to run the simulations long enough that I can get good second-order wake data.
# For the SPOD, I also probably need ~2000 snapshots. I will save every 10 timesteps.
# Thus I will start with: 10 * 2000 * 0.065 = 1300 stop time.

import itertools
import jinja_sim_utils as ju
from pathlib import Path

# From simulations 10, I decided to stick with filter factor 1.5 with correctionsOn
# I now need to run simulations that explore the effects frequency and amplitude

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name
nx, ny, nz = 256, 128, 128
single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        # if not provided, default_inputs will be used
        nx = nx,
        ny = ny,
        nz = nz,
        Lx = 25,
        Ly = 10,
        Lz = 10,
        t_dataDump = 10,
        CFL = -1,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = True,
        zLoc = 5,

    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "modal_wake_runs",
        # if not provided, default_inputs will be used
        n_hrs = 7,
    )
)

# 4 different movement types: stationary, surging, pitching, and pitching + surging
sf = [0.0, 1.0, 1.0, 1.0]
sa = [0.0, 0.5, 0.0, 0.5]
pa = [0.0, 0.0, 5.0, 5.0]
movement_iter = itertools.zip_longest(sf, sa, pa)

dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)]
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 1.5)]
ctp = [1.0]
sim_params_iter = itertools.zip_longest(dt, filterWidth)

nx = [256]
ny = [128]
nz = [128]
resolution_iter = itertools.zip_longest(nx, ny, nz)

cT = [1.0]
tstop = [1500]

varied_inputs = itertools.product(tstop, movement_iter, sim_params_iter, cT, resolution_iter)
varied_header = ["tstop", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "filterWidth", "cT", "nx", "ny", "nz"]

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)

