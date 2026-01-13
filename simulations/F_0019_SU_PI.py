import itertools
import jinja_sim_utils as ju
from pathlib import Path

# This set is to test Kirby's "fast" ADM PR. I can also compare with F_0015_SU_PI.py as it
# has the same simulations, just with a different build folder.

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
        t_dataDump = 50,
        tstop = 200,
        CFL = -1,

        do_time_budgets = False,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = True,
        zLoc = 5.0,
        cT = 1.33,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "kirby_PR_test",
        # if not provided, default_inputs will be used
        n_hrs = 1,
        queue = "skx",
    )
)

sf = [0.0, 0.25, 0.25]
sa = [0.0, 0.25, 0.0]
pa = [0.0, 0.0, 5.0]
build_folders = ["build_opti_fast", "build_opti"]

dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)]

import itertools

movement_iter = list(itertools.zip_longest(sf, sa, pa))
filterWidth = [ju.find_filter_width(single_inputs, nx=nx, ny=ny, nz=nz, factor=1.5)]
initial_runs = list(itertools.product(build_folders, movement_iter, dt, filterWidth))

# duplicate the runs
all_runs = initial_runs * 3   # length = 18
all_runs = all_runs[:-3]
flags = [False] * 6 + [True] * 6
flags += [True] * 3 # re-run the last set due to updated code
varied_inputs = [
    (flag, build, sf_i, sa_i, pa_i, dt_i, filter_width)
    for flag, (build, (sf_i, sa_i, pa_i), dt_i, filter_width)
    in zip(flags, all_runs)
]
varied_header = ["do_time_budgets", "build_folder", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "filterWidth"]

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)
