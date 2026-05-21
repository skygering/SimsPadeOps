
import itertools
import jinja_sim_utils as ju
from pathlib import Path

# This set of simulations is to test the sensitivity of the phase budgets to length of the run,
# the tolerance of the budget, and the amplitude

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name
nx, ny, nz = 256, 256, 256
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
        time_budget_start = 100,
        tidx_dump_time_budget = 20000,
        tidx_compute_time_budget = 1,
        CFL = -1,
        do_time_budgets = True,
        time_budgetType = 0,
        do_multi_phase_budgets = True,
        phases = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = True,
        zLoc = 5.0,
        cT = 2.00,
        pitch_amplitude = 0,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "phase_budgets",
        # if not provided, default_inputs will be used
        build_folder = "build_opti_phase",
        queue = "spr"
    )
)

# zip these together
# (0) non-moving turbine time-avg, (1) moving turbine time-avg, (2) moving turbine multiple phase-avg, tol = 1 -> should error
# (3) moving turbine one phase-avg, tol = 1 -> should be same as time avg, (4) moving turbine multiple phase-avg, tol = 0.05
# (5) both time and multiple phase-avg, tol = 0.05 -> should be the same as (1) and (4), respectively
sa = [0.6]
sf = [0.2, 0.8]
phase_tol = [0.05]
tstop = [800]
n_hrs = [24]
running_iter = itertools.zip_longest(tstop, n_hrs)

dt = [0.01]
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 2.5)]

varied_inputs = itertools.product(sa, sf, phase_tol, running_iter, dt, filterWidth)
varied_header = ["surge_amplitude", "surge_freq", "phase_tol", "tstop", "n_hrs", "dt", "filterWidth"]

# for v in varied_inputs: 
#     print(v)                                   

# # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

