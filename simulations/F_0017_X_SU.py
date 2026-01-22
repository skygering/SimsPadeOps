
import itertools
import jinja_sim_utils as ju
from pathlib import Path

# This set of simulations is to debug my new implementaiton of phase-averaged budgets. 
# The first set of experiments is to ensure I didn't break the time-averaged budgets both with and without turbine movement.
# The second set is to test out the phase_budgets. All of them use the new BUDGET_MULTI_PHASE_AVG namelist, which I have
# now added to my templates. 

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
        time_budget_start = 100, # put check to 100
        tidx_dump_time_budget = 3000, # put back to 3000
        tidx_compute_time_budget = 1,
        tstop = 250,
        CFL = -1,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = True,
        zLoc = 5.0,
        cT = 1.33,
        surge_amplitude = 0.4,
        pitch_amplitude = 0,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "testing_budgets",
        # if not provided, default_inputs will be used
        n_hrs = 4,
        build_folder = "build_opti_phase",
        queue = "skx"
    )
)

# zip these together
# (0) non-moving turbine time-avg, (1) moving turbine time-avg, (2) moving turbine multiple phase-avg, tol = 1 -> should error
# (3) moving turbine one phase-avg, tol = 1 -> should be same as time avg, (4) moving turbine multiple phase-avg, tol = 0.05
# (5) both time and multiple phase-avg, tol = 0.05 -> should be the same as (1) and (4), respectively
# (6) this is to test the wrap around calculations to ensure that they work properly
# (7) this is to ensure that the Budget1 terms work (in addition to the budget zero terms above)
sf = [0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
several_phases = [0.25, 0.5, 0.75, 1.0]
two_phases = [0.0, 1.0]
one_phase = [0.5]
sim_phases = [several_phases, several_phases, several_phases, one_phase, several_phases, several_phases, two_phases, several_phases]
do_time_budgets = [True, True, False, False, False, True, True, True]
do_multi_phase_budgets = [False, False, True, True, True, True, True, True]
phase_tol = [0.0, 0.0, 1.0, 1.0, 0.05, 0.05, 0.25, 0.05]
time_budgetType = [0, 0, 0, 0, 0, 0, 0, 1]
phase_movement_iter = itertools.zip_longest(sf, sim_phases, do_time_budgets, do_multi_phase_budgets, phase_tol, time_budgetType)

dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)]
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 2.5)]

varied_inputs = itertools.product(phase_movement_iter, dt, filterWidth)
varied_header = ["surge_freq", "phases", "do_time_budgets", "do_multi_phase_budgets", "phase_tol", "time_budgetType", "dt", "filterWidth"]

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 1)

