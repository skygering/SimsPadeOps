import itertools
import jinja_sim_utils as ju
from pathlib import Path

# The purpose of these simulations is to test out the reduced memory useafe of the budgets.

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("fixed_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name

Lx, Ly, Lz = 25, 10, 10
nx, ny, nz = 256, 128, 128

single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        # if not provided, default_inputs will be used
        Lx = Lx,
        Ly = Ly,
        Lz = Lz,
        nx = nx,
        ny = ny,
        nz = nz,
        t_dataDump = 100,
        tstop = 200,
        CFL = 1.0,
        useDynamicTurbine = False,
        tidx_compute_time_budget = 5,
        tidx_dump_time_budget = 3000,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        zLoc = 5.0,
        useCorrection = True,
        cT = 1.33,
        tilt = 0,
        yaw = 20,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "mem_test_sg",
        # if not provided, default_inputs will be used
        n_hrs = 2,
    )
)

# Varied yaw and tilt
do_time_budgets = [True, True, True, True, True, False, True]
time_budgetType = [0, 1, 2, 3, 4, 0, 0]
budget_iter = itertools.zip_longest(do_time_budgets, time_budgetType)
# simulation setup parameters
factor = 2.5
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = factor)]

varied_inputs = itertools.product(budget_iter, filterWidth)
varied_header = ["do_time_budgets", "time_budgetType", "filterWidth"]

# for v in varied_inputs:
#     print(v)   

# # # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 1)