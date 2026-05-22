import itertools
import jinja_sim_utils as ju
from pathlib import Path

# The purpose of this simulation is to run fixed-bottom simulations with static yaw and tilt
# with turbulence to try to get a nice cross section of the wake to compare the the k-l model
# in MITWindFarm with added tilt.

# I am going to run 4 simulations one with 0 yaw and tilt,
# one with non-zero yaw, one with non-zero tilt, and one
# with both yaw and tilt.

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
hit_template = ju.TEMPLATE_PATH.joinpath("hit_template.jinja")
interaction_template = ju.TEMPLATE_PATH.joinpath("interaction_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("hit_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name

Lx, Ly, Lz = 25, 10, 10
ny = 256  # HIT box must be a perfect square (Lx = Ly = Lz = 10 AND nx = ny = nz = 128 OR = 256)
nz = ny
nx = (ny / Ly) * Lx

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
        tstop = 400,
        CFL = 1.0,
        do_time_budgets = True,
        time_budgetType = 0,
        useDynamicTurbine = False,
        tidx_compute_time_budget = 5,
        tidx_dump_time_budget = 2500,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        zLoc = 5.0,
        useCorrection = True,
        cT = 1.33,
    ),
    hit = dict(
        # always need to provide the filepaths (no defaults)
        restartFile_TID = 0,
        restartFile_RID = 1,
        hit_inputdir = "/work2/08445/tg877441/shared_tmp/cube256_10D",
    ),
    interaction = dict(
        TI_target = -1.0,
        TI_fact = 0.3477662250832144,
        TI_xloc = 5.0,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "HIT_shear",
        job_name = "mem_test_sg",
        # if not provided, default_inputs will be used
        n_hrs = 14,
    )
)

# Varied yaw and tilt
yaw = [28, 0, 20]
tilt = [0, 28, 20]
position_budget_iter = itertools.zip_longest(yaw, tilt)

# simulation setup parameters
factor = 2.5
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = factor)]

varied_inputs = itertools.product(position_budget_iter, filterWidth)
varied_header = ["yaw", "tilt", "filterWidth"]

# for v in varied_inputs:
#     print(v)   

# # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template,
    hit_template = hit_template, interaction_template = interaction_template, node_cap = 8)