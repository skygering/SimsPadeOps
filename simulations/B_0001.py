import itertools
import jinja_sim_utils as ju
from pathlib import Path

# The purpose of this simulation is to run fixed-bottom simulations with static yaw and tilt
# to check my re-derivation of the UMM to include tilt. I will run a large sweep of yaw and tilt
# angles, from -30 to 30.

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("fixed_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name
nx, ny, nz = 256, 128, 128
single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        # if not provided, default_inputs will be used
        Lx = 25,
        Ly = 10,
        Lz = 10,
        nx = 256,
        ny = 128,
        CFL = 0.8,
        t_dataDump = 50,
        tstop = 250,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = True,
        zLoc = 5.0
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "yaw_tilt_sg",
        # if not provided, default_inputs will be used
        n_hrs = 1,
        queue = "spr"
    )
)
# Varied values
cT = [1.0, 4.0]
yaw = [0, 10, 20, 30]
tilt = [0, 10, 20, 30]

# simulation setup parameters
factor = 1.5
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = factor)]

varied_inputs = itertools.product(cT, yaw, tilt, filterWidth)
varied_header = ["cT", "yaw", "tilt", "filterWidth"]

# for v in varied_inputs:
#     print(v)   
# # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)