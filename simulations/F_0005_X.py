import itertools
import jinja_sim_utils as ju
from pathlib import Path

# The purpose of this simulation is to re-run some of the grid convergence work in the F_0004 series
# but use CT as an input, rather than CT'. This is meant to remove the effect of the thrust feedback,
# which should allow us to better understand the results of F_0004 series. 
# I will run a series of CT values: 1.0, 1.2, 1.5, and 4.0

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name

single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        # if not provided, default_inputs will be used
        Lx = 25,
        Ly = 10,
        Lz = 10,
        CFL = -1.0,
        t_dataDump = 50,
        ADM_Type = 6,  # takes in CT, but turbine can't move
        tstop = 150,
        # TODO: calculate dt so that it "matches" other simulatuions
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        # TODO: set CT, filterWidth, useCorrection
        filterWidth = 0.08,
        useCorrection = True
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "grid_resolution_test_sg",
        # if not provided, default_inputs will be used
        n_hrs = 4,
    )
)
# Varied values
cT = [1.0] #[1.0, 1.2, 1.5, 4.0]

# simulation setup parameters
nx = [128]
ny = [64]
nz = ny
CFL = 1.0
sf = 0.0
dt = [ju.find_min_dt(CFL, nx[i], ny[i], nz[i], sf, single_inputs) for i in range(len(nx))]

varied_header = ["cT", "nx", "ny", "nz", "dt"]
varied_inputs = itertools.product(cT, itertools.zip_longest(nx, ny, nz, dt))
# # write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)