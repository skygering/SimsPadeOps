import itertools
import jinja_sim_utils as ju
from pathlib import Path

# This set of simulations is for my final project for 2.29. 
# I will be testing several modal analysis methods on velocity data from
# floating offshore wind turbine wakes.

# I have decided to focus on the convergence of the methods spatially and temporally.

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
        Lx = 25,
        Ly = 5,
        Lz = 5,
        t_dataDump = 10,
        CFL = -1,
        tstop = 1500,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        yLoc = 2.5,
        useCorrection = True,
        surge_freq = 0.1,
        surge_amplitude = 0.5,
        pitch_amplitude = 15,
        cT = 1.0,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "modal_convergence",
        # if not provided, default_inputs will be used
        n_hrs = 10,
        queue = "spr"
    )
)

nx = [128, 256, 512]
ny = [64, 128, 256]
nz = [64, 128, 256]
filterWidth = [ju.find_filter_width(single_inputs, nx = nx[i], ny = ny[i], nz = nz[i], factor = 2.0) for i in range(len(nx))]
dt = [ju.find_min_dt(1.0, nx[i], ny[i], nz[i], 0.1, single_inputs, v = 0.0, w = 0.0) for i in range(len(nx))]
dt = dt + [0.5 * t for t in dt] + [0.25 * t for t in dt] + [2.0 * t for t in dt]
nt = int(len(dt) / len(nx))
varied_inputs = itertools.zip_longest(nx * nt, ny * nt, nz * nt, dt, filterWidth * nt)
varied_header = ["nx", "ny", "nz", "dt", "filterWidth"]

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

