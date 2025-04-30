import itertools
import jinja_sim_utils as ju
from pathlib import Path

# From simulations 10, I decided to stick with filter factor 1.5 with correctionsOn
# I now need to run simulations that explore the effects of CT'

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
        tstop = 250,
        CFL = -1,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = True
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "ct_ffects",
        # if not provided, default_inputs will be used
        n_hrs = 2,
    )
)

cT_prime = [2.0, 3.0]

sf = [1.0, 1.0]
sa = [0.5, 0.0]
pa = [0.0, 5.0]
dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)] * 2
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 1.5)] * 2

turbine_movement_iter = itertools.zip_longest(dt, sf, sa, pa, filterWidth)

varied_inputs = itertools.product(cT_prime, turbine_movement_iter)
varied_header = ["cT", "dt", "surge_freq", "surge_amplitude", "pitch_amplitude", "filterWidth"]

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)

