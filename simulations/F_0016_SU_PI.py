import itertools
import jinja_sim_utils as ju
from pathlib import Path

# From simulations 12, I knew that there was an increasing and then decreasing error for the UMM compared to LES at CT' = 1. I also found out that CT' = 4 isn't
# guarenteed to be stable, and also is very sensitive to PadeOps parameters. Also, just as an atrifact of running earlier simulations, I had a very weird, uneven grouping of
# simulations.

# I have decided to re-run everything with f = [0.2, 0.4, 0.6, 0.8, 1.0] and the surge amplitude's of A = [0.2, 0.4, 0.6, 0.8, 1.0] and pitch amplitude's of A = [4, 8, 12, 16, 20].
# I then have also decided to run everything for 2 - 4 values of CT' = [1.00, 1.33, 1.66, 2.0]. I should probably start to

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
        t_dataDump = 50,
        tstop = 300,
        CFL = -1,
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
        job_name = "ct_effects",
        # if not provided, default_inputs will be used
        n_hrs = 5,
        queue = "spr"
    )
)

cT = [1.33, 1.66, 2.0]

sf = [0.2, 0.4, 0.6, 0.8, 1.0]
sa = [0.2, 0.4, 0.6, 0.8, 1.0]
pa = [4.0, 8.0, 12.0, 16.0, 20.0]

surging_cases = itertools.product(sf, sa, [0.0])
pitching_cases = itertools.product(sf, [0.0], pa)
movement_iter = itertools.chain.from_iterable([surging_cases, pitching_cases])
dt = [ju.find_min_dt(1.0, nx, ny, nz, max(sf), single_inputs, v = 0.0, w = 0.0)]
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 2.5)]
varied_inputs_normal = itertools.product(cT, movement_iter, dt, filterWidth)

cT = [1.33]
small_sf = [0.05, 0.1]
small_sa = [0.05, 0.1]
small_pa = [0.5, 1.0]
small_surging_cases = itertools.product(small_sf, small_sa, [0.0])
small_pitching_cases = itertools.product(small_sf, [0.0], small_pa)
small_movement_iter = itertools.chain.from_iterable([small_surging_cases, small_pitching_cases])
varied_inputs_small = itertools.product(cT, small_movement_iter, dt, filterWidth)

varied_header = ["cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "filterWidth"]
varied_inputs = itertools.chain.from_iterable([varied_inputs_normal, varied_inputs_small])

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)

