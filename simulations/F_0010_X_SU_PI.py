import itertools
import jinja_sim_utils as ju
from pathlib import Path

# After deciding to stick with resolution 256 x 128 x 128, I realized that I was missing a lot of simulations at
# various grid filter parameters. In order to fill in those blanks, I decided to run a few more simulations. 

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
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "grid_resolution_test_sg",
        # if not provided, default_inputs will be used
        n_hrs = 3,
    )
)

# simulation setup parameters
surging_sims = []
CFL = 1.0
factors = [1.0, 1.5, 2.5]
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = f) for f in factors]
n_surging = 7
n_pitching = 6
sf = [1.0] * (n_surging + n_pitching)
dt = [ju.find_min_dt(CFL, nx, ny, nz, sf, single_inputs, v = 0.0, w = 0.0)] * (n_surging + n_pitching)

# for surging
sa = [0.5] * n_surging
pa = [0.0] * n_surging

cT_prime = [1.0] * 4
filter_list = [filterWidth[0]] + [filterWidth[1]] + [filterWidth[2]] * 2
correctionOn = [True, False, False, True]

cT_prime = cT_prime + ([4.0] * 3)
filter_list = filter_list + [filterWidth[0]] + [filterWidth[1]] + [filterWidth[2]]
correctionOn = correctionOn + [True, False, True]

# for pitching
sa = sa + [0.0] * n_pitching
pa = pa + [5.0] * n_pitching

cT_prime = cT_prime + [1.0] * 3
filter_list = filter_list + [filterWidth[0]] + [filterWidth[1]] + [filterWidth[2]]
correctionOn = correctionOn + [True, False, True]

cT_prime = cT_prime + [4.0] * 3
filter_list = filter_list + [filterWidth[0]] + [filterWidth[1]] + [filterWidth[2]]
correctionOn = correctionOn + [True, False, True]

varied_inputs = itertools.zip_longest(dt, cT_prime, sf, sa, pa, filter_list, correctionOn)
varied_header = ["dt", "cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "filterWidth", "useCorrection"]

# TODO: potentially add stationary cases to fill in the matrix

# for v in varied_inputs:
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)