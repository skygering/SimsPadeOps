import itertools
import jinja_sim_utils as ju
from pathlib import Path

# After determining the TI_Fact from the H_0009 simulations, I will now re-run many
# of my experiments from the grid convergence tests, but with added turbulence.

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
hit_template = ju.TEMPLATE_PATH.joinpath("hit_template.jinja")
interaction_template = ju.TEMPLATE_PATH.joinpath("interaction_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("hit_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name

Lx, Ly, Lz = 25, 10, 10
nx, ny, nz = 320, 128, 128
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
        t_dataDump = 50,
        tstop = 400,
        CFL = -1,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        xloc = 5.0,
        useCorrection = False,
        cT = 1.0,
    ),
    hit = dict(
        # always need to provide the filepaths (no defaults)
        restartFile_TID = 0,
        restartFile_RID = 1,
        hit_inputdir = "/work2/08445/tg877441/shared_tmp/cube128_10D",
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
        job_name = "TI_test_sg",
        # if not provided, default_inputs will be used
        n_hrs = 5,
    )
)

# simulation setup parameters
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 1)] * 2
sf = [1.0] * 2
dt = [ju.find_min_dt(1.0, nx, ny, nz, sf, single_inputs, v = 0.0, w = 0.0)] * 2

# for surging
sa = [0.5]
pa = [0.0]

# for pitching
sa = sa + [0.0]
pa = pa + [5.0]

varied_inputs = itertools.zip_longest(dt, sf, sa, pa, filterWidth)
varied_header = ["dt", "surge_freq", "surge_amplitude", "pitch_amplitude", "filterWidth"]

# TODO: potentially add cases with filter on and/or filter factor or 1.5 to fill in the matrix

# for v in varied_inputs:
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template,
    hit_template = hit_template, interaction_template = interaction_template, node_cap = 12)