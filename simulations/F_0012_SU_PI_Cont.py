import itertools
import jinja_sim_utils as ju
from pathlib import Path

# From simulations 12, I realized that I needed to run even more simulations,
# but it was getting to complex to keep them all in the same file and not ruin
# the order of the simulations

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
        useCorrection = True,
        zLoc = 5.0
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "ct_effects",
        # if not provided, default_inputs will be used
        n_hrs = 1,
        queue = "spr",
    )
)

cT = [1.0]
sf = [0.1, 0.5, 1.0]
sa_old = [0.1, 0.3, 0.5, 1.0]
sa_new = [0.2, 0.7, 0.9]
pa_none = [0.0]

pa_old = [1.0, 3.0, 5.0, 10.0]
pa_new = [2.0, 7.0, 9.0]
sa_none = [0.0]

low_freq_dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)]
mid_freq_dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.5, single_inputs, v = 0.0, w = 0.0)]
high_freq_dt = [ju.find_min_dt(1.0, nx, ny, nz, 2.0, single_inputs, v = 0.0, w = 0.0)]

add_amp_surge_movement_iter = itertools.product(cT, sf, sa_new, pa_none, low_freq_dt)
add_freq_surge_movement_iter = itertools.product(cT, [1.5], sa_new + sa_old, pa_none, mid_freq_dt)

add_amp_pitch_movement_iter = itertools.product(cT, sf, sa_none, pa_new, low_freq_dt)
add_freq_pitch_movement_iter = itertools.product(cT, [1.5], sa_none, pa_new + pa_old, mid_freq_dt)


movement_iter = itertools.chain.from_iterable([add_amp_surge_movement_iter, add_freq_surge_movement_iter,
                                               add_amp_pitch_movement_iter, add_freq_pitch_movement_iter])


filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 1.5)]
varied_inputs = itertools.product(movement_iter, filterWidth)
varied_header = ["cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "filterWidth"]

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)

