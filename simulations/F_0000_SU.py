import jinja_sim_utils as ju
import os
from pathlib import Path
import itertools

# TODO: need to add in scratch path so the run scripts have the right file path

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
        tstop = 250,
        t_dataDump = 50,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        surge_freq = 0.15,
        surge_amplitude = 0.25,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "surge_test_sg",
        # if not provided, default_inputs will be used
        n_hrs = 4,
    )
)

# varied_inputs = dict(sim = dict(dt = [0.2, 0.04]),  # big and small timesteps (max f expected in unsteady sims is ~1.2 so 1 / (20 * 1))
#                      turb = dict(cT = [1.0, 3.0], ))  # below and above the Betz limit


# ju.write_padeops_suite(single_inputs, varied_inputs, nested = True, default_input = default_inputs,
#     sim_template = sim_template, run_template = run_template, turb_template = turb_template)

cT = [1.0, 3.0]
dt = [0.1, 0.01, 0.0001]
nx = [160, 256, 384]
ny = [80, 128, 192]
nz = [80, 128, 192]
sf = [0, 1, 1]
sa = [0, 0.5, 0]
pa = [0, 0, 5.0]

varied_inputs = itertools.product(cT, itertools.zip_longest(dt, nx, ny, nz), itertools.zip_longest(sf, sa, pa))  # TODO: the dt are different based on if it is moving or not
# TODO: make sure filter is correct

varied_header = ["cT", "dt", "nx", "ny", "nz", "surge_freq", "surge_amplitude", "pitch_amplitude"]
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, nested = True, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template)