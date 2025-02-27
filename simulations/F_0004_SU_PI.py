import itertools
import jinja_sim_utils as ju
from pathlib import Path

# ...

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name

sf = 1.0
single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        # if not provided, default_inputs will be used
        Lx = 25,
        Ly = 10,
        Lz = 10,
        t_dataDump = 50,
        tstop = 150,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        surge_freq = sf,
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
varied_header = ["cT", "nx", "ny", "nz", "dt", "filterWidth", "useCorrection", "surge_amplitude", "pitch_amplitude"]


# surge_amplitude = 0.0, pitch_amplitude = 0.0,

# simulation setup parameters
cT = [1.0, 4.0]
nx = [64, 128, 256, 512]
ny = [32, 64, 128, 256]
nz = ny
CFL = 1.0
sf = 1.0
dt = [ju.find_min_dt(CFL, nx[i], ny[i], nz[i], sf, single_inputs) for i in range(len(nx))]
surge_amplitude = [0.5, 0.0]
pitch_amplitude = [0.0, 5.0]
filterWidth = [0.08] * len(dt)
useCorrection = [True] * len(dt)


# constant filter width (0.08) + corrected for CT' = 1 and 4 for both maximum surge and pitch
const_filter_varied_ct_all_setups = itertools.product(cT, itertools.zip_longest(nx, ny, nz, dt, filterWidth, useCorrection), itertools.zip_longest(surge_amplitude, pitch_amplitude))
# explore pitching (128, 64, 64) for CT = 1.0
surge_amplitude = [0.0]
pitch_amplitude = [5.0]
factor_list = [0.3, 1.5, 2.75]
nf = len(factor_list)
filterWidth = [ju.find_filter_width(single_inputs, nx = nx[1], ny = ny[1], nz = nz[1], factor = f) for f in factor_list]
varied_filter_corrections_med_coarse_exploration = itertools.product([1.0], itertools.zip_longest([nx[1]] * nf, [ny[1]] * nf, [nz[1]] * nf, [dt[1]] * nf, filterWidth, [True] * nf), itertools.zip_longest(surge_amplitude, pitch_amplitude))
# explore pitching (256, 128, 128) for CT = 1.0
varied_filter_no_corrections_med_coarse_exploration = itertools.product([1.0], itertools.zip_longest([nx[1]] * nf, [ny[1]] * nf, [nz[1]] * nf, [dt[1]] * nf, filterWidth, [False] * nf), itertools.zip_longest(surge_amplitude, pitch_amplitude))
set_filter_correction_both_true_false = itertools.product([1.0], itertools.zip_longest([nx[1]] * 2, [ny[1]] * 2, [nz[1]] * 2, [dt[1]] * 2, [0.08] * 2, [True, False]), itertools.zip_longest(surge_amplitude, pitch_amplitude))

varied_inputs = itertools.chain.from_iterable([const_filter_varied_ct_all_setups,
                                             varied_filter_corrections_med_coarse_exploration,
                                             varied_filter_no_corrections_med_coarse_exploration,
                                             set_filter_correction_both_true_false])
# for v in varied_inputs:
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)