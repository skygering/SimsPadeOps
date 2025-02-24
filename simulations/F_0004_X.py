import itertools
import jinja_sim_utils as ju
from pathlib import Path

# In this simulation, I will take what I learned from F_0002_X and F_0003_X
# to re-run the grid convergence study for static turbines. I will use a tiny filter
# and no correction, and use Ly = Lz = 10D, as I now have a better idea of how this effects
# power production.
# I will now vary the resolution, doubleing the number of grid lines in all directions at
# each timestep. I will run 4 total resolutions, varying from very coarse to fine.
# I will then repeate this for 3 CT' values (1, 4, and 8). If everything looks good,
# then I will create F_0004_SU and F_0004_PI to see how this might differ for surge and pitch.

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
        tstop = 150,
        t_dataDump = 50,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        surge_freq = 0.0,
        surge_amplitude = 0.0,
        pitch_amplitude = 0.0,
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
varied_header = ["cT", "nx", "ny", "nz", "dt", "filterWidth", "useCorrection"]
tstop_CT1 = 150
t_stop_CT4 = 350

# simulation setup parameters
cT = [1.0]
nx = [64, 128, 256, 512]
ny = [32, 64, 128, 256]
nz = ny
CFL = 1.0
sf = 0.0
dt = [ju.find_min_dt(CFL, nx[i], ny[i], nz[i], sf, single_inputs) for i in range(len(nx))]

# filter width changes with grid size + corrected
filterWidth = [ju.find_filter_width(single_inputs, nx = nx[i], ny = ny[i], nz = nz[i], factor = 1.0) for i in range(len(nx))]
grid_convergence_change_filter = itertools.product(cT, itertools.zip_longest(nx, ny, nz, dt, filterWidth, [True] * len(nx)))
# constant filter width (0.08) + corrected
grid_convergence_const_filter = itertools.product(cT, itertools.zip_longest(nx, ny, nz, dt, [0.08] * len(nx), [True] * len(nx)))
# constant filter width (0.08) + NOT corrected
grid_convergence_const_filter_no_correction = itertools.product(cT, itertools.zip_longest(nx, ny, nz, dt, [0.08] * len(nx), [False] * len(nx)))
# examine (128, 64, 64) again since it is giving weird results - vary the factors for the filter width
factor_list = [0.25, 1.0, 1.5, 2.75, 4.0]
nf = len(factor_list)
filterWidth = [ju.find_filter_width(single_inputs, nx = nx[1], ny = ny[1], nz = nz[1], factor = f) for f in factor_list]
grid_convergence_med_corse_exploration = itertools.product(cT, itertools.zip_longest([nx[1]] * nf, [ny[1]] * nf, [nz[1]] * nf, [dt[1]] * nf, filterWidth, [True] * nf))
# run the 0.08 filter width with corrections for CT' = 4.0 and 8.0
grid_convergence_const_filter_varying_CT = itertools.product([4.0, 8.0], itertools.zip_longest(nx, ny, nz, dt, [0.08] * len(nx), [True] * len(nx)))

# combine all of the above simulations
varied_inputs = itertools.chain.from_iterable([grid_convergence_change_filter, grid_convergence_const_filter,
                                               grid_convergence_const_filter_no_correction, grid_convergence_med_corse_exploration,
                                               grid_convergence_const_filter_varying_CT])

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)