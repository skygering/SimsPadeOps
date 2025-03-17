import itertools
import jinja_sim_utils as ju
from pathlib import Path

# In this simulation, I will re-run the simulations in F_0004_X except using a filter
# width of h/D = h. Using a filter width of 0.08 was numerically unstable for some of
# the pitching cases, so I want higher filter values to compare to.

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
        t_dataDump = 50,
        tstop = 250,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = False,
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

# simulation setup parameters

a = 0.5  # estimate of induction
cT = [1.0, 1.5]
cT_prime = [1.0] + [c / (1 - a)**2 for c in cT]

nx = [128, 256, 512]
ny = [64, 128, 256]
nz = ny
factor = 2.5
filterWidth = [ju.find_filter_width(single_inputs, nx = nx[i], ny = ny[i], nz = nz[i], factor = factor) for i in range(len(nx))]
domain_size_iter = itertools.zip_longest(nx, ny, nz, filterWidth)

sf = [0.0, 1.0, 1.0]
sa = [0.0, 0.5, 0.0]
pa = [0.0, 0.0, 5.0]
turbine_movement_iter = itertools.zip_longest(sf, sa, pa)

varied_inputs = itertools.chain.from_iterable([itertools.product(cT_prime, turbine_movement_iter, domain_size_iter),
                                              itertools.product(
                                                  [6.0], [(0.0, 0.0, 0.0)],
                                                  [(nx[1], ny[1], nz[1], ju.find_filter_width(single_inputs, nx = nx[1], ny = ny[1], nz = nz[1], factor = 2.0))])])

varied_inputs = [(ju.find_min_dt(1.0, vin[2][0], vin[2][1], vin[2][2], vin[1][0], single_inputs, v = 0.0, w = 0.0), ) + vin for vin in varied_inputs]
varied_header = ["dt", "cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "nx", "ny", "nz", "filterWidth"]

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)