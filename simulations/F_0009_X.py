import itertools
import jinja_sim_utils as ju
from pathlib import Path

# In this simulation, I will re-run the simulations in F_0004_X except using a filter
# width of h/D = h. Using a filter width of 0.08 was numerically unstable for some of
# the pitching cases, so I want higher filter values to compare to.

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
hit_template = ju.TEMPLATE_PATH.joinpath("hit_template.jinja")
interaction_template = ju.TEMPLATE_PATH.joinpath("interaction_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("hit_defaults.json")

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
        useCorrection = True,
    ),
        hit = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = "/work2/08445/tg877441/shared_tmp/cube128_10D"
    ),
    interactions = dict(
        TI_target = 0.03,
        TI_xloc = 5.0,
        TI_fact = -1.0

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
cT_prime = [1.0, 4.0]

nx = [256, 512]
ny = [128, 256]
nz = ny
filterWidth = [ju.find_filter_width(single_inputs, nx = nx[i], ny = ny[i], nz = nz[i], factor = 1.5) for i in range(len(nx))]
domain_size_iter = itertools.zip_longest(nx, ny, nz, filterWidth)

sf = [0.0, 1.0, 1.0]
sa = [0.0, 0.5, 0.0]
pa = [0.0, 0.0, 5.0]
turbine_movement_iter = itertools.zip_longest(sf, sa, pa)

varied_inputs = itertools.product(domain_size_iter, cT_prime, turbine_movement_iter)
varied_inputs = [(ju.find_min_dt(1.0, vin[0][0], vin[0][1], vin[0][2], vin[2][0], single_inputs, v = 0.0, w = 0.0), ) + vin for vin in varied_inputs]
varied_header = ["dt", "nx", "ny", "nz", "filterWidth", "cT", "surge_freq", "surge_amplitude", "pitch_amplitude"]

# for v in varied_inputs:
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)