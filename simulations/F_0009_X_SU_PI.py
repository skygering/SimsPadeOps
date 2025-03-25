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

Lx = 25
Ly = 10
single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ju.DATA_PATH + curr_script_name + "_Files",
        outputdir = ju.DATA_PATH + curr_script_name + "_Files",
        # if not provided, default_inputs will be used
        Lx = Lx,
        Ly = Ly,
        Lz = Ly,
        t_dataDump = 50,
        tstop = 400,
        CFL = -1,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        xloc = 5.0,
    ),
    hit = dict(
        # always need to provide the filepaths (no defaults)
        restartFile_TID = 0,
        restartFile_RID = 1
    ),
    interaction = dict(
        TI_target = -1.0,
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

# Set up simulations from grid convergence to re-run

# two resolutions
ny = [128, 256]  # HIT box must be a perfect square (Lx = Ly = Lz = 10 AND nx = ny = nz = 128 OR = 256)
nz = ny
nx = [(n / Ly) * Lx for n in ny]  # dx of HIT must be an integer mutliple of dx of AD simulation
ndims = len(ny)
hit_dirs = ["/work2/08445/tg877441/shared_tmp/cube128_10D", "/work2/08445/tg877441/shared_tmp/cube256_10D"]
TI_fact = [0.3477662250832144, 0.31252588355210836]  # for two resolutions, determined from H_0009
factors = [1.0, 1.5]
n_factors = len(factors)

filterWidth = [0] * (n_factors * ndims)
useCorrection = [False] * (n_factors * ndims)
c = 0
for f in factors:
    for i in range(ndims):
        filterWidth[c] = ju.find_filter_width(single_inputs, nx = nx[i], ny = ny[i], nz = nz[i], factor = f)
        if f == 1.5:
            useCorrection[c] = True
        c += 1

domain_size_iter = itertools.zip_longest(nx * n_factors, ny * n_factors, nz * n_factors, filterWidth, useCorrection, hit_dirs * n_factors, TI_fact * n_factors)

# two values of CT'
cT = [4.0]

# three movement
sf = [0.0, 1.0, 1.0]
sa = [0.0, 0.5, 0.0]
pa = [0.0, 0.0, 5.0]
turbine_movement_iter = itertools.zip_longest(sf, sa, pa)

# putting it all together! 
varied_inputs = itertools.product(cT, domain_size_iter, turbine_movement_iter)
varied_inputs = [(ju.find_min_dt(1.0, vin[1][0], vin[1][1], vin[1][2], sf = vin[2][0], single_inputs = single_inputs, v = 0.0, w = 0.0), ) + vin for vin in varied_inputs]
varied_header = ["dt", "cT", "nx", "ny", "nz", "filterWidth", "useCorrection", "hit_inputdir", "TI_fact", "surge_freq", "surge_amplitude", "pitch_amplitude"]

# for v in varied_inputs:
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template,
    hit_template = hit_template, interaction_template = interaction_template, node_cap = 12)