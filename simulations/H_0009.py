import itertools
import jinja_sim_utils as ju
from pathlib import Path

# In this simulation, I will figure out the turbine factor to be able to re-run many of my
# experiments from the grid convergence tests, but with added turbulence.
# I need is to determine the TI_fact. Thus, I ran with a TI_target of 0.035 to
# determine the TI_fact. I did this for both resolutions of HIT grid.

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
        CFL = -1
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = True,
        xloc = 5.0,
    ),
    hit = dict(
        # always need to provide the filepaths (no defaults)
        restartFile_TID = 0,
        restartFile_RID = 1
    ),
    interaction = dict(
        TI_target = 0.035,
        TI_xloc = 5.0,
        TI_fact = -1.0
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "HIT_shear",
        job_name = "TI_test_sg",
        # if not provided, default_inputs will be used
        n_hrs = 4,
    )
)

# Set up two simulations to find TI_fact ---------------------------------------------------------------

# vary resolutions
ny = [128, 256]  # HIT box must be a perfect square (Lx = Ly = Lz = 10 AND nx = ny = nz = 128 OR = 256)
nz = ny
nx = [(n / Ly) * Lx for n in ny]  # dx of HIT must be an integer mutliple of dx of AD simulation
# vary HIT information (resolution of HIT box must match up with ny resolution)
hit_dirs = ["/work2/08445/tg877441/shared_tmp/cube128_10D", "/work2/08445/tg877441/shared_tmp/cube256_10D"]
domain_size_iter = itertools.zip_longest(nx, ny, nz, hit_dirs)

# Turbulence information (in interaction file)
TI_target = [0.035] # sims to find out targets + simulations using targets
TI_fact = [-1] # disabled for sims to determine tagets
ti_iter = itertools.zip_longest(TI_target, TI_fact)

useWindTurbines = [False]
useDynamicTurbine = [False]
turbine_iter = itertools.zip_longest(useWindTurbines, useDynamicTurbine)

varied_inputs = itertools.product(domain_size_iter, ti_iter, turbine_iter)
varied_inputs = [(ju.find_min_dt(1.0, vin[0][0], vin[0][1], vin[0][2], sf = 0, single_inputs = single_inputs, v = 0.0, w = 0.0), ) + vin for vin in varied_inputs]
varied_header = ["dt", "nx", "ny", "nz", "hit_inputdir", "TI_target", "TI_fact", "useWindTurbines", "useDynamicTurbine"]

TI_fact_128 = 0.3477662250832144
TI_fact_256 = 0.31252588355210836

# for v in varied_inputs:
#     print(v)                                   

# write needed simulation files
# ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
#     sim_template = sim_template, run_template = run_template, turb_template = turb_template,
#     hit_template = hit_template, interaction_template = interaction_template, node_cap = 12)