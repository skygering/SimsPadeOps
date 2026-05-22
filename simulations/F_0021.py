import itertools
import jinja_sim_utils as ju
from pathlib import Path

# For this simulation, I am trying to run simulations with NO turbines and a surging inflow. This
# will allow me to test my triple decomposition code without having to deal with the complexities
# of a turbine. I will run pretty coarse simulations. 

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name
nx, ny, nz = 128, 128, 128
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
        tstop = 300,
        CFL = 0.8,
        useWindTurbines = False,
        do_time_budgets = True,
        time_budgetType = 1,
        do_multi_phase_budgets = True,
        phases = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        phase_tol = 0.05
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_floating",
        job_name = "surge_inflow_test",
        # if not provided, default_inputs will be used
        n_hrs = 3,
        queue = "skx",
        build_folder = "build_opti_surge_inflow"
    )
)

# original runs that I planned with "reasonable" values -->
# some of the higher amplitudes and frequencies struggle to run
# so I need to reduce the timesteps below
freqs = [0.1]
amps = [10]

inflow_cases = itertools.product(freqs, amps)
varied_header = ["inflow_amplit", "inflow_freq"]
varied_inputs = inflow_cases

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

# ju.make_batched_sbatch_files(
#     ju.DATA_PATH + curr_script_name + "_Files",
#     max_per_batch=12,
#     output_glob="*.out",
#     avg_hours = 4,
#     timeout_hours = 6,
#     sbatch_prefix="re_run_high_batch",
#     max_walltime_hours=12,
# )