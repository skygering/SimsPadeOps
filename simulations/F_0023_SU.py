import itertools
import jinja_sim_utils as ju
from pathlib import Path

# From simulations 16, I ran with all of the even frequencies and amplitudes. I wanted to run with odd amplitudes and frequencies
# to get more information and try to fill in a few of the blank spots where the even cases nan-ed out.

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name
nx, ny, nz = 256, 256, 256
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
        job_name = "odd_surging",
        # if not provided, default_inputs will be used
        n_hrs = 3,
        queue = "spr",
    )
)

# original runs that I planned with "reasonable" values -->
# some of the higher amplitudes and frequencies struggle to run
# so I need to reduce the timesteps below
cT = [1.33, 1.66, 2.0]

# I now want to add in "odd" frequencies and amplitudes - for largest amplitudes, I will cut the dt in half like above
sf_odd = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
sa_odd = [0.1, 0.3, 0.5, 0.7]
sa_odd_max = [0.9, 1.1]

dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)]
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 2.5)]

amp_odd_surging_iter = itertools.product(sf_odd, sa_odd, [0.0])
varied_inputs_amp_odd_surging_iter = itertools.product(cT, amp_odd_surging_iter, dt, filterWidth)
amp_odd_surging_large_iter = itertools.product(sf_odd, sa_odd_max, [0.0])
varied_inputs_amp_odd_surging_large_amp_iter = itertools.product(cT, amp_odd_surging_large_iter, [t / 2 for t in dt], filterWidth)

varied_header = ["cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "filterWidth"]
varied_inputs = itertools.chain.from_iterable([varied_inputs_amp_odd_surging_iter, varied_inputs_amp_odd_surging_large_amp_iter])

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

ju.make_batched_sbatch_files(
    ju.DATA_PATH + curr_script_name + "_Files",
    max_per_batch=12,
    output_glob="*.out",
    avg_hours = 1.5,
    timeout_hours = 6,
    sbatch_prefix="re_run_odd_surge_batch",
    max_walltime_hours=12,
)