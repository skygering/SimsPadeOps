import numpy as np
import itertools
import jinja_sim_utils as ju
from pathlib import Path

# Here I am running joint pitching and surging cases. Since "real-world" pitching is actually made
# up of both the surging and pitching motions in PadeOps. I have calculated what the minimum surging
# possible is for each pitching case in the "real-world" and then run only cases with surging greater
# than that, for cases that have been run with exclusively pitching and surging previousl (see F_0016).

# I started by running the pitch values of 8, 12, and 16 degrees with frequency of 0.4, 0.6, and 0.8.
# The surge velocities, where applicable are 0.4, 0.6, and 0.8. Later, higher and lower values may be run. 
# However, for now, I wanted to get a handle on how these simulations look. I also started with just CT' = 1.33
# and CT' = 2.0, as the cases don't seem to vary as much with CT' as expected in F_0016.

R = 150 / 240  # IEA 15MW has 150m hub height and a 240m diameter

def get_min_surge(theta0, f):
    # calc needed values
    omega = 2 * np.pi * f
    theta0r = np.deg2rad(theta0)
    # angle -> theta = theta0 * sin(omega * t)
    theta = theta0r * np.sin(omega * np.linspace(0, 4 / f, num = 100))
    # angular velocity -> theta_dot = theta0 * omega * cos(omega * t)
    theta_dot = theta0r * omega * np.cos(omega * f)
    # linear x-velocity -> v_x = theta_dot * z
    z = R * np.cos(theta)
    vx = np.max(np.abs(theta_dot * z))
    return vx

def get_all_possible_surge(theta0, f, min_surge, all_surge):
    extra_surge = all_surge[all_surge > min_surge]
    surge_vals = np.append(extra_surge, min_surge)
    surge_vals = np.sort(surge_vals)
    return surge_vals

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
        job_name = "ct_effects",
        # if not provided, default_inputs will be used
        n_hrs = 3,
        queue = "skx"
    )
)

cT = [1.33, 2.0]

pitch_amps = [8, 12, 16]
freqs = np.array([0.4, 0.6, 0.8])
surge_amps = np.array([0.2, 0.4, 0.6])

surge_min_cases = []
for pa in pitch_amps:
    for sf in freqs:
        # get x-velocity
        vx = get_min_surge(pa, sf)
        surge_min_cases.append((pa, sf, vx))

sf_vals = []
sa_vals = []
pa_vals = []
for (pa, sf, sa_min) in surge_min_cases:
    sa_all = get_all_possible_surge(pa, sf, sa_min, surge_amps)
    for sa in sa_all:
        sf_vals.append(sf)
        sa_vals.append(sa)
        pa_vals.append(pa)

movement_iter = itertools.zip_longest(sf_vals, sa_vals, pa_vals)
dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)] # have timestep based on sf = 1.0
filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 2.5)]
inital_exploration_iter = itertools.product(cT, movement_iter, dt, filterWidth)

varied_header = ["cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "filterWidth"]
varied_inputs = itertools.chain.from_iterable([inital_exploration_iter])

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

ju.make_batched_sbatch_files(
    ju.DATA_PATH + curr_script_name + "_Files",
    max_per_batch=12,
    output_glob="*.out",
    avg_hours = 3,
    timeout_hours = 6,
    sbatch_prefix="re_run_pitch_and_surge_batches",
    max_walltime_hours=12,
)