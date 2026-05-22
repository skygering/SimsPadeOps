import itertools
import jinja_sim_utils as ju
from pathlib import Path

# From simulations 16, I have most of the needed simulations to undertand the problem space. However, I do want to run some extra sensitivity testing. 
# I will try some smaller timesteps, larger filter widths, and some same "period-step" simulations rather than same timestep simulations.

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
        job_name = "sensitivity_tests",
        # if not provided, default_inputs will be used
        queue = "spr",
    )
)

cT = [2.0]
As = [0.6, 0.8, 1.0]
Fs = [0.6, 1.0]

# change the timesteps
dts = [0.005, 0.01, 1/60]
n_long = 10
n_med = 5
n_hrs = [n_long, n_med, n_med]
dt_hrs = itertools.zip_longest(dts, n_hrs)
surging_cases = itertools.product(Fs, As, [0.0])
filterWidth_25 = ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 2.5)
changing_dt_cases = itertools.product(cT, surging_cases, dt_hrs, [filterWidth_25], [ny], [nz])

# change the filter width
surging_cases = itertools.product(Fs, As, [0.0])
filterWidths = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = fac) for fac in [3.5, 4.5]]
changing_filterWidth_cases = itertools.product(cT, surging_cases, [0.01], [n_med], filterWidths, [ny], [nz])

# test really fine highest amplitude to see if it will run
filterWidth_35 = ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 3.5)
surging_cases = itertools.zip_longest([0.2, 0.6, 0.6], [1.0, 1.0, 1.2], [0.0 for _ in range(3)])
high_amp_stability_cases = itertools.product(cT, surging_cases, [0.005], [n_long], [filterWidth_25, filterWidth_35], [ny], [nz])

# change the resolution to see what effects that has!
surging_cases = itertools.product(Fs, As, [0.0])
changing_resolution_cases = itertools.product(cT, surging_cases, [0.01], [n_med / 2], [filterWidth_25], [(ny / 2, nz / 2), (ny / 4, nz / 4)])

# test medium fine timestep with highest amplitude to see if it will run
surging_cases = itertools.zip_longest([0.2, 0.6, 0.6], [1.0, 1.0, 1.2], [0.0 for _ in range(3)])
high_amp_med_time_stability_cases = itertools.product(cT, surging_cases, [0.01], [n_med], [filterWidth_25], [ny], [nz])

# extra cases to bring greater variability in frequency and amplitude:
As_extra = [0.2]
fs_extra = [0.2]

extra_f_surge_cases = itertools.product(fs_extra, As, [0.0])
extra_A_surge_cases = itertools.product(fs_extra + Fs, As_extra, [0.0])
extra_surge_cases = itertools.chain(extra_f_surge_cases, extra_A_surge_cases)
extra_surge_cases = list(extra_surge_cases)

extra_changing_dt_cases = itertools.product(cT, extra_surge_cases, itertools.zip_longest([0.005, 0.01], [8, 5]), [filterWidth_25], [ny], [nz])

extra_changing_filterWidth_cases = itertools.product(cT, extra_surge_cases, [0.01], [5], filterWidths, [ny], [nz])

extra_changing_resolution_cases = itertools.product(cT, extra_surge_cases, [0.01], [3], [filterWidth_25], [(ny / 2, nz / 2), (ny / 4, nz / 4)])

surging_matrix = list(itertools.product([0.2, 0.6, 1.0], [0.2, 0.6, 1.0], [0.0]))
dt_002_surge_cases = itertools.product(cT, surging_matrix, [0.02], [3], [filterWidth_25], [ny], [nz])

pitching_matrix = itertools.product([0.2, 0.6, 1.0], [0.0], [4, 16])
pitch_dt_hrs = itertools.zip_longest([0.005, 0.01, 0.02], [8, 5, 3])
pitching_cases = itertools.product(cT, pitching_matrix, pitch_dt_hrs, [filterWidth_25], [ny], [nz])

smaller_filters_256 = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = fac) for fac in [1.5, 2.0]]
smaller_filter_surge_256 = itertools.product(cT, surging_matrix, [0.01], [5], smaller_filters_256, [ny], [nz])

# filters_128 = [ju.find_filter_width(single_inputs, nx = nx, ny = ny/2, nz = nz/2, factor = fac) for fac in [1.5, 2.5, 3.5]]
# smaller_filter_surge_128 = itertools.product(cT, surging_matrix, [0.01], [5], filters_128, [ny/2], [nz/2])

varied_header = ["cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "n_hrs", "filterWidth", "ny", "nz"]
varied_inputs = itertools.chain.from_iterable([changing_dt_cases,changing_filterWidth_cases,high_amp_stability_cases,changing_resolution_cases,
                                               high_amp_med_time_stability_cases,
                                               extra_changing_dt_cases, extra_changing_filterWidth_cases, extra_changing_resolution_cases,
                                               dt_002_surge_cases, pitching_cases,
                                               smaller_filter_surge_256])

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 8)

ju.make_batched_sbatch_files(
    ju.DATA_PATH + curr_script_name + "_Files",
    max_per_batch=12,
    output_glob="*.out",
    avg_hours = 6,
    timeout_hours = 12,
    sbatch_prefix="batch_surge_big_filter",
    max_walltime_hours=18,
    min_sim = 67,
    max_sim = 74
)