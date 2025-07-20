import itertools
import jinja_sim_utils as ju
from pathlib import Path

# From simulations 10, I decided to stick with filter factor 1.5 with correctionsOn
# I now need to run simulations that explore the effects frequency and amplitude

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name
nx, ny, nz = 256, 128, 128
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
        tstop = 250,
        CFL = -1,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        useCorrection = True,
        cT = 1.0,
        zLoc = 5.0
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "ct_effects",
        # if not provided, default_inputs will be used
        n_hrs = 1.25,
    )
)

cT = [1.0, 4.0]

sf = [0.1, 0.5, 1.0]
sa = [0.1, 0.3, 0.5]
pa = [1.0, 3.0, 5.0]
nsf = len(sf)
nsa = len(sa)
npa = len(pa)

high_sf = [2.0]
high_sa = [1.0]
high_pa = [10.0]

# intial range
low_freq_dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)]
surge_movement_iter = itertools.filterfalse( lambda x: x[0] == 1.0 and x[1] == 0.5, itertools.product(sf, sa, [0], low_freq_dt))
pitch_movement_iter = itertools.filterfalse( lambda x: x[0] == 1.0 and x[2] == 5.0, itertools.product(sf, [0], pa, low_freq_dt))
# low_freq_dt = ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)
# dt = [low_freq_dt] * (nsf * (nsa + npa) - 2) # previously ran f = 1.0, A = 0.5/5.0
# higher amplitude previously run
high_amp_surge_movement_iter = itertools.product(sf, high_sa, [0], low_freq_dt)
high_amp_pitch_movement_iter = itertools.product(sf, [0], high_pa, low_freq_dt)
# dt = dt + [low_freq_dt] * (nsf + nsf)
# higher freqency previously run
high_freq_dt = [ju.find_min_dt(1.0, nx, ny, nz, 2.0, single_inputs, v = 0.0, w = 0.0)]
high_freq_surge_movement_iter = itertools.product(high_sf, sa + high_sa, [0], high_freq_dt)
high_freq_pitch_movement_iter = itertools.product(high_sf, [0], pa + high_pa, high_freq_dt)
# high_freq_dt = ju.find_min_dt(1.0, nx, ny, nz, 2.0, single_inputs, v = 0.0, w = 0.0)
# dt = dt + [high_freq_dt] * ((nsa + 1) + (npa + 1))

movement_iter = itertools.chain.from_iterable([surge_movement_iter, pitch_movement_iter,
                                               high_amp_surge_movement_iter, high_amp_pitch_movement_iter,
                                               high_freq_surge_movement_iter, high_freq_pitch_movement_iter])


filterWidth = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 1.5)]
varied_inputs = itertools.product(cT, movement_iter, filterWidth)
varied_header = ["cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "filterWidth"]

for v in varied_inputs: 
    print(v)                                   

# write needed simulation files
# ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
#     sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)

