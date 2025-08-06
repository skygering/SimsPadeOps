import itertools
import jinja_sim_utils as ju
from pathlib import Path

# From simulations 12 and 12_Cont, I was needing to check the stability of various runs by switching up the filter factor. Since the other runs weren't really set up to run with
# an additional filter factor, I decided to just manually write those out here.
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
        zLoc = 5.0
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "ct_effects",
        # if not provided, default_inputs will be used
        n_hrs = 4,
        queue = "spr",
    )
)
mid_resolution = [nx, ny, nz]
high_resolution = [2 * nx, 2 * ny, 2 * nz]

low_filter_width = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 1.5)]
mid_filter_width = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 2.5)]
high_filter_width = [ju.find_filter_width(single_inputs, nx = nx, ny = ny, nz = nz, factor = 3.5)]

low_filter_width_high_resolution = [ju.find_filter_width(single_inputs, nx = 2 * nx, ny = 2 * ny, nz = 2 * nz, factor = 1.5)]
mid_filter_width_high_resolution  = [ju.find_filter_width(single_inputs, nx = 2 * nx, ny = 2 * ny, nz = 2 * nz, factor = 2.5)]
high_filter_width_high_resolution  = [ju.find_filter_width(single_inputs, nx = 2 * nx, ny = 2 * ny, nz = 2 * nz, factor = 3.5)]
really_filter_width_high_resolution  = [ju.find_filter_width(single_inputs, nx = 2 * nx, ny = 2 * ny, nz = 2 * nz, factor = 5)]


low_freq_dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.0, single_inputs, v = 0.0, w = 0.0)]
mid_freq_dt = [ju.find_min_dt(1.0, nx, ny, nz, 1.5, single_inputs, v = 0.0, w = 0.0)]

low_freq_dt_high_resolution = [ju.find_min_dt(1.0, 2 * nx, 2 * ny, 2 * nz, 1.0, single_inputs, v = 0.0, w = 0.0)]
mid_freq_dt_high_resolution = [ju.find_min_dt(1.0, 2 * nx, 2 * ny, 2 * nz, 1.5, single_inputs, v = 0.0, w = 0.0)]

# cT,surge_freq,surge_amplitude,pitch_amplitude,dt,filterWidth
rerun_1 = [1.0,1.5,1.0,0.0] + mid_freq_dt + mid_filter_width + mid_resolution # run 15 from 12 cont
rerun_2 = [4.0,0.5,0,10.0] + low_freq_dt + mid_filter_width + mid_resolution # run 50 from 12
rerun_3 = [4.0,1.0,0,10.0] + low_freq_dt + mid_filter_width + mid_resolution # run 51 from 12
rerun_4 = [4.0,0.5,0,3.0] + low_freq_dt + mid_filter_width + mid_resolution # run 42 from 12
rerun_5 = [4.0,1.0,0,3.0] + low_freq_dt + mid_filter_width + mid_resolution # run 45 from 12
rerun_6 = [4.0,0.1,0.5,0] + low_freq_dt + mid_filter_width + mid_resolution # run 32 from 12
rerun_7 = [4.0,0.5,0.5,0] + low_freq_dt + mid_filter_width + mid_resolution # run 35 from 12
rerun_8 = [4.0,1.0,0.5,0] + low_freq_dt + mid_filter_width + mid_resolution # run ?? from ??  -> probably run during stabliltiy testing

# new check for stability of surging at CT' = 4
run_9 = [4.0,1.0,1.0,0] + low_freq_dt + low_filter_width + mid_resolution # checking for highest amplitude stability
run_10 = [4.0,1.0,1.0,0] + low_freq_dt + mid_filter_width + mid_resolution # checking for highest amplitude stability
run_11 = [4.0,1.5,1.0,0] + mid_freq_dt + low_filter_width + mid_resolution # checking for highest amplitude/frequency stability
run_12 = [4.0,1.5,1.0,0] + mid_freq_dt + mid_filter_width + mid_resolution # checking for highest amplitude/frequency stability

# run with even higher filter width
rerun_13 = [1.0,1.5,1.0,0.0] + mid_freq_dt + high_filter_width + mid_resolution # run 15 from 12 cont
rerun_14 = [4.0,0.5,0,10.0] + low_freq_dt + high_filter_width + mid_resolution # run 50 from 12
rerun_15 = [4.0,1.0,0,10.0] + low_freq_dt + high_filter_width + mid_resolution # run 51 from 12
rerun_16 = [4.0,0.5,0,3.0] + low_freq_dt + high_filter_width + mid_resolution # run 42 from 12
rerun_17 = [4.0,1.0,0,3.0] + low_freq_dt + high_filter_width + mid_resolution # run 45 from 12
rerun_18 = [4.0,0.1,0.5,0] + low_freq_dt + high_filter_width + mid_resolution # run 32 from 12
rerun_19 = [4.0,0.5,0.5,0] + low_freq_dt + high_filter_width + mid_resolution # run 35 from 12
rerun_20 = [4.0,1.0,0.5,0] + low_freq_dt + high_filter_width + mid_resolution # run ?? from ??  -> probably run during stabliltiy testing

# new check for stability of surging at CT' = 4
run_21 = [4.0,1.0,1.0,0] + low_freq_dt + high_filter_width + mid_resolution # checking for highest amplitude stability
run_22 = [4.0,1.5,1.0,0] + mid_freq_dt + high_filter_width + mid_resolution # checking for highest amplitude/frequency stability

# check convergence using a higher resolution
rerun_23 = [4.0,0.5,0,10.0] + low_freq_dt_high_resolution + mid_filter_width_high_resolution + high_resolution # run 50 from 12
rerun_24 = [4.0,1.0,0,10.0] + low_freq_dt_high_resolution + mid_filter_width_high_resolution + high_resolution # run 51 from 12
rerun_25 = [4.0,0.5,0,3.0] + low_freq_dt_high_resolution + mid_filter_width_high_resolution + high_resolution # run 42 from 12
rerun_26 = [4.0,1.0,0,3.0] + low_freq_dt_high_resolution + mid_filter_width_high_resolution + high_resolution # run 45 from 12
rerun_27 = [4.0,0.1,0.5,0] + low_freq_dt_high_resolution + mid_filter_width_high_resolution + high_resolution # run 32 from 12
rerun_28 = [4.0,0.5,0.5,0] + low_freq_dt_high_resolution + mid_filter_width_high_resolution + high_resolution # run 35 from 12
rerun_29 = [4.0,1.0,0.5,0] + low_freq_dt_high_resolution + mid_filter_width_high_resolution + high_resolution # run ?? from ??  -> probably run during stabliltiy testing

rerun_30 = [4.0,0.5,0,10.0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run 50 from 12
rerun_31 = [4.0,1.0,0,10.0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run 51 from 12
rerun_32 = [4.0,0.5,0,3.0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run 42 from 12
rerun_33 = [4.0,1.0,0,3.0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run 45 from 12
rerun_34 = [4.0,0.1,0.5,0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run 32 from 12
rerun_35 = [4.0,0.5,0.5,0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run 35 from 12
rerun_36 = [4.0,1.0,0.5,0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run ?? from ??  -> probably run during stabliltiy testing

rerun_37 = [4.0,0.1,0.5,0] + low_freq_dt_high_resolution + low_filter_width_high_resolution + high_resolution # run 32 from 12
rerun_38 = [4.0,0.5,0.5,0] + low_freq_dt_high_resolution + low_filter_width_high_resolution + high_resolution # run 35 from 12
rerun_39 = [4.0,1.0,0.5,0] + low_freq_dt_high_resolution + low_filter_width_high_resolution + high_resolution # run ?? from ??  -> probably run during stabliltiy testing

rerun_40 = [4.0,0.1,0.5,0] + [dt / 2 for dt in low_freq_dt_high_resolution] + low_filter_width_high_resolution + high_resolution # run 32 from 12
rerun_41 = [4.0,0.5,0.5,0] + [dt / 2 for dt in low_freq_dt_high_resolution] + low_filter_width_high_resolution + high_resolution # run 35 from 12


rerun_42 = [4.0,0.1,0.5,0] + [dt / 4 for dt in low_freq_dt_high_resolution] + low_filter_width_high_resolution + high_resolution # run 32 from 12
rerun_43 = [4.0,0.1,0.5,0] + [dt / 4 for dt in low_freq_dt_high_resolution] + mid_filter_width_high_resolution + high_resolution # run 32 from 12
rerun_44 = [4.0,0.1,0.5,0] + [dt / 4 for dt in low_freq_dt_high_resolution] + high_filter_width_high_resolution + high_resolution # run 32 from 12

rerun_45 = [4.0,1.0,0,10.0] + low_freq_dt_high_resolution + low_filter_width_high_resolution + high_resolution # run 51 from 12

rerun_46 = [4.0, 0.0, 0.0, 0] + low_freq_dt_high_resolution + low_filter_width_high_resolution + high_resolution
rerun_47 = [4.0, 0.0, 0.0, 0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution

rerun_48 = [4.0,0.1,0.5,0] + low_freq_dt_high_resolution + really_filter_width_high_resolution + high_resolution # run ?? from ??  -> probably run during stabliltiy testing
rerun_49 = [2.0,0.1,0.5,0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run ?? from ??  -> probably run during stabliltiy testing
rerun_50 = [1.0,0.1,0.5,0] + low_freq_dt_high_resolution + high_filter_width_high_resolution + high_resolution # run ?? from ??  -> probably run during stabliltiy testing

varied_inputs = [rerun_1, rerun_2, rerun_3, rerun_4, rerun_5, rerun_6, rerun_7, rerun_8, run_9, run_10, run_11, run_12,
                rerun_13, rerun_14, rerun_15, rerun_16, rerun_17, rerun_18, rerun_19, rerun_20, run_21, run_22,
                rerun_23, rerun_24, rerun_25, rerun_26, rerun_27, rerun_28, rerun_29,
                rerun_30, rerun_31, rerun_32, rerun_33, rerun_34, rerun_35, rerun_36,
                rerun_37, rerun_38, rerun_39, rerun_40, rerun_41, rerun_42, rerun_43, rerun_44, rerun_45, rerun_46, rerun_47, rerun_48,
                rerun_49, rerun_50]
varied_header = ["cT", "surge_freq", "surge_amplitude", "pitch_amplitude", "dt", "filterWidth", "nx", "ny", "nz"]

# for v in varied_inputs: 
#     print(v)                                   

# write needed simulation files
ju.write_padeops_suite(single_inputs, varied_inputs, varied_header = varied_header, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template, node_cap = 12)

