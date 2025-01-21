import jinja_sim_utils as ju
from pathlib import Path

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# file name based on current script name
curr_script_name = Path(__file__).with_suffix('').name
scratch_path = "./scratch/10264/sgering/"

single_inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = scratch_path + curr_script_name,
        outputdir = scratch_path + curr_script_name,
        # if not provided, default_inputs will be used
        tstop = 250,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "turbines",
        problem_name = "AD_coriolis_shear",
        job_name = "steady_turbine_test_sg",
        # if not provided, default_inputs will be used
        n_hrs = 4,
    )
)

varied_inputs = dict(sim = dict(dt = [0.5, 1/(20 * 1.2)]),  # big and small timesteps (max f expected in unsteady sims is ~1.2 so 1 / (20 * 1))
                     turb = dict(cT = [1.0, 3.0]))  # below and above the Betz limit

ju.write_pardeops_suite(single_inputs, varied_inputs, default_input = default_inputs,
    sim_template = sim_template, run_template = run_template, turb_template = turb_template)