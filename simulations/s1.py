import jinja_sim_utils as ju
from pathlib import Path

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

print(default_inputs)

inputs = dict(
    sim = dict(
        # always need to provide the filepaths (no defaults)
        inputdir = ".",
        outputdir = ".",
        turb_dirname = "turbs",
        # if not provided, default_inputs will be used
        nx = 0,
        ny = 0,
        nz = 0,
        tstop = 0,
        CFL = -1,
        dt = 0.25,
        Lx = 0,
        Ly = 0,
        Lz = 0,
    ),
    turb = dict(  # can only provide one turbine right now - update when needed
        # if not provided, default_inputs will be used
        xLoc = 5,
        yLoc = 5,
        zLoc = 2.5,
        cT = 1.33,
        surge_freq = 0,
        surge_amplitude = 0,
        pitch_amplitude = 0,
    ),
    run = dict(
        # always need to provide the filepaths (no defaults)
        problem_dir = "",
        problem_name = "",
        job_name = "",
        # if not provided, default_inputs will be used
        n_hrs = 24,
        # calculated regardless of input (could change later...)
        n_nodes = 1,
    )
)

ju.write_padeops_files(inputs, default_input=default_inputs,
                       sim_template = sim_template, run_template = run_template, turb_template = turb_template)