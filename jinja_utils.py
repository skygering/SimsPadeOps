"""
Helper functions for writing input files

Skylar Gering
2025 Janudary 16
"""

import numpy as np
import jinja2
import re
from pathlib import Path
import warnings
import json

# paths to folders
BASE = Path(__file__).parent
TEMPLATE_PATH = BASE.joinpath("template")
DEFAULTS_PATH = BASE.joinpath("defaults")
SIMS_PATH = BASE.jointpath("simulations")

# paths to templates
# SIM_TEMPLATE = TEMPLATE_PATH.joinpath("sim_template.jinja")
# TURB_TEMPLATE = TEMPLATE_PATH.joinpath("turb_template.jinja")
# RUN_TEMPLATE = TEMPLATE_PATH.joinpath("run_template.jinja")

# important constants
TASKS_PER_NODE = 48  # skx nodes on Stampede3

def safe_mkdir(dst, quiet=False):
    """make output directory without overwriting files"""

    dst.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print("Created directory", dst.resolve())

    return dst

def get_nnodes(inputs):
    """
    Computes the ideal number of nodes for PadeOps

    Rounds to an even number
    """
    nx = inputs["nx"]
    ny = inputs["ny"]
    nz = inputs["nz"]
    # compute n_nodes; round to nearest even number
    n_nodes = 2 * np.round(nx * ny * nz / 32**3 / TASKS_PER_NODE / 2)
    return n_nodes

def prep_parameters(new_inputs, default_path):
    params = json.load(default_path)
    for param_type in new_inputs:
        for param_name, param_val in new_inputs[param_type].items():
            params[param_type][param_name] = param_val
    return params

def write_padeops_files(params, template_path, quiet = False, sim_param_path = "sim.dat"):
    # update default params with user provided
    prep_parameters(new_inputs, default_path)
    # make output directory
    OUTPUT = safe_mkdir(params['outputdir'], quiet=quiet)
    # load sim template and write simulation's .dat file
    with open(template_path, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)
    # render output
    out = template.render(params)
    with open(OUTPUT / sim_param_path, "w") as f:
        f.write(out)
    # load turbine template and write simulation's .ini files (if there are turbines)

    # load run template and write .sh file to run simulation
    with open(OUTPUT / "run.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs, inputfile_name, problem_name="neutral_pbl", n_hrs=n_hrs
            )
        )
    if not quiet:
        print("\tDone writing spinups files")

# TODO:
# (1) Combine defaults and specified values
# (2) Fill in Jinja template for .dat file
# (3) Fill in Jinja template for .ini file
# (4) Fill in Jinja template for .sh file
# Write script that can generate a suite of experiments and has a CSV noting which are which

def write_sim(template, params, quiet):


def write_turbine(template, inputs, quiet=False, dst=None, n=1):
    """Writes a turbine file by copying ActuatorDisk_0001_input.j2"""
    # make output directory
    OUTPUT = safe_mkdir(inputs['turbine_dir'], quiet=quiet)

    with open(TURB_TEMPLATE, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / f"ActuatorDisk_{n:04d}_input.inp", "w") as f:
        f.write(out)

    if not quiet:
        print(f"\tDone writing ActuatorDisk_{n:04d}_input.inp file")

def sbatch_prep_args(
    inputs, name="input.dat", problem_name="neutral_pbl", node_cap=128, n_hrs=24
):
    """
    Returns a dictionary to populate fields in submit.j2 template

    Parameters
    ----------
    inputs : dict
        Dictionary, including the following fields:
            nx, ny, dirname
    name : str
        Name of input file to run
    problem_name : str, optional
        Name of problem file. Defaults to 'neutral_pbl'
    node_cap : int, optional
        Maximum number of nodes. Defaults to 128
    n_hrs : int, optional
        Number of wall clock hours. Defaults to 24

    Returns
    -------
    dict : ret
        Dictionary to populate the submit file template
    """

    n_hrs = int(n_hrs)
    n_nodes = get_nnodes(inputs)

    # fill out submit file
    if problem_name == "neutral_pbl_concurrent":
        ret = dict(
            n_nodes=n_nodes * 2,
            n_hrs=n_hrs,
            problem_dir="turbines",
            problem_name=problem_name,
        )
    elif problem_name == "neutral_pbl":
        ret = dict(
            n_nodes=n_nodes,
            n_hrs=n_hrs,
            problem_dir="incompressible",
            problem_name=problem_name,
        )
    else:
        raise NotImplementedError
    ret["n_nodes"] = int(np.max([np.min([ret["n_nodes"], node_cap]), 1]))
    ret["inputfile_name"] = name
    ret["dirname"] = inputs["dirname"]
    return ret


def sbatch_write_file(inputs, name, problem_name="neutral_pbl", node_cap=128, n_hrs=24):
    """Returns a filled out submit.sh template"""
    # laod template
    with open(TEMPLATE_SUBMIT, "r") as f:
        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # fill out submit file data
    submit_data = sbatch_prep_args(
        inputs, name, problem_name=problem_name, node_cap=node_cap, n_hrs=n_hrs
    )
    return template.render(submit_data)


def write_neutral(
    inputs,
    dst=None,
    quiet=False,
    inputfile_name="input_neutral.dat",
    n_hrs=24,  # allocated time, in hours
):
    """
    Write input file for spinup simulation

    Parameters
    ----------
    inputs : dict
        Dictionary of inputs, including:
            nx, ny, nz
            dirname,
            tstop,
            Lx, Ly (optional)
    dst : Path, optional
        Destination of written files. If none, defaults to inputs['dirname']
    quiet : bool, optional
        Silences print statements. Default False
    inputfile_name : str, optional
        String to title new input file
    """

    # make output directory
    OUTPUT = safe_mkdir(dst or inputs['dirname'], quiet=quiet)

    # load spinup template and write template:
    with open(TEMPLATE_NEUTRAL, "r") as f:

        template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

    # render output
    out = template.render(inputs)
    with open(OUTPUT / inputfile_name, "w") as f:
        f.write(out)

    # make submit.sh file
    with open(OUTPUT / "submit.sh", "w") as f:
        f.write(
            sbatch_write_file(
                inputs, inputfile_name, problem_name="neutral_pbl", n_hrs=n_hrs
            )
        )

    if not quiet:
        print("\tDone writing spinups files")

### Functions from Kirby I don't need yet! ###

# def find_last_restart(inputs, return_frameangle=True):
#     """
#     Finds the final restart file and gleans the TID
#     and frame angle, if requested
#     """
#     basedir = Path(inputs["restart_dir"])
#     RID = inputs["restart_rid"]
#     restart_files = basedir.glob(f"RESTART_Run{RID:02d}_info*")

#     # find the last restart file; largest TID
#     filename = None
#     tid = -1
#     for file in restart_files:
#         new_tid = int(
#             re.findall(r"info.(\d+)$", str(file))[0]
#         )  # glean the TID from the string
#         if new_tid > tid:
#             tid = new_tid
#             filename = file

#     if tid == 0:
#         # this did not find any files
#         warnings.warn(
#             "find_last_restart(): no restart files found, defaulting to TID 0"
#         )

#     if not return_frameangle:
#         return tid
#     else:
#         print(filename)
#         data = np.genfromtxt(filename, dtype=None)
#         if len(data) < 2:
#             frameangle = 0
#         else:
#             frameangle = -data[1]

#         return tid, frameangle

# def write_upsample(
#     inputs,
#     dst=None,
#     quiet=False,
#     inputfile_name="input_upsample.dat",
# ):
#     """
#     Writes upsampled fields inputs

#     nx, ny in `inputs` are the FINAL number of points in x and y, after upsampling
#     """

#     # update inputs
#     update_upsample_inputs(inputs)

#     # make output directory
#     OUTPUT = safe_mkdir(dst or inputs['dirname'], quiet=quiet)

#     # load spinup template and write template:
#     with open(TEMPLATE_UPSAMPLE, "r") as f:
#         template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

#     # render output
#     out = template.render(inputs)
#     with open(OUTPUT / inputfile_name, "w") as f:
#         f.write(out)

#     if not quiet:
#         print("\tDone writing upsampling files")


# def update_upsample_inputs(inputs):
#     """Glean the restart TID from the restart RID"""
#     TID = find_last_restart(inputs, return_frameangle=False)
#     inputs["restart_tid"] = TID
#     return  # updates dictionary, returns nothing


# def write_rotate(
#     inputs,
#     dst=None,
#     quiet=False,
#     inputfile_name="input_rotate.dat",
#     n_hrs=6,
# ):
#     """
#     Writes inputs for the rotation phase
#     """

#     # adds frame angle line to RESTART file
#     prep_rotation(inputs, quiet=quiet)

#     # make output directory
#     OUTPUT = safe_mkdir(dst or inputs['dirname'], quiet=quiet)

#     # load spinup template and write template:
#     with open(TEMPLATE_ROTATE, "r") as f:
#         template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

#     # render output
#     out = template.render(inputs)
#     with open(OUTPUT / inputfile_name, "w") as f:
#         f.write(out)

#     # make submit.sh file
#     with open(OUTPUT / "submit_rotation.sh", "w") as f:
#         f.write(
#             sbatch_write_file(
#                 inputs, inputfile_name, problem_name="neutral_pbl", n_hrs=n_hrs
#             )
#         )

#     if not quiet:
#         print("\tDone writing rotation files")


# def prep_rotation(inputs, rid=2, tid=0, frameangle=0.0, quiet=False):
#     """Appends frame angle line to rotation restart files"""
#     fname = Path(inputs["restart_dir"]) / f"RESTART_Run{rid:02d}_info.{tid:06d}"

#     try:
#         data = np.genfromtxt(fname, dtype=None)
#         if len(np.atleast_1d(data)) > 1:
#             return  # frame angle line already exists
#     except FileNotFoundError:
#         raise

#     with open(fname, "a") as src:
#         src.write(f"{frameangle:11.1f}")

#     if not quiet:
#         print("Added frame angle to retart files for rotation phase")


# def write_concurrent(
#     inputs,
#     dst=None,
#     quiet=False,
#     n_hrs=24,
# ):
#     """
#     Write concurrent files, main function. Calls helper function
#     _write_concurrent() after gleaning the restart TID and frame angle
#     from restart files.
#     """

#     # make output directory
#     OUTPUT = safe_mkdir(dst or inputs['dirname'], quiet=quiet)
#     # finish populating restart_tid, frame_angle fields
#     tid, frameangle = find_last_restart(inputs, return_frameangle=True)
#     inputs.update(frameangle=frameangle, restart_tid=tid)

#     for _template, name in zip(
#         [TEMPLATE_PRIMARY, TEMPLATE_PRECURSOR, TEMPLATE_CONCURRENT],
#         ["primary", "precursor", "main"],
#     ):
#         # load spinup template and write template:
#         with open(_template, "r") as f:
#             template = jinja2.Template(f.read(), undefined=jinja2.StrictUndefined)

#         # render output
#         out = template.render(inputs)
#         with open(OUTPUT / f"input_{name}.dat", "w") as f:
#             f.write(out)

#         if not quiet:
#             print(f"\tDone writing {name} files")

#     # make submit.sh file
#     with open(OUTPUT / "submit_concurrent.sh", "w") as f:
#         f.write(
#             sbatch_write_file(
#                 inputs,
#                 "input_main.dat",
#                 problem_name="neutral_pbl_concurrent",
#                 n_hrs=n_hrs,
#             )
#         )

if __name__ == "__main__":
    pass
