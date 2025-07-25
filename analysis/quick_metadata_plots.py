import padeopsIO as pio
import analysis_utils as au
# from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import numpy as np
import os
import csv
import math
import statistics
import numpy as np
import glob
from pathlib import Path
import moviepy.video.io.ImageSequenceClip
import re

# single image
def plot_run_power(run_folder, label = "", runid = 0):
    fig, ax = plt.subplots(figsize=(9, 3))
    sim = pio.BudgetIO(run_folder, padeops = True, runid = runid)
    power = sim.read_turb_power("all", turb=1)
    Cp = [au.power_to_Cp(p) for p in power]
    n_steps = len(Cp)
    strt_step = math.ceil(n_steps * 0.2)
    Cp = Cp[strt_step:]
    ax.plot(Cp, label=label, lw=0.7)
    plt.legend(loc="lower right")
    os.path.join(run_folder, 'run_power.png')
    plt.savefig(os.path.join(run_folder, 'run_power.png'))
    return


def get_sim_varied_params(suite_folder):
    fields = []
    rows = []
    with open(os.path.join(suite_folder, "sim_ids.csv"), mode='r') as file:
        csvreader = csv.reader(file)
        # extracting field names through first row
        fields = next(csvreader)
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
    return rows, fields


def plot_suite_power(suite_folder, save = True, figsize = (9, 3)):
    sub_folders = [f.path for f in os.scandir(suite_folder) if f.is_dir()]
    # Open the CSV file for reading
    fields = []
    rows = []

    with open(os.path.join(suite_folder, "sim_ids.csv"), mode='r') as file:
        csvreader = csv.reader(file)
        # extracting field names through first row
        fields = next(csvreader)
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
    
    # Make empty figure
    fig, ax = plt.subplots(figsize = figsize)
    # Plot power from each run
    for i, folder in enumerate(sub_folders):
        try:
            sim = pio.BudgetIO(folder, padeops = True, runid = 0)
            dt = sim.input_nml["input"]["dt"]
            trans_tau = int(math.ceil(50 / dt) + 1)
            # TODO: ask Kirby if default should be "all", rather than None
            # as it was confusing and when it is saved as a file first, they all printed I think?
            power = sim.read_turb_power("all", turb=1)[trans_tau:]
            Cp = [au.power_to_Cp(p) for p in power]
            time = [50 + dt * n for n in range(len(Cp))]
            label = ""
            nfields = len(fields)
            for j, fld in enumerate(fields):
                if fld != "id":
                    label += f"{fld}: {rows[i][j]}"
                    if nfields != j + 1: 
                        label += ", "
        except FileNotFoundError:
            continue

        ax.plot(time, Cp, label=label, lw=0.7)

    if save:
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(suite_folder, 'suite_cp.png'))
    return fig, ax

def _plot_instantaneous_field(save_folder, sim, *, tidx, field, xlim = [-5, 20], ylim =  [-5, 5],  zlim = 0, vmin = 0, vmax = 1, suptitle = "", set_plot_lims = False, plot_ylim = [0, 1.2], plot_zlim = None, colormap = "bwr"):
    ds = sim.slice(field_terms=[field], xlim = xlim, ylim = ylim, zlim = zlim, tidx = tidx)
    dims = ds.sizes
    ndims = len(dims)
    fig, ax = plt.subplots()
    if ndims == 1:
        dim_name = list(ds.sizes.keys())[0]
        ax.plot(ds[dim_name], ds[field])
        ax.set_xlabel(dim_name + '/D')
        ax.set_ylabel(field)
    elif ndims == 2:
        ds[field].imshow(ax = ax, cmap = colormap, vmin = vmin, vmax = vmax)
    if set_plot_lims:
        ax.set_ylim(plot_ylim)
    if plot_zlim is not None:
        ax.set_zlim(plot_zlim)
    plt.suptitle(suptitle)
    plt.title("TIDX: " + str(tidx))
    plt.savefig(os.path.join(save_folder, field + '_' + str(tidx) + '.png'))
    plt.close()
    return

def plot_instantaneous_field(sim_folder, runid, tidx = 0, field = "u", xlim = [-5, 20], ylim =  [-5, 5],  zlim = 0, **kwargs):
    run_folder = au.get_run_folder(sim_folder, runid)
    sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")
    save_folder = run_folder
    if tidx == "all":
        tidx_list = sim.unique_tidx()
        save_folder = os.path.join(run_folder, field + "_plots")
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        # get max/min values from last slice
        last_field_slice = sim.slice(field_terms=[field], xlim = xlim, ylim = ylim, zlim = zlim, tidx = tidx_list[-1])[field]
        vmin, vmax = np.min(last_field_slice), np.max(last_field_slice)
        for tidx in tidx_list:
            try:  # if simulation died early -> 
                _plot_instantaneous_field(save_folder, sim, tidx = tidx, field = field, xlim = xlim, ylim = ylim, zlim = zlim, vmin = vmin, vmax = vmax, **kwargs)
            except:
                return save_folder
    else:
        _plot_instantaneous_field(run_folder, sim, tidx = tidx, field = field, **kwargs)
    return save_folder

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(re.compile('([0-9]+)'), s)] 
 
def film_instantaneous_field(image_folder, fps = 10, video_name = "video.mp4"):
    image_files = [os.path.join(image_folder,img)
                for img in os.listdir(image_folder)
                if img.endswith(".png")]
    image_files.sort(key=natural_sort_key)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(os.path.join(image_folder, video_name))
    return

def plot_requested_turb_power(ax, folder, runs, labels, zoom = None, **kwargs):
    for i, r in enumerate(runs):
        run_folder = au.get_run_folder(folder, r)
        sim = pio.BudgetIO(run_folder, padeops = True, runid = 0)
        dt = sim.input_nml["input"]["dt"]
        trans_tau = int(math.ceil(50 / dt) + 1)
        # TODO: ask Kirby if default should be "all", rather than None
        # as it was confusing and when it is saved as a file first, they all printed I think?
        try:
            power = sim.read_turb_power("all", turb=1)[trans_tau:]
            Cp = [au.power_to_Cp(p) for p in power]
            time = [50 + dt * n for n in range(len(Cp))]
            if zoom is not None:
                time, Cp = au.x_zoom_plot(zoom, time, Cp)
            ax.plot(time, Cp, label = labels[i], **kwargs)
        except:
            continue
    return

def plot_requested_turb_u_velocity(ax, folder, runs, labels, zoom = None, induced = False, u_infty = 1, **kwargs):
    for i, r in enumerate(runs):
        run_folder = au.get_run_folder(folder, r)
        sim = pio.BudgetIO(run_folder, padeops = True, runid = 0)
        dt = sim.input_nml["input"]["dt"]
        trans_tau = int(math.ceil(50 / dt) + 1)
        # TODO: ask Kirby if default should be "all", rather than None
        # as it was confusing and when it is saved as a file first, they all printed I think?
        try:
            u_vel = sim.read_turb_uvel("all", turb=1)[trans_tau:]
            if induced:
                u_vel = [u_infty - u for u in u_vel]
            time = [50 + dt * n for n in range(len(u_vel))]
            if zoom is not None:
                time, u_vel = au.x_zoom_plot(zoom, time, u_vel)
            ax.plot(time, u_vel, label = labels[i], **kwargs)
        except:
            continue
    return

def plot_theoretical_turb_power(ax, CT, folder, run, zoom = None, color = 'black', **kwargs):
    run_folder = au.get_run_folder(folder, run)
    sim = pio.BudgetIO(run_folder, padeops = True, runid = 0)
    dt = sim.input_nml["input"]["dt"]
    trans_tau = int(math.ceil(50 / dt) + 1)
    power = sim.read_turb_power("all", turb=1)[trans_tau:]
    time = [50 + dt * n for n in range(len(power))]
    a = au.analytical_a(CT)
    Cp = [au.a_to_Cp(a)] * len(time)
    if zoom is not None:
        time, Cp = au.x_zoom_plot(zoom, time, Cp)
    ax.plot(time, Cp, label = "Analytical $C_p$", color = color, **kwargs)
    return

def plot_TI_vals(path, logfile):
    log_file_dict = pio.query_logfile(os.path.join(path, logfile), search_terms=["TIDX", "Time", "TI_fact", "TI_inst"], crop_equal = False)
    plt.plot(log_file_dict["TIDX"][::2], log_file_dict["TI_inst"][1:])
    plt.savefig(os.path.join(path, "test_inst.png"))

    plt.figure()
    plt.plot(log_file_dict["TIDX"][::2], log_file_dict["TI_fact"][1:])
    plt.savefig(os.path.join(path, "test_fact.png"))
    return