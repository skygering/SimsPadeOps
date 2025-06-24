import os
import analysis_utils as au
import padeopsIO as pio
import xarray as xr
import pandas as pd
import quick_metadata_plots as mplts
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import modred as mr
import math

keylab = pio.budgetkey.key_labels()

def get_timeslices(sim, fields, tidx_list = None):
    print("Collect Data")
    fields = ["u"]
    xlim = [-5, 20]
    ylim = 0
    zlim = [-2.5, 2.5]
    if tidx_list is None:
        tidx_list = sim.unique_tidx()
        tidx_list.sort()
    tidx_start = tidx_list[0]
    tidx_list = tidx_list[1:]
    ds = sim.slice(field_terms=fields, xlim = xlim, ylim = ylim, zlim = zlim, tidx = tidx_start)
    for tidx in tidx_list:
        try:
            ds_tidx = sim.slice(field_terms=fields, xlim = xlim, ylim = ylim, zlim = zlim, tidx = tidx)
            ds = xr.concat([ds, ds_tidx], dim = "time")
        except:
            break
    return ds

def get_data(folder, idx, suptitle, file_extension, turb_pos):
    run_folder = au.get_run_folder(folder, idx)
    # get the data
    print("Collect Data")
    sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin = turb_pos)
    ds = get_timeslices(sim, ["u", "p"])
    print("Save Data")
    ds.to_netcdf(os.path.join(modal_folder, file_extension + "_data.nc"))
    # make the plots
    print("Make Plots")
    save_folder = os.path.join(run_folder, fields[0] + "_" + fields[1] + "_plots")
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    colormap = "bwr"
    u_range = np.max(np.abs(1 - ds.u))  # centered at 1
    p_range = np.max(np.abs(ds.p))  # centered at 0
    n = len(tidx_list)
    sidx = round(n * 0.5)
    eidx = round(n * 0.75)
    for i in range(sidx, eidx + 1):
        print("Plot " + str(tidx_list[i]))
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5))
        try:
            ds_i = ds.isel(time=i)
        except:
            break
        # plot velocity
        im0 = ds_i[fields[0]].imshow(ax = ax0, cmap = colormap, vmin = 1 - u_range, vmax = 1 + u_range, cbar = False)
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes('right', size='5%', pad=0.05)
        cbar0 = fig.colorbar(im0, cax=cax0, orientation='vertical')
        cbar0.ax.tick_params(axis='both', which='major', labelsize=12)
        cbar0.set_label(keylab['u'], fontsize=14)
        ax0.set_title("Streamwise Velocity", size = 16)
        ax0.set_xlabel(keylab['x'], size = 14)
        ax0.set_ylabel(keylab['z'], size = 14)
        ax0.tick_params(axis='both', which='major', labelsize=12)
        # plot pressure
        im1 = ds_i[fields[1]].imshow(ax = ax1, cmap = colormap, vmin = -p_range, vmax = p_range, cbar = False)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cbar1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
        cbar1.ax.tick_params(axis='both', which='major', labelsize=12)
        cbar1.set_label(keylab['p'], fontsize=14)
        ax1.set_title("Streamwise Pressure", size = 16)
        ax1.set_xlabel(keylab['x'], size = 14)
        ax1.set_ylabel(keylab['z'], size = 14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        #
        plt.suptitle(suptitle + " (TIDX: " + str(tidx) + ")", size = 18)
        fig.tight_layout()
        # save
        plt.savefig(os.path.join(save_folder, fields[0] + "_" + fields[1] + '_' + str(tidx) + '.png'), dpi=400)
        plt.close()
    # print("Make Video")
    # qmplt.film_instantaneous_field(save_folder, fps = 20, video_name = "velocity_pressure_" + file_extension + ".mp4")
    return

# modal_folder = os.path.join(au.DATA_PATH, "F_0013_X_SU_PI_Files")

# stationary_idx = 0
# get_data(modal_folder, stationary_idx, "Stationary Turbine", "stationary", (5.0, 5.0, 2.5))

# surging_idx = 1
# surging_ds = get_data(modal_folder, surging_idx, "Surging Turbine", "surging", (5.0, 5.0, 2.5))

# pitching_idx = 2
# pitching_ds = get_data(modal_folder, pitching_idx, "Pitching Turbine", "pitching", (5.0, 5.0, 2.5))

# surging_pitching_idx = 3
# surging_pitching_ds = get_data(modal_folder, surging_pitching_idx, "Surging and Pitching Turbine", "surging_pitching", (5.0, 5.0, 5.0))

modal_folder = os.path.join(au.DATA_PATH, "F_0014_X_SU_PI_Files")

# stationary_idx = 0
# get_data(modal_folder, 8, "(nx, ny, nz) = (512, 256, 256) and dt = 0.012", "run08", (5.0, 2.5, 2.5))
# save_folder = os.path.join(modal_folder, "Sim_0005/u_p_plots")
# qmplt.film_instantaneous_field(save_folder, fps = 20, video_name = "velocity_pressure_05.mp4")

def get_pod_data(ds, num_modes = 8, tidx = 200):
    da = ds.u
    da = da.transpose("x", "z", "time")
    da_mean = da.mean(dim = "time")
    da_prime = da - da_mean
    vecs = np.array(da_prime)
    nx, nz, nt = vecs.shape
    vecs = vecs.reshape(nx * nz, nt)
    POD_res = mr.compute_POD_arrays_snaps_method(vecs, list(mr.range(num_modes)))
    modes = POD_res.modes.reshape(nx, nz, num_modes)
    eigvals = POD_res.eigvals
    proj_coeffs = POD_res.proj_coeffs[0:num_modes, tidx]
    reconstruct = da_mean + np.sum([proj_coeffs[i] * modes[:, :, i] for i in range(modes.shape[2])], axis = 0)
    rmse = math.sqrt((np.square(da[:, :, tidx] - reconstruct)).mean())
    return eigvals, rmse

def get_spod_data(ds, mode_idxs = range(0, 3), freq_idxs = range(1, 3)):
    return

df = pd.DataFrame(columns = ["nx", "dt", "rmse", "mode", "e_val", "e_ratio"])
turb_pos = (5.0, 2.5, 2.5)
rows, fields = mplts.get_sim_varied_params(modal_folder)
ids, nx, ny, nz, dt, filterWidth= zip(*rows)
for (i, id_str) in enumerate(ids):
    print("Collecting data from folder: " + id_str)
    run_folder = os.path.join(modal_folder, "Sim_" + id_str)
    sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin = turb_pos)
    try:
        ds = get_timeslices(sim, ["u"], tidx_list = range(2500, 5500, 10))
        eigvals, rmse = get_pod_data(ds)
        for (j, e) in enumerate(eigvals[:8]):
            df.loc[len(df)] = [nx[i], dt[i], rmse, str(j), e, e/eigvals[0]]
    except:
        continue
df.to_csv(os.path.join(modal_folder, 'pod_data.csv'))
    
