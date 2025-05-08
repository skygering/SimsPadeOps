import os
import analysis_utils as au
import padeopsIO as pio
import xarray as xr
import pandas as pd
import quick_metadata_plots as qmplt
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

keylab = pio.budgetkey.key_labels()

def get_data(folder, idx, suptitle, file_extension, turb_pos):
    run_folder = au.get_run_folder(folder, idx)
    # get the data
    print("Collect Data")
    sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin = turb_pos)
    fields = ["u", "p"]
    xlim = [-5, 20]
    ylim = 0
    zlim = [-2.5, 2.5]
    tidx_list = sim.unique_tidx()
    tidx_list.sort()
    tidx_list = tidx_list[1:]
    ds = sim.slice(field_terms=fields, xlim = xlim, ylim = ylim, zlim = zlim, tidx = 0)
    for tidx in tidx_list:
        ds_tidx = sim.slice(field_terms=fields, xlim = xlim, ylim = ylim, zlim = zlim, tidx = tidx)
        ds = xr.concat([ds, ds_tidx], dim = "time")
    print("Save Data")
    ds.to_netcdf(os.path.join(modal_folder, file_extension + "_data_r_256_t_0.nc"))
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
    for i, tidx in enumerate(tidx_list[sidx:eidx]):
        print("Plot " + str(tidx))
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5))
        ds_i = ds.isel(time=i)
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

modal_folder = os.path.join(au.DATA_PATH, "F_0013_X_SU_PI_Files")

# stationary_idx = 0
# get_data(modal_folder, stationary_idx, "Stationary Turbine", "stationary", (5.0, 5.0, 2.5))

# surging_idx = 1
# surging_ds = get_data(modal_folder, surging_idx, "Surging Turbine", "surging", (5.0, 5.0, 2.5))

# pitching_idx = 2
# pitching_ds = get_data(modal_folder, pitching_idx, "Pitching Turbine", "pitching", (5.0, 5.0, 2.5))

surging_pitching_idx = 3
surging_pitching_ds = get_data(modal_folder, surging_pitching_idx, "Surging and Pitching Turbine", "surging_pitching", (5.0, 5.0, 5.0))

# total_ds = xr.concat([stationary_ds, surging_ds, pitching_ds, surging_pitching_ds], pd.Index(["stationary", "surging", "pitching", "surging_pitching"], name="movement"))
# total_ds = total_ds.assign_attrs(resolution="(256, 128, 128)", turbulence="NA")
# total_ds.to_netcdf(os.path.join(modal_folder, "data_r_256_t_0.nc"))