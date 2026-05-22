# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import padeopsIO as pio
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
keylab = pio.budgetkey.key_labels()
from pathlib import Path
import random
import imageio
import shutil
import pandas as pd

import modred as mr

data_folder = "/Users/sky/src/HowlandLab/data"


# %%
def get_clean_data(file_name, data_folder = "/Users/sky/src/HowlandLab/data"):
    with xr.open_dataset(os.path.join(data_folder, file_name)) as ds:
        ds = ds.isel(time = slice(100, None))
    return ds


# %%
def plot_u(ds, t_idx, mean_val = 1):
    da = ds.u
    range_val = np.max(np.abs(mean_val - da))  # centered at 1
    fig = plot_field(da.isel(time = t_idx), mean_val - range_val, mean_val + range_val, "Streamwise Velocity", keylab['u'])
    return fig

def plot_p(ds, t_idx):
    da = ds.p
    range_val = np.max(np.abs(da))
    fig = plot_field(da.isel(time = t_idx), -range_val, range_val, "Streamwise Pressure", keylab['p'])
    return fig

def plot_vals(arr, min_val, max_val, title, label):
    fig = plot_field(arr, min_val, max_val, title, label)
    return fig

def plot_field(da, min_val, max_val, title, label):
    colormap = "bwr"
    fig, ax = plt.subplots(1, 1, dpi = 300)
    # plot velocity
    im = da.imshow(ax = ax, cmap = colormap, vmin = min_val, vmax = max_val, cbar = False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(axis='both', which='major', labelsize=12)
    cbar.set_label(label, fontsize=14)
    ax.set_title(title, size = 16)
    ax.set_xlabel(keylab['x'], size = 14)
    ax.set_ylabel(keylab['z'], size = 14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    return fig

def ms2fps( ms ):
    ''' Converts milliseconds to frames per second '''
    return 1000 / ms

def animate_field(ds, frame_function, anim_path, fps = 10, loop = False, **kwargs):
    # make a temporary directory
    fp_out = Path( anim_path ).resolve()
    fp_out_base = fp_out.stem
    dir_out = fp_out.parents[0]
    randstr = str(random.randint(1000000, 9999999) )
    dir_tmp = dir_out / f'xanim_tmp_{fp_out_base}.{randstr}'
    os.mkdir(dir_tmp)
    # plot data
    ntimes = len(ds.time)
    ndigits = len(str(ntimes))
    fp_list = []
    for i in range(ntimes):
        # try:
        fig = frame_function(ds, i, **kwargs)
        idx_str = str(i).zfill(ndigits)
        fn_out_ii = dir_tmp / f'xanim_tmpfile_{idx_str}.png'
        fig.savefig(fn_out_ii, dpi = 300)
        fp_list.append( dir_tmp / fn_out_ii )
        plt.close(fig)
        # except:
        #     break
    print(fp_list)
    frames = []
    for ii in range(ntimes):
        image = imageio.v2.imread(fp_list[ii])
        frames.append(image)

    duration = ms2fps( fps )
    imageio.mimsave(fp_out, frames, duration = duration, loop = loop)    
    shutil.rmtree(dir_tmp)



# %%
def get_u_pod_modes(ds, num_modes, **kwargs):
    return get_pod_modes(ds.u, num_modes, label = keylab['u'], **kwargs)

def get_p_pod_modes(ds, num_modes, **kwargs):
    return get_pod_modes(ds.p, num_modes, label = keylab['p'], **kwargs)

def get_pod_modes(da, num_modes, label = "", avg_val = None, plot = True, tidx = 0, **kwargs):
    da = da.transpose("x", "z", "time")
    da_mean = da.mean(dim = "time")
    da_prime = da - da_mean
    vecs = np.array(da_prime)
    nx, nz, nt = vecs.shape
    vecs = vecs.reshape(nx * nz, nt)
    POD_res = mr.compute_POD_arrays_snaps_method(vecs, list(mr.range(num_modes)))
    modes = POD_res.modes.reshape(nx, nz, num_modes)
    eigvals = POD_res.eigvals
    reconstruct = calc_reconstruct(POD_res, da_mean, num_modes, tidx, nx, nz)
    if plot:
        plot_pod_modes(modes, eigvals, avg_val, **kwargs)
        plot_reconstruction(reconstruct, **kwargs)
    return POD_res, reconstruct

def calc_reconstruct(POD_res, da_mean, num_modes, tidx, nx, nz):
    modes = POD_res.modes.reshape(nx, nz, num_modes)
    proj_coeffs = POD_res.proj_coeffs[0:num_modes, tidx]
    reconstruct = da_mean + np.sum([proj_coeffs[i] * modes[:, :, i] for i in range(modes.shape[2])], axis = 0)
    return reconstruct

def plot_pod_modes(modes, eigvals, avg_val, mode_path = None, **kwargs):
    nmodes = modes.shape[2]
    ncols = 2
    nrows = math.ceil(nmodes / ncols)
    fig, axes = plt.subplots(nrows, ncols, dpi = 300, figsize = (8, 6), sharex=True, sharey = True)
    if avg_val is None:
        avg_val = np.average(modes[:, :, :])
    range_val = np.max(np.abs(modes - avg_val))
    r, c = 0, 0
    e1 = eigvals[0]
    for i in range(nmodes):
        imode = modes[:, :, i]
        iax = axes[r, c]
        im = iax.imshow(imode.transpose(), cmap='bwr', vmin = avg_val - range_val, vmax = avg_val + range_val, extent=[-5,20,-2.5,2.5])
        iax.set_title(f"$\lambda_{i + 1} / \lambda_1$ = " + str(round(eigvals[i] / e1, ndigits=3)), size = 14)
        if c == 0:
            iax.set_ylabel(keylab['z'], size = 12)
        if r == nrows - 1:
            iax.set_xlabel(keylab['x'], size = 12)
        iax.tick_params(axis='both', which='major', labelsize=10)
        c += 1
        if c == ncols:
            c = 0
            r += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.01, 0.25, 0.01, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    if mode_path is not None:
        fig.savefig(mode_path, dpi = 300)
    return

def plot_reconstruction(reconstruct, recon_path = None, **kwargs):
    fig = plot_field(reconstruct, min_val=0, max_val=2, title = "Reconstructed Streamwise Velocity", label = keylab["u"]);
    if recon_path is not None:
        fig.savefig(recon_path, dpi = 300)
    return


# %% [markdown]
# ## POD

# %%
ds5 = get_clean_data("run05_data.nc")
tidx = 250
fig = plot_u(ds5, tidx, mean_val=1);
fig.savefig(os.path.join(data_folder, "init_u_250.png"), dpi = 300)

# %%
fig = plot_p(ds5, tidx);
fig.savefig(os.path.join(data_folder, "init_p_250.png"), dpi = 300)

# %%
nmodes = 8
POD_res, reconstruct = get_u_pod_modes(ds5, nmodes, tidx = tidx, mode_path = os.path.join(data_folder, "pod_5.png"), recon_path = os.path.join(data_folder, "recon_5.png"))

# %%
POD_res.modes.shape


# %%
def rmse(ds, reconstruct, tidx):
    return math.sqrt((np.square(ds[tidx, :, :] - reconstruct)).mean())


# %%
nmodes = 8
da_mean = ds5.u.mean(dim = "time")
(nt, nx, nz) = ds5.u.shape
POD_res, reconstruct = get_u_pod_modes(ds5, nmodes, tidx = 0, plot = False)
rmse_vals = np.array([rmse(ds5.u, reconstruct, 0)])
for t in ds5["time"].values[1:]:
    reconstruct = calc_reconstruct(POD_res, da_mean, 8, tidx, nx, nz)
    POD_res, reconstruct = get_u_pod_modes(ds5, nmodes, tidx = t, plot = False)
    rmse_vals = np.append(rmse_vals, rmse(ds5.u, reconstruct, t))
rmse_vals

# %%
fig, ax = plt.subplots(1, 1, dpi = 300)
ax.plot(ds5["time"].values, rmse_vals)
ax.set_title("RMSE of 8 Mode Reconstruction over Time", size = 18)
ax.set_ylabel("RMSE")
ax.set_xlabel("Timesteps")
fig.tight_layout()
fig.savefig(os.path.join(data_folder, "pod_rmse.png"), dpi = 300)

# %%
np.average(rmse_vals)

# %% [markdown]
# ## Explore using different length time series

# %%
nmodes = 8
n_ndt = 3
rmse_vals = np.zeros(n_ndt)
modes = np.zeros((n_ndt, nmodes))
dt_vals = np.zeros(n_ndt)
tidx = 100
for (i, n_dt) in enumerate(range(200, 700, 200)):
    dt_vals[i] = n_dt
    ds_trimmed = ds5.isel(time = slice(0, n_dt))
    POD_res, reconstruct = get_u_pod_modes(ds_trimmed, nmodes, tidx = tidx, plot = False)
    rmse_vals[i] = rmse(ds_trimmed.u, reconstruct, tidx)
    modes[i] = POD_res.eigvals[:8]

dt_vals, rmse_vals, modes

# %%
fig, ax0 = plt.subplots(1, 1, figsize = (5, 4))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for (i, nt) in enumerate(dt_vals):
    ax0.scatter(range(1, len(modes[i])), modes[i][1:] / modes[i, 0], color = colors[i], label = str(round(nt)))

ax0.set_yscale('log')
ax0.set_ylabel("$\lambda_i / \lambda_0$")
ax0.set_xlabel("Mode ($i$)")
ax0.legend(title = "Saved Timesteps")
ax0.set_title("$\lambda_i / \lambda_0$ for Different Timesteps")
fig.tight_layout()
fig.savefig(os.path.join(data_folder, "timestep_effect_5.png"), dpi = 300)

# %%
# fig, ax0 = plt.subplots(1, 1, figsize = (5, 4))
# ax0.scatter(dt_vals, rmse_vals)
# ax0.set_ylabel("RMSE")
# ax0.set_xlabel("# of Saved Timesteps")
# ax0.set_title("RMSE of Reconstruction with 8 Modes")
# fig.tight_layout()
# fig.savefig(os.path.join(data_folder, "timestep_effect_rmse_5.png"), dpi = 300)

# %%
import seaborn as sns
pod_df = pd.read_csv(os.path.join(data_folder, "new_pod_data.csv"))
pod_df["timestep"] = pod_df.apply(lambda row: round(row["dt"], ndigits = 3), axis = 1)
pod_df_ratio = pod_df[pod_df["mode"] != 0]

fig, ax0 = plt.subplots(1, 1, figsize = (5, 4), dpi = 300)
sns.scatterplot(ax = ax0, data = pod_df_ratio, x = "mode", y = "e_ratio", hue = "nx", style = "timestep", palette="tab10", s = 60)
ax0.set_title("Simulation Parameter Effects on POD", size = 18, pad = 20)
ax0.set_yscale('log')
ax0.set_ylabel("$\lambda_i / \lambda_0$")
ax0.set_xlabel("Mode ($i$)")
leg = ax0.legend()
bb = leg.get_bbox_to_anchor().transformed(ax0.transAxes.inverted())
xOffset = 0.5
bb.x0 += xOffset
bb.x1 += xOffset
leg = leg.set_bbox_to_anchor(bb, transform = ax0.transAxes)
fig.savefig(os.path.join(data_folder, "pod_mode_convergence.png"), dpi = 300)

# %%
pod_df_small_dt = pod_df[pod_df["timestep"] == 0.024]
modes = pod_df_small_dt["mode"].unique()
modes.sort()
modes = modes[1:]
p_vals = [0 for _ in modes]
e_vals = [0 for _ in modes]
d24 = [0 for _ in modes]
d12 = [0 for _ in modes]
for (i, m) in enumerate(modes):
    mode_dt = pod_df_small_dt[pod_df_small_dt["mode"] == m]
    v4 = mode_dt[mode_dt["nx"] == 128]["e_ratio"].values[0]
    v2 = mode_dt[mode_dt["nx"] == 256]["e_ratio"].values[0]
    v1 = mode_dt[mode_dt["nx"] == 512]["e_ratio"].values[0]
    d24[i] = v2 - v4
    d12[i] = v1 - v2
    p_vals[i] = np.log(d24[i] / d12[i]) / np.log(2)
e_vals = [abs(d12[i] / ((2**p) - 1)) for (i, p) in enumerate(p_vals)]

# %%
data = {'Modes': range(1, len(modes) + 1),
        "Low/Medium Difference": d24,
        "Medium/High Difference": d12,
        'Order of Accuracy': p_vals,
        '|Discretization Error|': e_vals}

# Create DataFrame
df = pd.DataFrame(data)
cm = sns.color_palette("vlag", as_cmap=True)
df.style.background_gradient(cmap=cm, subset = ["|Discretization Error|"])

# %%
pod_df_med_res = pod_df[pod_df["nx"] == 256]
modes = pod_df_med_res["mode"].unique()
modes.sort()
modes = modes[1:]
p_vals = [0 for _ in modes]
e_vals = [0 for _ in modes]
d24 = [0 for _ in modes]
d12 = [0 for _ in modes]
for (i, m) in enumerate(modes):
    mode_dt = pod_df_med_res[pod_df_med_res["mode"] == m]
    v4 = mode_dt[mode_dt["timestep"] == 0.049]["e_ratio"].values[0]
    v2 = mode_dt[mode_dt["timestep"] == 0.024]["e_ratio"].values[0]
    v1 = mode_dt[mode_dt["timestep"] == 0.012]["e_ratio"].values[0]
    d24[i] = v2 - v4
    d12[i] = v1 - v2
    p_vals[i] = np.log(d24[i] / d12[i]) / np.log(2)
e_vals = [abs(d12[i] / ((2**p) - 1)) for (i, p) in enumerate(p_vals)]

# %%
data = {'Modes': range(1, len(modes) + 1),
        "Low/Medium Difference": d24,
        "Medium/High Difference": d12,
        'Order of Accuracy': p_vals,
        '|Discretization Error|': e_vals}

# Create DataFrame
df = pd.DataFrame(data)
cm = sns.color_palette("vlag", as_cmap=True)
df.style.background_gradient(cmap=cm, subset = ["|Discretization Error|"])

# %%
pod_df_med_res = pod_df[pod_df["nx"] == 128]
modes = pod_df_med_res["mode"].unique()
modes.sort()
modes = modes[1:]
p_vals = [0 for _ in modes]
e_vals = [0 for _ in modes]
d24 = [0 for _ in modes]
d12 = [0 for _ in modes]
for (i, m) in enumerate(modes):
    mode_dt = pod_df_med_res[pod_df_med_res["mode"] == m]
    v4 = mode_dt[mode_dt["timestep"] == 0.195]["e_ratio"].values[0]
    v2 = mode_dt[mode_dt["timestep"] == 0.098]["e_ratio"].values[0]
    v1 = mode_dt[mode_dt["timestep"] == 0.049]["e_ratio"].values[0]
    d24[i] = v2 - v4
    d12[i] = v1 - v2
    p_vals[i] = np.log(d24[i] / d12[i]) / np.log(2)
e_vals = [abs(d12[i] / ((2**p) - 1)) for (i, p) in enumerate(p_vals)]

# %%
data = {'Modes': range(1, len(modes) + 1),
        "Low/Medium Difference": d24,
        "Medium/High Difference": d12,
        'Order of Accuracy': p_vals,
        '|Discretization Error|': e_vals}

# Create DataFrame
df = pd.DataFrame(data)
cm = sns.color_palette("vlag", as_cmap=True)
df.style.background_gradient(cmap=cm, subset = ["|Discretization Error|"])

# %% [markdown]
# ## SPOD

# %%
import sys
SPOD_path = "/Users/sky/src/HowlandLab/spod_python"
sys.path.insert(0, SPOD_path)
import spod
import h5py
import pylab
from matplotlib import cm

# %%
save_fig  = True  # postprocess figs
save_path = data_folder

# %%
params={
'axes.labelsize': '20',
'xtick.labelsize': '16',
'ytick.labelsize': '16',
'lines.linewidth': 1.5,
'legend.fontsize': '14',
'figure.figsize': '8, 6'    # set figure size
}
pylab.rcParams.update(params)

def figure_format(xtitle, ytitle, zoom, legend):
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.axis(zoom)
    if legend != 'None':
        plt.legend(loc=legend)


# %%
ds5 = get_clean_data("run05_data.nc")
da = ds5.u
da = da.transpose("time", "x", "z")
da_mean = da.mean(dim = "time")
da_prime = da - da_mean
vecs = np.array(da_prime)
nt, nx, nz = vecs.shape
vecs = vecs.reshape(nt, nx * nz)
dt = 0.0244

# %%
spod.spod(vecs, dt, save_path, method='fast')

# %%
SPOD_LPf  = h5py.File(os.path.join(save_path,'SPOD_LPf.h5'),'r')
L = SPOD_LPf['L'][:,:]    # modal energy E(f, M)
P = SPOD_LPf['P'][:,:,:]  # mode shape
f = SPOD_LPf['f'][:]      # frequency
SPOD_LPf.close()

# %%
fig = spod.plot_spectrum(f,L,hl_idx=5)
figure_format(xtitle='Frequency (Hz)', ytitle='SPOD mode energy',
              zoom=[1, 1.9*10**1, 10**-1, 2*10**3], legend='best')

if save_fig:
    plt.savefig(os.path.join(save_path,'Spectrum.png'), dpi=300, bbox_inches='tight')
    plt.close()
fig


# %%
def plot_spod_modes(P, L, f, M_idxs, f_idxs, mode_path = None, nx = 512, nz = 256, label = "", avg_val = None, **kwargs):
    ncols = len(f_idxs)
    nrows = len(M_idxs) + 1
    fig, axes = plt.subplots(nrows, ncols, dpi = 300, sharex=True, sharey = True, gridspec_kw={'height_ratios': [0.5] + [4] * (nrows - 1)})
    fig.suptitle("SPOD Frequency Modes", size = 18)
    for (c, fj) in enumerate(f_idxs):
        axes[0, c].axis("off")
        axes[0, c].set_title(f"Frequency: {round(f[fj], 2)}", size = 15)

    modes_real = np.real(P[:,:,:])
    if avg_val is None:
        avg_val = np.average(modes_real)
    range_val = np.max(np.abs(modes_real - avg_val))
    for (i, Mi) in enumerate(M_idxs):
        for (j, fj) in enumerate(f_idxs):
            e1 = L[fj, 0]  # largest energy in a given frequency
            ijmode = modes_real[fj, : ,Mi]
            ijmode = ijmode.reshape(nx, nz)
            ijax = axes[i + 1, j]
            im = ijax.imshow(ijmode.transpose(), cmap='bwr', vmin = avg_val - range_val, vmax = avg_val + range_val, extent=[-5,20,-2.5,2.5])
            ijax.set_title(f"$\lambda_{Mi + 1} / \lambda_1$ = " + str(round(L[fj, Mi] / e1, ndigits=3)), size = 14)
            if j == 0:
                ijax.set_ylabel(keylab['z'], size = 12)
            if i == nrows - 2:
                ijax.set_xlabel(keylab['x'], size = 12)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.01, 0.25, 0.01, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(axis='both', which='major', labelsize=12)
    fig.tight_layout()
    if mode_path is not None:
        fig.savefig(mode_path, dpi = 300)
    return


# %%
Ms = np.array([0, 1, 2, 3, 4, 5, 6, 7])
fs = np.array([0, 1, 2, 3, 4])
plot_spod_modes(P, L, f, Ms, fs, mode_path = os.path.join(data_folder, "spod_modes.png"))

# %%
data_rec = spod.reconstruct_time_method(vecs,dt,f,P,Ms,fs, method = "fast", save_path=save_path)

# %%
tidx = 250
data_rec_tidx = da_mean + data_rec.reshape(nt, nx, nz)[tidx, :, :]
plot_field(data_rec_tidx, 0, 2, "SPOD Reconstructed Velocity Field", keylab['u']);

# %%
rmse(ds5.u, data_rec_tidx, tidx)

# %%
ds5["time"].values.shape

# %%
da_mean = ds5.u.mean(dim = "time")
(nt, nx, nz) = ds5.u.shape
rmse_vals = []
for t in ds5["time"].values:
    data_rec_t = da_mean + data_rec.reshape(nt, nx, nz)[t, :, :]
    rmse_vals = np.append(rmse_vals, rmse(ds5.u, data_rec_t, t))
rmse_vals

# %%
fig, ax = plt.subplots(1, 1, dpi = 300)
ax.plot(ds5["time"].values, rmse_vals)
ax.set_title("RMSE of 3 Mode/Frequency Reconstruction over Time", size = 18)
ax.set_ylabel("RMSE")
ax.set_xlabel("Timesteps")
fig.tight_layout()
fig.savefig(os.path.join(data_folder, "spod_rmse.png"), dpi = 300)

# %% [markdown]
# # Laminar Flow
#
# I began by exploration with laminar flows at a medium resolution of (nx, ny, nz) = (256, 128, 128). I also started with a high amplitude and frequency of movemement with frequency $f = 1$ for all moving turbines and amplitude of $A = 0.5$ for surging turbines and $A = 5^\circ$ for pitching turbines.

# %% [markdown]
# ## Stationary Turbine

# %%
# stationary_ds  = get_clean_data("stationary_data_r_256_t_0.nc")

# %%
# plot_u(stationary_ds, 1)

# %%
# plot_p(stationary_ds, 1800)

# %% [markdown]
# ## Surging Turbine

# %%
# surging_ds  = get_clean_data("surging_data_r_256_t_0.nc")

# %%
# plot_u(surging_ds, 1800)

# %%
# plot_p(surging_ds, 1800)

# %%
# modes, eigenvals = get_u_pod_modes(surging_ds, 6)

# %%
# modes, eigs = get_p_pod_modes(surging_ds, 4)

# %%
# params = {"time_step":0.062, "n_space_dims":2, "n_variables":1, "n_dft":50, "overlap":50, "mean_type":"longtime", "n_modes_saved":6}

# %%
# standard  = spod_standard(params=params, comm=None)
# spod = standard.fit(data_list=surging_ds.u)

# %%
# spod.plot_eigs()

# %%
# spod.plot_eigs_vs_frequency()

# %%
# f1, f1_idx = spod.find_nearest_freq(freq_req=2 * math.pi, freq=spod.freq)
# spod.plot_2d_modes_at_frequency(freq_req=f1, freq=spod.freq, modes_idx=[0,1,2], x1=surging_ds.x, x2=surging_ds.z, equal_axes=True)

# %% [markdown]
# ## Pitching Turbine

# %%
# pitching_ds = get_clean_data("pitching_data_r_256_t_0.nc")

# %%
# plot_u(pitching_ds, 1800)

# %%
# plot_p(pitching_ds, 1800)

# %%
# modes, eigenvals = get_u_pod_modes(pitching_ds, 6)

# %%
# modes, eigenvals = get_p_pod_modes(pitching_ds, 6)

# %%
# spod = standard.fit(data_list=pitching_ds.u)

# %%
# spod.plot_eigs_vs_frequency()

# %% [markdown]
# ## Surging and Pitching Turbine

# %%
# surging_pitching_ds = get_clean_data("surging_pitching_data_r_256_t_0.nc")

# %%
# plot_u(surging_pitching_ds, 1800)

# %%
# plot_p(surging_pitching_ds, 1800)

# %%
# modes, eigenvals = get_u_pod_modes(surging_pitching_ds, 6)

# %%
# modes, eigenvals = get_p_pod_modes(surging_pitching_ds, 6)

# %%

# %%
