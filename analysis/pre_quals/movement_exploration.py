# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: simspadeops-py3.11
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from UnifiedMomentumModel.Momentum import UnifiedMomentum, ThrustBasedUnified
from matplotlib.colors import ListedColormap
from scipy.stats import describe
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pickle
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, irfft, rfftfreq


# %% [markdown]
# # Analytical Models

# %% [markdown]
# ## Define Analytical Models

# %%
def classical(ctp):  # return Cp for classical momentum theory
    a = ctp / (4 + ctp)
    return 4 * a * (1 - a)**2


# %%
def johlas(ctp, sa, pa):  # return Cp for classical momentum theory adjusted by coefficients in Johlas 2020
    cp = classical(ctp)
    x_disp = (1 + (3 * (sa**2)) / 2)
    theta_disp = np.mean([np.cos(np.deg2rad(pa) * np.sin(x))**3 for x in np.linspace(0, 2 * np.pi, 250)])
    return cp * x_disp * theta_disp  # coefficients from turbine movement


# %%
def umm(ctp, sf, sa, pa):
    model = UnifiedMomentum()
    sf_rad = np.deg2rad(sf)
    sin_vals = np.array([np.sin(sf_rad * x) for x in np.linspace(0, 2 * np.pi / sf_rad, 250)])
    sa_vals = (sa * sin_vals) + 1
    sa_vals_cubed = sa_vals**3
    pa_vals = np.deg2rad(pa) * sin_vals
    umm_vals = [sa_vals_cubed * model(Ctprime = c, yaw = 0.0, tilt = pa_vals).Cp for c in np.atleast_1d(ctp)] # list of list
    Cp_stats_list = [describe(vals) for vals in umm_vals]
    umm_means = np.array([stats.mean for stats in Cp_stats_list])
    umm_std = np.array([np.std(vals) for vals in umm_vals])
    umm_skewness = np.array([stats.skewness for stats in Cp_stats_list])
    umm_kurtosis = np.array([stats.kurtosis for stats in Cp_stats_list])
    return umm_means, umm_std, umm_skewness, umm_kurtosis


# %% [markdown]
# ## Plot Analytical Models

# %%
palette =['tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:red']

# %% [markdown]
# ### For this initial exploration, we will use $f = 1.0$ and $A_S = 0.5$ & $A_P = 5^\circ$. 

# %%
ctp_list = np.linspace(0.01, 10, num = 100)


# %%
def fix_plot_legend(ax, xOffset = 1.85, title = ""):
    leg = ax.legend(title = title)
    bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
    return


# %%
def plot_mean_cp_models(ctp_list, f, sa, pa, palette = palette):
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey = True, figsize = (10, 4), dpi = 300)
    fig.suptitle("Mean $C_P$ vs $C_T^'$", size = 16, y = 1.04)
    ax0.set_title(f"Surging ($f = {f}$, A = ${sa}$)", size = 14, y = 1.04)
    ax1.set_title(f"Pitching ($f = {f}$, A = ${pa}^\circ$)", size = 14, y = 1.04)
    ax0.set_xlabel('$C_T^\'$', size = 15)
    ax1.set_xlabel('$C_T^\'$', size = 15)
    ax0.set_ylabel('Mean $C_P$', size = 15)
    ax1.set_ylabel('Mean $C_P$', size = 15)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax0.set_ylim(0, 1)
    linewidth = 3.5
    classical_cp = classical(ctp_list)
    # plot surging
    johlas_cp_surge = johlas(ctp_list, sa, 0)
    umm_cp_surge_means, _, _, _ = umm(ctp_list, f, sa, 0)
    sns.lineplot(ax = ax0, x = ctp_list, y = classical_cp, color=palette[0], label = "Classical", linewidth = linewidth)
    sns.lineplot(ax = ax0, x = ctp_list, y = johlas_cp_surge, color=palette[1], label = "Johlas", linewidth = linewidth)
    sns.lineplot(ax = ax0, x = ctp_list, y = umm_cp_surge_means, color=palette[2], label = "Quasi-Steady UMM", linewidth = linewidth)
    # plot pitching
    johlas_cp_pitch = johlas(ctp_list, 0, pa)
    umm_cp_pitch_means, _, _, _ = umm(ctp_list, f, 0, pa)
    sns.lineplot(ax = ax1, x = ctp_list, y = classical_cp, color=palette[0], linewidth = linewidth, legend=False)
    sns.lineplot(ax = ax1, x = ctp_list, y = johlas_cp_pitch, color=palette[1], linewidth = linewidth, legend=False)
    sns.lineplot(ax = ax1, x = ctp_list, y = umm_cp_pitch_means, color=palette[2], linewidth = linewidth, legend=False)

    # adjust plot legend and spacing
    fix_plot_legend(ax0)
    fig.subplots_adjust(wspace=0.25)
    return fig, (ax0, ax1)



# %%
f, sa, pa = 1, 0.5, 5
plot_mean_cp_models(ctp_list, f, sa, pa);


# %% [markdown]
# # Analyze LES Data

# %%
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

df = pd.read_csv("/Users/sky/src/HowlandLab/data/expanded_moving_analysis_july_27_25.csv")
df['Movement'] = df.apply(lambda row: ("Stationary" if row.marker == "o" else ("Surging" if row.marker == "s" else "Pitching")), axis = 1)
df['rounded_dt'] = df.apply(lambda row: round(row["dt"], ndigits = 3), axis = 1)
df = df[df.nx >= 256]  # only keep simulations with high enough resolutions
df = df[df.filterFactor >= 1.5] # only keep simulations with high enough filter factors
df = df[df.useCorrection]
df = df[df.turbulence == False]

# %% [markdown]
# ## Sensitivity to LES Parameters

# %% [markdown]
# ## Failed Runs
#
# Some runs fail due to instability. The runs are more likely to fail when the resolution is increased, as the floe fields are less smooth. In order to resolve these fields, the timestep may need to be decreased. When attempting to increase the resolution, I had quite a few runs fail at $C_T' = 4$. In hindsight, running at this high of a $C_T'$ might not even be physical, so I decided to move on and work with more reasonable $C_T'$ values. However, here I cataloged the runs that became unstable and crashed.

# %%
df_nans = df[df.isna().any(axis=1)]
df_nans = df_nans.sort_values(by = ["nx", "dt", "filterFactor", "surge_freq", "surge_amplitude", "pitch_amplitude"])
df_nans[["CT_prime", "nx", "ny", "dt", "filterFactor", "filter", "surge_freq", "surge_amplitude", "pitch_amplitude", "mean_Cp"]]

# %% [markdown]
# We can see here that the runs with $n_x = 256$ that crashed all has a surge amplitude of $1$. This makes sense, as this basically brings the turbine to a standstill. Then, for $n_x = 512$, the runs will smaller surge frequencies and amplitude of 0.5 all crashed. To fix this, I tried all different filter factors (1.5 - 3.5), but this didn't help. Lowering the timestep artifically low for the higher frequency runs did help. This makes sense as this helps to resolve the more complex flow field. 

# %% [markdown]
# ### Plot the Unstable Surging Runs (all but one!)

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, dpi = 100, figsize = (10, 6), sharey = True, sharex = True)
fig.suptitle("Unstable Surging Runs")
ax0.set_title("$n_x = 256$")
ax0.set_xlabel('Frequency [St]', size = 12)
ax0.set_ylabel('$\\frac{\Delta}{h}$', size = 12)
ax0.tick_params(axis='both', which='major', labelsize=12)
ax1.set_title("$n_x = 512$")
ax1.set_xlabel('Frequency [St]', size = 12)
ax1.set_ylabel(' ', size = 12)
ax1.tick_params(axis='both', which='major', labelsize=12)

sns.scatterplot(ax = ax0, data = df_nans[(df_nans.surge_amplitude != 0) & (df_nans.nx == 256)], x = "surge_freq", y = "filterFactor", hue = "surge_amplitude", style = "rounded_dt", palette = "tab10", s = 50)
sns.scatterplot(ax = ax1, data = df_nans[(df_nans.surge_amplitude != 0) & (df_nans.nx == 512)], x = "surge_freq", y = "filterFactor", hue = "surge_amplitude", style = "rounded_dt", palette = "tab10", s = 50)

# %% [markdown]
# ## Completed Runs
#
# We now consider all of the rest of the runs that sucessfully completed.

# %%
df = df.dropna()
cols_to_keep = ["Movement", "nx", "filterFactor", "filterFactor", "useCorrection", "turbulence", "CT_prime", "surge_freq", "surge_amplitude", "pitch_amplitude"]
df = df.drop_duplicates(subset = cols_to_keep, keep = 'last')

# %% [markdown]
# We also create a few specialized datasets to help us plot specific subsets of the data, either by frequency and amplitude, or by LES parameters.

# %%
low_res_df = df[(df.ny == 128) & (df.filterFactor == 1.5)]
high_res_df = df[(df.ny == 256)]

# %% [markdown]
# ## Plot Mean $C_P$ LES Data for $f = 1$, $A_s = 0.5$, and $A_p = 5^\circ$ over $C_T'$ Values

# %%
max_motion_df = low_res_df[low_res_df["surge_freq"] == 1]
max_motion_df = max_motion_df[(max_motion_df["surge_amplitude"] == 0.5) | (max_motion_df["pitch_amplitude"] == 5)]

# %%
fig, (ax0, ax1) = plot_mean_cp_models(ctp_list, f, sa, pa);
surging_max_motion_df = max_motion_df[max_motion_df["Movement"] == "Surging"]
sns.scatterplot(ax  = ax0, data = surging_max_motion_df, x = "CT_prime", y = "mean_Cp", color = 'k', zorder = 5, s=50)
pitching_max_motion_df = max_motion_df[max_motion_df["Movement"] == "Pitching"]
sns.scatterplot(ax = ax1, data = pitching_max_motion_df, x = "CT_prime", y = "mean_Cp", color = 'k', zorder = 5, s=50, legend = False)
fix_plot_legend(ax0, xOffset = 2.2)

# %% [markdown]
# ## Calculate Differences Between UMM and LES

# %%
vals = df.apply((lambda row: umm(row.CT_prime, row.surge_freq, row.surge_amplitude, row.pitch_amplitude)), axis = 1)
means, stds, _, _ = zip(*vals)
df["umm_mean"] = np.ndarray.flatten(np.array(means))
df["percent_diff_umm"] = df.apply((lambda row: 100 * (row.umm_mean - row.mean_Cp) / row.mean_Cp), axis = 1)


# %%
def get_yaxis_type(yaxis_key):
    if yaxis_key == "mean_Cp_ground" or yaxis_key == "percent_diff_umm_mean" or yaxis_key == "mean_Cp" or yaxis_key == "percent_diff_umm":
        yaxis_type = "Mean"
    elif yaxis_key == "std_Cp_ground" or yaxis_key == "percent_diff_umm_std":
        yaxis_type = "STD"
    elif yaxis_key == "skewness_Cp_ground" or yaxis_key == "percent_diff_umm_skewness":
        yaxis_type = "Skewness"
    elif yaxis_key == "kurtosis_Cp_ground" or yaxis_key == "percent_diff_umm_kurtosis":
        yaxis_type = "Kurtosis"
    else:
        raise ValueError('Unknown yaxis_key!')
    return yaxis_type


# %%
def get_yaxis_umm_key(yaxis_key):
    if yaxis_key == "mean_Cp_ground" or yaxis_key == "mean_Cp":
        get_yaxis_umm_key = "umm_mean"
    elif yaxis_key == "std_Cp_ground":
        get_yaxis_umm_key = "umm_std"
    elif yaxis_key == "skewness_Cp_ground":
        get_yaxis_umm_key = "umm_skewness"
    elif yaxis_key == "kurtosis_Cp_ground":
        get_yaxis_umm_key = "umm_kurtosis"
    else:
        raise ValueError('Unknown yaxis_key!')
    return get_yaxis_umm_key


# %%
def plot_umm_les(df, mask = None, yaxis_key = "mean_Cp", style_key = None, subtitle = None, sharey = True):
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey = sharey, figsize = (8, 3), dpi = 300)
    yaxis_type = get_yaxis_type(yaxis_key)
    umm_yaxis_key = get_yaxis_umm_key(yaxis_key)
    title_str = yaxis_type + f" $C_P$ vs Amplitude for Varying Frequencies"
    title_y = 1.08
    if subtitle is not None:
        title_str += "\n" + subtitle
        title_y += 0.08
    fig.suptitle(title_str, size = 16, y = title_y)
    ax0.set_title("Surging", size = 14)
    ax1.set_title("Pitching", size = 14)
    ax0.set_xlabel('Amplitude [-]', size = 12)
    ax1.set_xlabel('Amplitude [$^\circ$]', size = 12)
    ax0.set_ylabel(yaxis_type + ' $C_P$', size = 12)
    ax1.set_ylabel(' ', size = 12)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    ax0_data = df[(df["Movement"] == "Surging")]
    ax1_data = df[(df["Movement"] == "Pitching")]
    if mask is not None:
        ax0_data = ax0_data[mask]
        ax1_data = ax1_data[mask]
    if style_key is None:
        style_key = "surge_freq"
    sns.scatterplot(ax = ax0, data = ax0_data, x = "surge_amplitude", y = yaxis_key, hue = "surge_freq", style = style_key, palette = palette, s=50)
    sns.lineplot(ax = ax0, data = ax0_data, x = "surge_amplitude", y = umm_yaxis_key, color = "k", label = "UMM")

    sns.scatterplot(ax = ax1, data = ax1_data, x = "pitch_amplitude", y = yaxis_key, hue = "surge_freq", style = style_key, palette = palette, s=50, legend = False)
    sns.lineplot(ax = ax1, data = ax1_data, x = "pitch_amplitude", y = umm_yaxis_key, color = "k")

    fix_plot_legend(ax0, xOffset = 1.7, title = "Frequency")
    hspace = 0.5
    if not sharey:
        hspace = 0.9
    fig.subplots_adjust(hspace = hspace)



# %%
def plot_umm_error(df, mask = None, yaxis_key = "percent_diff_umm", style_key = None, subtitle = None, sharey = True):
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey = sharey, figsize = (8, 3), dpi = 300)
    yaxis_type = get_yaxis_type(yaxis_key)
    title_str = "UMM " + yaxis_type + f" $C_P$ Error vs Amplitude for Varying Frequencies"
    title_y = 1.08
    if subtitle is not None:
        title_str += "\n" + subtitle
        title_y += 0.08
    fig.suptitle(title_str, size = 16, y = title_y)
    ax0.set_title("Surging", size = 14)
    ax1.set_title("Pitching", size = 14)
    ax0.set_xlabel('Amplitude [-]', size = 12)
    ax1.set_xlabel('Amplitude [$^\circ$]', size = 12)
    ax0.set_ylabel('UMM ' + yaxis_type + ' $C_P$ Error [%]', size = 12)
    ax1.set_ylabel(' ', size = 12)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    surge_mask = (df["Movement"] == "Surging")
    pitch_mask = (df["Movement"] == "Pitching")
    if mask is not None:
        surge_mask = surge_mask & mask
        pitch_mask = pitch_mask & mask
    ax0_data = df[surge_mask]
    ax1_data = df[pitch_mask]

    if style_key is None:
        style_key = "surge_freq"

    sns.scatterplot(ax = ax0, data = ax0_data, x = "surge_amplitude", y = yaxis_key, hue = "surge_freq", style = style_key, palette = palette, s=50)
    sns.scatterplot(ax = ax1, data = ax1_data, x = "pitch_amplitude", y = yaxis_key, hue = "surge_freq", style = style_key, palette = palette, s=50, legend = False)

    fix_plot_legend(ax0, xOffset = 1.7, title = "Frequency")
    hspace = 0.5
    if not sharey:
        hspace = 0.9
    fig.subplots_adjust(hspace = hspace)



# %% [markdown]
# ## Plot Mean $C_P$ from UMM for $C_T' = 1$ for a range of $f$ and $A$

# %%
ctp = 1
mask = (df.filterFactor == 1.5) & (df.ny == 128) & (df.surge_freq < 1.5) & (df.CT_prime == ctp)
plot_umm_les(df, mask = mask, subtitle= "$C_T' = 1$")
plot_umm_error(df, mask = mask, subtitle= "$C_T' = 1$")

# %% [markdown]
# ## Plot Mean $C_P$ from UMM for $C_T' = 4$ for a range of $f$ and $A$

# %% [markdown]
# I don't really trust the results of $C_T' = 4$. As can be seen below, the results change a lot based on the various LES parameters. I think $C_T' = 4$ is just too turbulent and may not even be reflective of any real phenomenon. Due to this, I decided to work with lower $C_T'$ values for the analysis after this section.

# %%
ctp = 4
mask = (df.filterFactor == 1.5) & (df.ny == 128) & (df.surge_freq < 1.5) & (df.CT_prime == ctp)
plot_umm_les(df, mask = mask, subtitle= "$C_T' = 4$")
plot_umm_error(df, mask = mask, subtitle= "$C_T' = 4$")

# %% [markdown]
# ## High $C_T'$ Run Sensitivity
#
# For high $C_T'$ values, small changes in LES parameters can lead to large changes in values. I decided to test this out for the above $C_T' = 4$ values as I wasn't sure I could trust them. This turned out to be the correct mindset, as the values differed a LOT based on the LES parameters. This, paired with my issues getting high resolution $C_T' = 4$ simulations to run, documented above, led me to give up on analyzing the $C_T' = 4$ case for now and focus on more realistic parameters.

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (8, 3), dpi = 300)
fig.suptitle("Surging $A = 0.5$ and $C_T' = 4$", size = 14)
ax0.set_xlabel('Frequency [-]', size = 12)
ax0.set_ylabel('Mean $C_P$', size = 12)
ax0.tick_params(axis='both', which='major', labelsize=12)

ax1.set_xlabel('Frequency [-]', size = 12)
ax1.set_ylabel('UMM $C_P$ Error [%]', size = 12)
ax1.tick_params(axis='both', which='major', labelsize=12)

ax0_data = df[(df.surge_amplitude == 0.5) & (df.CT_prime == 4)]

sns.scatterplot(ax = ax0, data = ax0_data, x = "surge_freq", y = "mean_Cp", hue = "nx", style = "rounded_dt", palette = "tab10", size = "filterFactor")
sns.lineplot(ax = ax0, data = ax0_data, x = "surge_freq", y = "umm_mean", color = "k", label = "UMM")

sns.scatterplot(ax = ax1, data = ax0_data, x = "surge_freq", y = "percent_diff_umm", hue = "nx", style = "rounded_dt", palette = "tab10", size = "filterFactor", legend = False)
fix_plot_legend(ax0)
fig.subplots_adjust(wspace=0.3)


# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (8, 3), dpi = 300)
fig.suptitle("Pitching $A = 10^\circ$ and $C_T' = 4$", size = 14)
ax0.set_xlabel('Frequency [-]', size = 12)
ax0.set_ylabel('Mean $C_P$', size = 12)
ax0.tick_params(axis='both', which='major', labelsize=12)

ax1.set_xlabel('Frequency [-]', size = 12)
ax1.set_ylabel('UMM $C_P$ Error [%]', size = 12)
ax1.tick_params(axis='both', which='major', labelsize=12)

ax0_data = df[(df.pitch_amplitude == 10) & (df.CT_prime == 4)]

sns.scatterplot(ax = ax0, data = ax0_data, x = "surge_freq", y = "mean_Cp", hue = "nx", style = "rounded_dt", palette = "tab10", size = "filterFactor")
sns.lineplot(ax = ax0, data = ax0_data, x = "surge_freq", y = "umm_mean", color = "k", label = "UMM")

sns.scatterplot(ax = ax1, data = ax0_data, x = "surge_freq", y = "percent_diff_umm", hue = "nx", style = "rounded_dt", palette = "tab10", size = "filterFactor", legend = False)
fix_plot_legend(ax0)
fig.subplots_adjust(wspace=0.3)

# %% [markdown]
# ## Analysis of $C_T' = 1.33, 1.66, 2.0$ Cases along various $f$ and $A$

# %% [markdown]
# At this point, I decided to run a new set of simulations at more "reasonable" values of $C_T'$. I also decided to up the resolution in the y and z directions after noticing streaks in the flow fields. I also upped the filter factor after noticing Gibb's phenomenon over the rotor. However, due to my increase in resoltuion, the absolutde width of the filter stayed about the same.
#
# I decided to work with even numbers of $f$ and $A$ and to push the pitch amplitude up a lot since nothing interesting was happening below $10^\circ$ anyways.
#
# I also updated the way that I was calculating $a_n$ and $C_P$. I now calcualte it in two frames of reference.
#
# In the **ground frame**, I add the movement of the turbine to the disk velocity. In this frame of reference, I still say that $u_\infty = 1$, but I say that $u_\text{rel} = u_t + u_d$. 
#
# I also then added in calculations in the **turbine frame of reference**. In this frame of reference, $u_\infty = 1 - u_t$ and $u_\text{rel} = u_d$. 
#
# Thus at any given moment, where $\phi$ is tilt:
#
# $a_n =  1 - \frac{u_\text{rel}}{u_\infty * \cos(\phi)}$
#
# $C_P = \frac{P}{\frac{1}{2} \rho \pi (\frac{D}{2})^2 u_\infty^3 \cos^3(\phi)}$
#
# We will compare the results per frame of reference below. Since I changed the names of the columns in the new dataset, I need to redefine the plotting code.

# %%
df_final = pd.read_csv("/Users/sky/src/HowlandLab/data/updated_moving_analysis_08_06.csv")
# add in rows for easier analysis
df_final['Movement'] = df_final.apply(lambda row: ("Stationary" if row.marker == "o" else ("Surging" if row.marker == "s" else "Pitching")), axis = 1)
df_final['rounded_dt'] = df_final.apply(lambda row: round(row["dt"], ndigits = 3), axis = 1)
vals = df_final.apply((lambda row: umm(row.CT_prime, row.surge_freq, row.surge_amplitude, row.pitch_amplitude)), axis = 1)
means, stds, skewness, kurtosis = zip(*vals)
df_final["umm_mean"] = np.ndarray.flatten(np.array(means))
df_final["umm_std"] = np.ndarray.flatten(np.array(stds))
df_final["umm_skewness"] = np.ndarray.flatten(np.array(skewness))
df_final["umm_kurtosis"] = np.ndarray.flatten(np.array(kurtosis))
# clean dataset
df_final = df_final.dropna()
df_final = df_final.drop_duplicates(subset = cols_to_keep, keep = 'last')

# %%
df_final["percent_diff_umm_mean"] = df_final.apply((lambda row: 100 * (row.umm_mean - row.mean_Cp_ground) / row.mean_Cp_ground), axis = 1)
df_final["percent_diff_umm_std"] = df_final.apply((lambda row: 100 * (row.umm_std - row.std_Cp_ground) / row.std_Cp_ground), axis = 1)
df_final["percent_diff_umm_skewness"] = df_final.apply((lambda row: 100 * (row.umm_skewness - row.skewness_Cp_ground) / row.skewness_Cp_ground), axis = 1)
df_final["percent_diff_umm_kurtosis"] = df_final.apply((lambda row: 100 * (row.umm_kurtosis - row.kurtosis_Cp_ground) / row.kurtosis_Cp_ground), axis = 1)

# %%
df_final

# %% [markdown]
# ## Mean Values

# %%
ctp = 1.33
mask = (df_final.CT_prime == ctp)
plot_umm_les(df_final, mask = mask, yaxis_key = "mean_Cp_ground", subtitle= "$C_T' = 1.33$")
plot_umm_error(df_final, mask = mask, yaxis_key= "percent_diff_umm_mean", subtitle= "$C_T' = 1.33$")

# %%
ctp = 2
mask = df_final.CT_prime == ctp
plot_umm_les(df_final, mask = mask, yaxis_key = "mean_Cp_ground", subtitle= "$C_T' = 2$")
plot_umm_error(df_final, mask = mask, yaxis_key= "percent_diff_umm_mean", subtitle= "$C_T' = 2$")

# %% [markdown]
# ## STD Values

# %%
ctp = 1.33
mask = (df_final.CT_prime == ctp)
plot_umm_les(df_final, mask = mask, yaxis_key = "std_Cp_ground", subtitle= "$C_T' = 1.33$", sharey = False)
plot_umm_error(df_final, mask = mask, yaxis_key= "percent_diff_umm_std", subtitle= "$C_T' = 1.33$", sharey = False)

# %%
ctp = 2
mask = (df_final.CT_prime == ctp)
plot_umm_les(df_final, mask = mask, yaxis_key = "std_Cp_ground", subtitle= "$C_T' = 2$", sharey = False)
plot_umm_error(df_final, mask = mask, yaxis_key= "percent_diff_umm_std", subtitle= "$C_T' = 2$", sharey = False)

# %% [markdown]
# # LES Data (with all timepoints)
#
# When analyzing the LES data, we must recall that $U_d$ as output by PadeOps, already includes the turbine motion (i.e. it is in the perspective of a stationary turbine). We also want to calculate $a_n$, $C_T$, and $C_P$ from the perspective of a stationary turbine.
#
# ## Read in Data

# %%
plt.rcParams["figure.dpi"] = 300

# %%
rho, uinf, D = 1, 1, 1

# %%
df_les = pd.read_csv("/Users/sky/src/HowlandLab/data/sim_16_all_runs_data_points.csv")
df_les = df_les.dropna()
df_les = df_les[((df_les["Thrust Coefficient"] != 1.66) & (df_les["Frequency"] > 0.1))] # (df_les["Frequency"] < 1)
df_les = df_les[(((df_les["Movement"] == "Pitch") & (df_les["Amplitude"] < 20)) | ((df_les["Movement"] == "Surge") & (df_les["Amplitude"] < 1)))]
df_les["Model"] = "LES"

# %%
df_les = df_les.rename(columns={'UDisk': 'UDisk_Turb', 'Thrust Coefficient': 'Local Thrust Coefficient'}) # disk velocity in the turbine frame of reference
df_les["UDisk_Ground"] = df_les["UDisk_Turb"] + df_les["UTurb"] # disk velocity in the ground frame of reference

# %%
df_les["UInf_Turb"] = (uinf - df_les["UTurb"]) * np.cos(df_les["Tilt"])
df_les["UInf_Ground"] = uinf * np.cos(df_les["Tilt"])


# %% [markdown]
# ## Calculate $a_n$, $C_T$ and $C_P$

# %%
def calc_an(df, ud_key, uinf_key):
    return 1 - (df[ud_key] / df[uinf_key])

def calc_ct(df, ud_key, uinf_key):
    return np.sign(df[ud_key]) * df["Local Thrust Coefficient"] * ((df[ud_key])**2 / (df[uinf_key])**2)

def calc_ctp(df, ud_key, uinf_key):
    return (df["ct_Turb"]) / ((df[ud_key])**2 / (df[uinf_key])**2)

def calc_cp(df, uinf_key = "UInf_Turb"):  # power in PadeOps and UMM is calculated in the turbine frame of reference!
    return df.Power / (0.5 * rho * math.pi * (D/2)**2 * (df[uinf_key])**3)


# %%
df_les["an_Turb"] = calc_an(df_les, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb")
df_les["an_Ground"] = calc_an(df_les, ud_key = "UDisk_Ground", uinf_key = "UInf_Ground")
df_les["an_Simple"] = calc_an(df_les, ud_key = "UDisk_Turb", uinf_key = "UInf_Ground")

LES_an_Turb = df_les["an_Turb"]
df_les["an_Turb_error"] = 0
df_les["an_Turb_abs_error"] = 0
df_les["an_Ground_error"] = 0

# %%
df_les["ct_Turb"] = calc_ct(df_les, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb")
df_les["ct_Ground"] = calc_ct(df_les, ud_key = "UDisk_Ground", uinf_key = "UInf_Ground")
df_les["ct_Simple"] = calc_ct(df_les, ud_key = "UDisk_Turb", uinf_key = "UInf_Ground")

LES_ct_Turb = df_les["ct_Turb"]
df_les["ct_Turb_error"] = 0
df_les["ct_Turb_abs_error"] = 0
df_les["ct_Ground_error"] = 0
df_les["ct_Simple_abs_error"] = 0
df_les["ct_Simple_error"] = 0

# %%
df_les["cp_Turb"] = calc_cp(df_les, uinf_key = "UInf_Turb")
df_les["cp_Ground"] = calc_cp(df_les, uinf_key = "UInf_Ground")
df_les["cp_Ground_abs_error"] = 0

LES_cp_Turb = df_les["cp_Turb"]
df_les["cp_Turb_error"] = 0


# %%
def plot_an_ct_vals(df, title):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharey = True, sharex = True, figsize = (10, 8))
    fig.suptitle(title, size = 18)

    ax0.set_title("$U_\infty$", size = 16)
    ax0.set_ylabel(" ")
    ax0.tick_params(axis='both', which='major', labelsize=12)
    sns.scatterplot(ax = ax0, data=df, x = "Time", y = "UTurb", label = "$u_{t, g}$", color = "grey")
    sns.scatterplot(ax = ax0, data=df, x = "Time", y = "UInf_Ground", label = "$U_{\infty, g}$", color = plt.cm.tab10(0))
    sns.scatterplot(ax = ax0, data=df, x = "Time", y = "UInf_Turb", label = "$U_{\infty, t} = U_{\infty, g} - u_{t, g}$", color = plt.cm.tab10(1))

    ax1.set_title("$u_d$", size = 16)
    ax1.set_ylabel(" ")
    ax1.tick_params(axis='both', which='major', labelsize=12)
    sns.scatterplot(ax = ax1, data=df, x = "Time", y = "UDisk_Ground", label = "$u_{d, g}$",  color = plt.cm.tab10(0))
    sns.scatterplot(ax = ax1, data=df, x = "Time", y = "UDisk_Turb", label = "$u_{d, t} = u_{d, g} - u_{t, g}$", color = plt.cm.tab10(1))

    ax2.set_title("$a_n$", size = 16)
    ax2.set_ylabel(" ")
    ax2.tick_params(axis='both', which='major', labelsize=12)
    sns.scatterplot(ax = ax2, data=df, x = "Time", y = "an_Simple", label = "$a_{n, s} = 1 - (u_{d, t} / U_{\infty, g})$", color = plt.cm.tab10(2))
    sns.scatterplot(ax = ax2, data=df, x = "Time", y = "an_Turb", label = "$a_{n, t} = 1 - (u_{d, t} / U_{\infty, t})$", color = plt.cm.tab10(1))
    sns.scatterplot(ax = ax2, data=df, x = "Time", y = "an_Ground", label = "$a_{n, g} = 1 - (u_{d, g} / U_{\infty, g})$", color = plt.cm.tab10(0))


    ax3.set_title("$C_T$", size = 16)
    ax3.set_ylabel(" ")
    ax3.tick_params(axis='both', which='major', labelsize=12)
    sns.scatterplot(ax = ax3, data=df, x = "Time", y = "ct_Simple", label = "$C_{T, s} = = C_T'(u_{d, t}^2 / U_{\infty, g}^2)$", color = plt.cm.tab10(2))
    sns.scatterplot(ax = ax3, data=df, x = "Time", y = "ct_Turb", label = "$C_{T, t} = C_T'(u_{d, t}^2 / U_{\infty, t}^2) $", color = plt.cm.tab10(1))
    sns.scatterplot(ax = ax3, data=df, x = "Time", y = "ct_Ground", label = "$C_{T, g} = C_T'(u_{d, g}^2 / U_{\infty, g}^2)$", color = plt.cm.tab10(0))

    fix_plot_legend(ax0, xOffset = -1.1, title = "$U_\infty$")
    fix_plot_legend(ax1, xOffset = 0.6, title = "$u_d$")
    fix_plot_legend(ax2, xOffset = -1.1, title = "$a_n$")
    fix_plot_legend(ax3, xOffset = 0.7, title = "$C_T$")

# %%
def_les_f2_A4 = df_les[(df_les["Local Thrust Coefficient"] == 1.33) & (df_les["Amplitude"] == 0.4) & (df_les["Frequency"] == 0.2)]
def_les_f2_A4 = def_les_f2_A4[def_les_f2_A4["Time"] > 290]
plot_an_ct_vals(def_les_f2_A4, "$a_n$ and $C_T$ Calculations for Surge with $C_T' = 1.33$, $f = 0.2$ and $A = 0.4$")


# %%
def plot_an_ct_pitch_vals(df, title):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharey = False, sharex = True, figsize = (10, 8))
    fig.suptitle(title, size = 18)
    
    ax0.set_ylim(0.95, 1.05)
    ax0.set_title("$U_\infty$", size = 16)
    ax0.set_ylabel(" ")
    ax0.tick_params(axis='both', which='major', labelsize=12)
    sns.scatterplot(ax = ax0, data=df, x = "Time", y = "UInf_Ground", label = "$U_{\infty, g}$", color = plt.cm.tab10(0))
    sns.scatterplot(ax = ax0, data=df, x = "Time", y = "UInf_Turb", label = "$U_{\infty, t} = U_{\infty, g} - u_{t, g}$", color = plt.cm.tab10(1))

    ax1.set_ylim(0.7, 0.8)
    ax1.set_title("$u_d$", size = 16)
    ax1.set_ylabel(" ")
    ax1.tick_params(axis='both', which='major', labelsize=12)
    sns.scatterplot(ax = ax1, data=df, x = "Time", y = "UDisk_Ground", label = "$u_{d, g}$")
    sns.scatterplot(ax = ax1, data=df, x = "Time", y = "UDisk_Turb", label = "$u_{d, t} = u_{d, g} - u_{t, g}$")

    ax2.set_ylim(0.23, 0.26)
    ax2.set_title("$a_n$", size = 16)
    ax2.set_ylabel(" ")
    ax2.tick_params(axis='both', which='major', labelsize=12)
    sns.scatterplot(ax = ax2, data=df, x = "Time", y = "an_Ground", label = "$a_{n, g} = 1 - (u_{d, g} / U_{\infty, g})$")
    sns.scatterplot(ax = ax2, data=df, x = "Time", y = "an_Turb", label = "$a_{n, t} = 1 - (u_{d, t} / U_{\infty, t})$")
    sns.scatterplot(ax = ax2, data=df, x = "Time", y = "an_Simple", label = "$a_{n, s} = 1 - (u_{d, t} / U_{\infty, g})$")

    ax3.set_ylim(0.75, 0.78)
    ax3.set_title("$C_T$", size = 16)
    ax3.set_ylabel(" ")
    ax3.tick_params(axis='both', which='major', labelsize=12)
    sns.scatterplot(ax = ax3, data=df, x = "Time", y = "ct_Ground", label = "$C_{T, g} = C_T'(u_{d, g}^2 / U_{\infty, g}^2)$")
    sns.scatterplot(ax = ax3, data=df, x = "Time", y = "ct_Turb", label = "$C_{T, t} = C_T'(u_{d, t}^2 / U_{\infty, t}^2) $")
    sns.scatterplot(ax = ax3, data=df, x = "Time", y = "ct_Simple", label = "$C_{T, s} = = C_T'(u_{d, t}^2 / U_{\infty, g}^2)$")

    fix_plot_legend(ax0, xOffset = -1.15, title = "$U_\infty$")
    fix_plot_legend(ax1, xOffset = 0.5, title = "$u_d$")
    fix_plot_legend(ax2, xOffset = -1.2, title = "$a_n$")
    fix_plot_legend(ax3, xOffset = 0.6, title = "$C_T$")


# %%
def_les_f2_A4 = df_les[(df_les["Local Thrust Coefficient"] == 1.33) & (df_les["Amplitude"] == 12) & (df_les["Frequency"] == 0.2)]
def_les_f2_A4 = def_les_f2_A4[def_les_f2_A4["Time"] > 290]
plot_an_ct_pitch_vals(def_les_f2_A4, "$a_n$ and $C_T$ Calculations for Pitch with $C_T' = 1.33$, $f = 0.2$ and $A = 12^\circ$")

# %% [markdown]
# From here, I decided to use the values that are calculated in the turbine frame of reference. 

# %% [markdown]
# ## Period-Average LES Data
#
# Calculate the phase that each data point is in using the frequency and time values.
#
# ### Correct Period Offset

# %%
period = 1 / df_les["Frequency"]
df_les["Phase_Time"] = (df_les["Time"] % period) / period


# %%
def first_pos2neg_zero_cross_phase(phase, uturb):
    """Return phase at first positive→negative crossing of uturb, else NaN."""
    uturb = np.asarray(uturb)
    phase = np.asarray(phase)

    # detect sign change from + to -
    mask = (uturb[:-1] > 0) & (uturb[1:] <= 0)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.nan
    
    i = idx[0]  # first crossing
    # linear interpolation for more accurate phase at zero
    u1, u2 = uturb[i], uturb[i+1]
    p1, p2 = phase[i], phase[i+1]

    if u2 == u1:  # avoid division by zero
        return p1
    frac = -u1 / (u2 - u1)   # fraction between i and i+1 where u crosses zero
    return p1 + frac * (p2 - p1)

def compute_phase_offsets(df):
    group_cols = ["Movement", "Frequency", "Amplitude",
                  "Local Thrust Coefficient", "Model"]

    results = []
    for keys, group in df.groupby(group_cols):
        if keys[0] == "Surge":
            vals, phase_cross_avg = group["UTurb"], 0.25
        else:
            vals, phase_cross_avg = group["Tilt"], 0.5
        phase_cross = first_pos2neg_zero_cross_phase(group["Phase_Time"], vals)
        if np.isnan(phase_cross):
            offset = np.nan
        else:
            offset = phase_cross_avg - phase_cross
        results.append(dict(zip(group_cols, keys),
                            phase_cross=phase_cross,
                            phase_offset=offset))
    return pd.DataFrame(results)

def apply_phase_offsets(df, offsets_df):
    group_cols = ["Movement", "Frequency", "Amplitude",
                  "Local Thrust Coefficient", "Model"]

    # Merge offsets into the original df
    df_with_offsets = df.merge(offsets_df, on=group_cols, how="left")

    # Apply correction (wrap result into [0,1] interval if needed)
    df_with_offsets["Phase"] = (
        df_with_offsets["Phase_Time"] + df_with_offsets["phase_offset"]
    ) % 1.0  # keep phase in [0,1)

    return df_with_offsets


# %%
offsets_df = compute_phase_offsets(df_les)
df_les = apply_phase_offsets(df_les, offsets_df)

# %%
surge_df_les = df_les[df_les.Movement == "Surge"]
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharey = True, sharex = True, figsize = (12, 6))
sns.lineplot(ax = ax0, data = surge_df_les[surge_df_les["Local Thrust Coefficient"] == 1.33], x = "Phase_Time", y = "UTurb", hue = "Amplitude", style = "Frequency", palette="Set2")
sns.lineplot(ax = ax1, data = surge_df_les[surge_df_les["Local Thrust Coefficient"] == 2], x = "Phase_Time", y = "UTurb", hue = "Amplitude", style = "Frequency", legend = False, palette="Set2")

sns.lineplot(ax = ax2, data = surge_df_les[surge_df_les["Local Thrust Coefficient"] == 1.33], x = "Phase", y = "UTurb", hue = "Amplitude", style = "Frequency", legend = False, palette="Set2")
sns.lineplot(ax = ax3, data = surge_df_les[surge_df_les["Local Thrust Coefficient"] == 2], x = "Phase", y = "UTurb", hue = "Amplitude", style = "Frequency", legend = False, palette="Set2")

fix_plot_legend(ax0, xOffset = 1.55)
fig.suptitle("Surging $u_{t, t}$ vs Phase Fraction", y = 1.02, fontsize=18)
fig.supylabel("$u_{t, t}$", fontsize=18)
fig.supxlabel("$t / T$", fontsize=18)

ax0.set_title("$C_T' = 1.33$", size = 16)
ax1.set_title("$C_T' = 2$",  size = 16)
ax0.set_ylabel("Uncorrected", size = 14)
ax2.set_ylabel("Corrected", size = 14)

# %%
specific_surge = surge_df_les[(surge_df_les.Frequency == 1) & (surge_df_les.Amplitude == 0.6) & (surge_df_les["Local Thrust Coefficient"] == 1.33)]
np.mean(specific_surge["cp_Ground"])

# %%
pitch_df_les = df_les[df_les.Movement == "Pitch"]
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharey = True, sharex = True, figsize = (12, 6))
sns.lineplot(ax = ax0, data = pitch_df_les[pitch_df_les["Local Thrust Coefficient"] == 1.33], x = "Phase_Time", y = "Tilt", hue = "Amplitude", style = "Frequency", palette="Set2")
sns.lineplot(ax = ax1, data = pitch_df_les[pitch_df_les["Local Thrust Coefficient"] == 2], x = "Phase_Time", y = "Tilt", hue = "Amplitude", style = "Frequency", legend = False, palette="Set2")

sns.lineplot(ax = ax2, data = pitch_df_les[pitch_df_les["Local Thrust Coefficient"] == 1.33], x = "Phase", y = "Tilt", hue = "Amplitude", style = "Frequency", legend = False, palette="Set2")
sns.lineplot(ax = ax3, data = pitch_df_les[pitch_df_les["Local Thrust Coefficient"] == 2], x = "Phase", y = "Tilt", hue = "Amplitude", style = "Frequency", legend = False, palette="Set2")

fix_plot_legend(ax0, xOffset = 1.75)
fig.suptitle("Pitching $\phi$ vs Phase Fraction", y = 1.02, fontsize=18)
fig.supylabel("$\phi$ [rads]", fontsize=18)
fig.supxlabel("$t / T$", fontsize=18)

ax0.set_title("$C_T' = 1.33$", size = 16)
ax1.set_title("$C_T' = 2$",  size = 16)
ax0.set_ylabel("Uncorrected", size = 14)
ax2.set_ylabel("Corrected", size = 14)

# %% [markdown]
# ### Fourier Transform of $a_n$ and $C_T$ data

# %%
fft_records = []
for (ct, move, freq, amp, model), group in df_les.groupby(["Local Thrust Coefficient", "Movement", "Frequency", "Amplitude", "Model"]):
    T = group["Time"].iloc[1] - group["Time"].iloc[0]  # sampling interval

    an_vals = group["an_Turb"].values
    ct_vals = group["ct_Turb"].values

    y_an = an_vals - np.mean(an_vals)
    y_ct = ct_vals - np.mean(ct_vals)
    N = len(y_an)

    yf_an = fft(y_an)
    yf_ct = fft(y_ct)

    N = len(y_an)
    xf = fftfreq(N, T)[:N//2] # positive freqs only

    power_an = 2.0 / N * np.abs(yf_an[0:N//2])
    power_ct = 2.0 / N * np.abs(yf_ct[0:N//2])

    temp_df = pd.DataFrame({
        "fft_freq": xf,
        "fft_power_an": power_an,
        "fft_power_ct": power_ct,
        "Local Thrust Coefficient": ct,
        "Movement": move,
        "Amplitude": amp,
        "Frequency": freq,
        "Model": model,
    })
    fft_records.append(temp_df)

fft_df = pd.concat(fft_records, ignore_index=True)
fft_surge_df = fft_df[fft_df["Movement"] == "Surge"]
fft_pitch_df = fft_df[fft_df["Movement"] == "Pitch"]


# %%
def get_hue_order(df):
    if (df["Movement"] == "Surge").all():
        hue_order=[0.8, 0.6, 0.4, 0.2]
    elif (df["Movement"] == "Pitch").all():
        hue_order=[16, 12, 8, 4]
    return hue_order


# %%
def plot_fft(df, y_key, title):
    g = sns.relplot(
        data=df,
        x="fft_freq",
        y=y_key,
        hue="Amplitude",
        hue_order = get_hue_order(df),
        style = "Model",
        col="Frequency",
        row = "Local Thrust Coefficient",
        kind="line",
        height=4,
        aspect=1.5,
        palette="Set2",
        facet_kws={"sharey": True, "sharex": False},
        errorbar= None,
        linewidth=3.5,
    )

    g.set_titles(row_template="$C_T'$ = {row_name}", col_template="Amplitude = {col_name}", size = 22)
    g.set_axis_labels("Frequency (Hz)", "FFT Power", fontsize = 20)
    g._legend.set_title("Model", prop={'size': 22}); [t.set_fontsize(20) for t in g._legend.get_texts()]
    g._legend.set_bbox_to_anchor((1.04, 0.5))  # (x, y) position relative to figure
    g._legend.set_loc("center left")
    [ax.tick_params(labelsize=22) for ax in g.axes.flat]
    g.figure.suptitle(title, fontsize=26, y = 1.02)
    plt.tight_layout()
    g.set(xlim=(0, 5))


# %%
plot_fft(fft_surge_df, "fft_power_an", "FFT of $a_{n, t}$ for Surge Motion")
plot_fft(fft_surge_df, "fft_power_ct", "FFT of $C_{T, t}$ for Surge Motion")

# %%
plot_fft(fft_pitch_df, "fft_power_an", "FFT of $a_{n, t}$ for Pitch Motion")
plot_fft(fft_pitch_df, "fft_power_ct", "FFT of $C_{T, t}$ for Pitch Motion")

# %% [markdown]
# ### Average over Rounded Phase Fraction
#
# The first thing to do is to round the phase values and then plot the values of $a_n$ and $C_T$ to see what the standard deviation of the points within the given phase "buckets". After noting that the standard deviation is minimal, I move forward by interpolating the points across equally spaced range of phase.

# %%
cols_to_group_by = ["Movement", "Frequency", "Amplitude", "Local Thrust Coefficient", "Model", "Phase"]
cols_to_calc = ["UTurb", "Tilt", "Power", "UInf_Turb", "UInf_Ground", "UDisk_Turb", "UDisk_Ground", "an_Turb", "an_Turb_error", "an_Turb_abs_error", "an_Ground", "an_Ground_error", "an_Simple", "ct_Turb", "ct_Turb_error", "ct_Turb_abs_error", "ct_Ground",  "ct_Ground_error", "ct_Simple", "ct_Simple_abs_error", "ct_Simple_error", "cp_Turb", "cp_Ground", "cp_Ground_abs_error"]
avg_phase_df_les = df_les[cols_to_group_by + cols_to_calc]
avg_phase_df_les["Rounded_Phase"] = np.round(avg_phase_df_les["Phase"], decimals = 10)
avg_phase_df_les["Rounded_Phase"] = np.where(avg_phase_df_les["Rounded_Phase"] == 0, 1.0, avg_phase_df_les["Rounded_Phase"])


# %%
def add_shading_4panel(df, xkey, axes):
    if (df["Movement"] == "Surge").all():
        upwind_uturb = df[(df["UTurb"] <= 0)]
        shade_label = '$U_{\infty, t} > U_{\infty, g}$ ($u_{t, g} < 0$)'
    elif (df["Movement"] == "Pitch").all():
        upwind_uturb = df[(df["Tilt"] >= 0)]
        shade_label = '$\phi > 0$'

    x1, x2 = np.min(upwind_uturb[xkey]), np.max(upwind_uturb[xkey])
    for ax in axes:
    # vertical lines at x1 and x2
        ax.axvline(x1, color='darkgrey', linestyle='--', linewidth=1)
        ax.axvline(x2, color='darkgrey', linestyle='--', linewidth=1)
        # shaded vertical region between x1 and x2
        ax.axvspan(x1, x2, color='lightgrey', alpha=0.3, label=shade_label)
        ax.tick_params(labelsize=12)
        ax.set_xlim(0, 1)



# %%
def plot_4panel_an_ct(df, xkey = "Rounded_Phase", errorbar = "sd", pre_title = "Surging", sharey = True, palette = "Set2", two_panel = False):
    n_rows = 1 if two_panel else 1
    fig_height = 6 if two_panel else 3
    fig, axes = plt.subplots(n_rows, 2, sharey = sharey, sharex = True, figsize = (12, fig_height))
    if not two_panel:
        ((ax0, ax1), (ax2, ax3)) = axes
    else:
        (ax0, ax1) = axes
    sns.lineplot(ax = ax0, data = df[df["Local Thrust Coefficient"] == 1.33], x = xkey, y = "an_Turb", hue = "Amplitude", style = "Frequency", errorbar=errorbar, palette = palette)
    sns.lineplot(ax = ax1, data = df[df["Local Thrust Coefficient"] == 2], x = xkey, y = "an_Turb", hue = "Amplitude", style = "Frequency", legend = False, errorbar=errorbar, palette = palette)
    if not two_panel:
        sns.lineplot(ax = ax2, data = df[df["Local Thrust Coefficient"] == 1.33], x = xkey, y = "ct_Turb", hue = "Amplitude", style = "Frequency", legend = False, errorbar=errorbar, palette = palette)
        sns.lineplot(ax = ax3, data = df[df["Local Thrust Coefficient"] == 2], x = xkey, y = "ct_Turb", hue = "Amplitude", style = "Frequency", legend = False, errorbar=errorbar, palette = palette)

    add_shading_4panel(df, xkey, (ax0, ax1, ax2, ax3))

    fix_plot_legend(ax0, xOffset = 2.25)
    fig.suptitle(pre_title + " $a_{n, t}$ and $C_T$ vs Phase Fraction", y = 1.02, fontsize=18)

    ax0.set_title("$C_T' = 1.33$", size = 16)
    ax1.set_title("$C_T' = 2$",  size = 16)
    ax0.set_ylabel("$a_{n, t}$", size = 14)

    if not two_panel:
        ax0.set_xlabel(" ")
        ax1.set_xlabel(" ")

        ax2.set_ylabel("$C_{T, t}$", size = 14)
        ax2.set_xlabel("$t / T$", size = 14)
        ax3.set_xlabel("$t / T$", size = 14)
    else:
        ax0.set_xlabel("$t / T$", size = 14)
        ax1.set_xlabel("$t / T$", size = 14)
    
    return fig, axes


# %%
# def add_shading(g, df, phase_key = "Phase"):
#     upwind_uturb = df[(df["UTurb"] < 0) & (df["Frequency"] == 0.2) & (df["Amplitude"] == 0.2) & (df["Local Thrust Coefficient"] == 2)]
#     x1, x2 = np.min(upwind_uturb[phase_key]), np.max(upwind_uturb[phase_key])
#     for frequency, ax in g.axes_dict.items():
#         # vertical lines at x1 and x2
#         ax.axvline(x1, color='darkgrey', linestyle='--', linewidth=1, alpha=0.7)
#         ax.axvline(x2, color='darkgrey', linestyle='--', linewidth=1, alpha=0.7)
#         # shaded vertical region between x1 and x2
#         ax.axvspan(x1, x2, color='lightgrey', alpha=0.2)
#     # Create new legend element
#     region_heading = Line2D([], [], color='none', label='Regions')
#     shade_element = Patch(facecolor='lightgrey', label='$U_{\infty, t} > U_{\infty, g}$ ($u_{t, g} < 0$)')
#     # Grab existing legend handles and labels
#     handles, labels = g.axes.flat[0].get_legend_handles_labels()
#     # Update the FacetGrid legend to include the custom elements
#     g._legend.remove()  # remove old legend
#     g.add_legend(handles=list(handles) + [region_heading, shade_element], loc="center left")

# %%
def plot_8panel_val(df, xkey, ykey, errorbar, pre_title, xkey_title, ykey_title, sharey = True, sharex = True, add_shading = True, hue_key = "Amplitude", palette="Set2"):
    if hue_key == "Amplitude":
        col_key = "Frequency"
    elif hue_key == "Frequency":
        col_key = "Amplitude"
    g = sns.relplot(
        data=df.sort_values(xkey),
        x=xkey,
        y=ykey,
        hue = hue_key,
        style = "Model",
        hue_order=get_hue_order(df),
        col=col_key,
        row = "Local Thrust Coefficient",
        kind="line",
        height=4,
        palette=palette,
        facet_kws={"sharey": sharey, "sharex": sharex},
        linewidth = 3,
        alpha = 0.6,
        errorbar = errorbar,
    )

    if add_shading:
        if (df["Movement"] == "Surge").all():
            upwind_uturb = df[(df["UTurb"] <= 0)]
            shade_label = '$U_{\infty, t} > U_{\infty, g}$ ($u_{t, g} < 0$)'
        elif (df["Movement"] == "Pitch").all():
            upwind_uturb = df[(df["Tilt"] > 0)]
            shade_label = '$\phi > 0$'

        x1, x2 = np.min(upwind_uturb[xkey]), np.max(upwind_uturb[xkey])

        for _, ax in g.axes_dict.items():
            # vertical lines at x1 and x2
            ax.axvline(x1, color='darkgrey', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(x2, color='darkgrey', linestyle='--', linewidth=1, alpha=0.7)
            # shaded vertical region between x1 and x2
            ax.axvspan(x1, x2, color='lightgrey', alpha=0.2)

        # Create new legend element
        region_heading = Line2D([], [], color='none', label='Regions')
        shade_element = Patch(facecolor='lightgrey', label=shade_label)
        # Grab existing legend handles and labels
        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        # Update the FacetGrid legend to include the custom elements
        g._legend.remove()  # remove old legend
        g.add_legend(handles=list(handles) + [region_heading, shade_element], loc="center left")

    g.figure.suptitle(pre_title + " Motion " + ykey_title + " vs " + xkey_title, fontsize=22, y = 1.02)
    g.set_titles(col_template= col_key + " = {col_name}", row_template="$C_T' = {row_name}$", size = 16)
    g.set_axis_labels(xkey_title, ykey_title, fontsize = 16)
    [ax.tick_params(labelsize=18) for ax in g.axes.flat]
    g._legend.set_title(" ", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
    g._legend.set_bbox_to_anchor((1, 0.5))  # (x, y) position relative to figure
    plt.tight_layout()
    fig.subplots_adjust(wspace= 2, hspace = 2)
    return g


# %%
surge_avg_phase_les = avg_phase_df_les[avg_phase_df_les["Movement"] == "Surge"]
fig, (ax0, ax1) = plt.subplots(2, 1, sharey = True, sharex = True)
sns.lineplot(ax = ax0, data = surge_avg_phase_les[surge_avg_phase_les["Local Thrust Coefficient"] == 1.33], x = "Rounded_Phase", y = "UDisk_Turb", hue = "Amplitude", style = "Frequency", palette = "Set2")
sns.lineplot(ax = ax1, data = surge_avg_phase_les[surge_avg_phase_les["Local Thrust Coefficient"] == 2], x = "Rounded_Phase", y = "UDisk_Turb", hue = "Amplitude", style = "Frequency", legend = False, palette = "Set2")
add_shading_4panel(surge_avg_phase_les, "Rounded_Phase", (ax0, ax1))

fix_plot_legend(ax0, xOffset = 1.75)
fig.suptitle("Surging $u_{d, t}$ vs Phase Fraction", y = 1.02, fontsize=18)

ax0.set_title("$C_T' = 1.33$", size = 16)
ax0.set_ylabel("$u_{d, t}$", size = 14)
ax0.set_xlabel("t / T")

ax1.set_title("$C_T' = 2$",  size = 16)
ax1.set_xlabel("t / T")

# %%
plot_4panel_an_ct(surge_avg_phase_les, xkey = "Rounded_Phase", errorbar = "sd", pre_title = "Surging", sharey = False)

# %%
pitch_avg_phase_les = avg_phase_df_les[avg_phase_df_les["Movement"] == "Pitch"]
plot_4panel_an_ct(pitch_avg_phase_les, xkey = "Rounded_Phase", errorbar = "sd", pre_title = "Pitch",sharey = "row")

# %%
avg_df_les = avg_phase_df_les.groupby(["Frequency", "Amplitude", "Model", "Movement", "Local Thrust Coefficient", "Rounded_Phase"], as_index=False).agg({
    "an_Turb": "mean",
    "an_Turb_error": "mean",
    "an_Turb_abs_error": "mean",
    "an_Simple": "mean",
    "ct_Turb": "mean",
    "ct_Turb_error": "mean",
    "ct_Turb_abs_error": "mean",
    "cp_Turb": "mean",
    "UDisk_Turb": "mean",
    "UInf_Turb": "mean",
    "an_Ground": "mean",
    "an_Ground_error": "mean",
    "ct_Ground": "mean",
    "ct_Ground_error": "mean",
    "cp_Ground": "mean",
    "ct_Simple": "mean",
    "ct_Simple_abs_error": "mean",
    "ct_Simple_error": "mean",
    "cp_Ground_abs_error": "mean",
    "UDisk_Ground": "mean",
    "UInf_Ground": "mean",
    "UTurb": "mean",
    "Tilt": "mean",
    "Power": "mean"
})
surge_avg_df_les = avg_df_les[avg_df_les["Movement"] == "Surge"]
pitch_avg_df_les = avg_df_les[avg_df_les["Movement"] == "Pitch"]

# %%
plot_4panel_an_ct(surge_avg_df_les, xkey = "Rounded_Phase", errorbar = "sd", pre_title = "Averaged Surging", sharey = "row")
plot_4panel_an_ct(pitch_avg_df_les, xkey = "Rounded_Phase", errorbar = "sd", pre_title = "Averaged Pitch", sharey = "row")

# %%
plot_8panel_val(surge_avg_df_les, "an_Turb", "ct_Turb", None, "Averaged Surging", "$a_n$", "$C_T$", add_shading = False)

# %%
plot_8panel_val(pitch_avg_df_les, "an_Turb", "ct_Turb", None, "Averaged Pitching", "$a_n$", "$C_T$", "row", True, add_shading = False)


# %% [markdown]
# # Existing Models

# %% [markdown]
# At this point, I am just going to work with $C_T' = 1.33$ for now.

# %% [markdown]
# ## Classical Momentum Theory

# %%
def copy_df(df, model_name):
    new_df = df.copy()
    new_df["Model"] = model_name
    # these must all be recalculated for a new model
    new_df["Power"] = np.nan
    new_df["UDisk_Turb"] = np.nan
    new_df["UDisk_Ground"] = np.nan
    new_df["an_Turb"] = np.nan
    new_df["an_Ground"] = np.nan
    new_df["an_Simple"] = np.nan
    new_df["an_Turb_error"] = np.nan
    new_df["an_Turb_abs_error"] = np.nan
    new_df["an_Ground_error"] = np.nan
    return new_df


# %%
avg_classical_df = copy_df(avg_df_les, "Classical Momentum")
avg_classical_df["an_Turb"] = avg_classical_df["Local Thrust Coefficient"] / (4 + avg_classical_df["Local Thrust Coefficient"])
avg_classical_df["ct_Turb"] = avg_classical_df["Local Thrust Coefficient"] * (1 - avg_classical_df["an_Turb"])**2

# %%
avg_classical_df["UDisk_Turb"] = (1 - avg_classical_df["an_Turb"])
avg_classical_df["U_Diff_Turb"] = avg_classical_df["UDisk_Turb"] - avg_classical_df["UInf_Turb"]

# %% [markdown]
# ## UMM Data
#
# Now that we have the values for the LES data, we can calcualte the UMM results using the same tilt and turbine velocity data, at the same times. We will calculate the UMM results both with the $C_T'$-based UMM and the $C_T$-based UMM. 

# %%
pre_calculated = False

def get_umm_values(df, model, filename, thrust_coeff_key = "Local Thrust Coefficient", pre_calculated = True):
    if not pre_calculated:
        umm_results = df.apply(lambda row:(model(row[thrust_coeff_key], yaw = 0.0, tilt = row["Tilt"])), axis = 1)
        with open(filename, 'wb') as file:
            pd.to_pickle(umm_results, file)
            print(f'UMM list successfully saved to "{filename}"')
    else:
        with open(filename, "rb") as file:
            umm_results = pickle.load(file)
    return umm_results


# %%
def calculate_umm_vals(new_df, umm_results, les_df, ctp_input = True):
    # calculate disk velocities
    output_an_vals = np.array([umm_sol.an for umm_sol in umm_results])
    output_ct_vals = np.array([umm_sol.Ct for umm_sol in umm_results])
    output_ctp_vals = np.array([umm_sol.Ctprime for umm_sol in umm_results])
    output_cp_vals = np.array([umm_sol.Cp for umm_sol in umm_results])
    new_df["UDisk_Turb"] = (1 - output_an_vals) * new_df["UInf_Turb"]
    new_df["UDisk_Ground"] = new_df["UDisk_Turb"] + new_df["UTurb"]
    # calculate induction values
    new_df["an_Turb"] = calc_an(new_df, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb")
    new_df["an_Ground"] = calc_an(new_df, ud_key = "UDisk_Ground", uinf_key = "UInf_Ground")
    new_df["an_Direct"] = output_an_vals
    new_df["an_Simple"] = calc_an(new_df, ud_key = "UDisk_Turb", uinf_key = "UInf_Ground")
    # calculate induction errors
    new_df["an_Turb_error"] = np.where(les_df["an_Turb"] != 0, (new_df["an_Turb"] - les_df["an_Turb"]) / les_df["an_Turb"] * 100, np.nan)
    new_df["an_Turb_abs_error"] = new_df["an_Turb"] - les_df["an_Turb"]
    new_df["an_Ground_error"] = np.where(les_df["an_Ground"] != 0, (new_df["an_Ground"] - les_df["an_Ground"]) / les_df["an_Ground"] * 100, np.nan)
    # calculate thrust coefficient values
    new_df["ct_Direct"] = output_ct_vals
    new_df["Local Thrust Coefficient Direct"] = output_ctp_vals
    if ctp_input:
        new_df["ct_Turb"] = calc_ct(new_df, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb")
        new_df["ct_Ground"] = calc_ct(new_df, ud_key = "UDisk_Ground", uinf_key = "UInf_Ground")
        new_df["ct_Simple"] = calc_ct(new_df, ud_key = "UDisk_Turb", uinf_key = "UInf_Ground")
        new_df["ct_Direct"] = output_ct_vals * (new_df["UInf_Turb"])**2
        new_df["ct_Turb_error"] = np.where(les_df["ct_Turb"] != 0, (new_df["ct_Turb"] - les_df["ct_Turb"]) / les_df["ct_Turb"] * 100, np.nan)
        new_df["ct_Turb_abs_error"] = (new_df["ct_Turb"] - les_df["ct_Turb"])
        new_df["ct_Ground_error"] = np.where(les_df["ct_Ground"] != 0, (new_df["ct_Ground"] - les_df["ct_Ground"]) / les_df["ct_Ground"] * 100, np.nan)
        new_df["ct_Simple_abs_error"] = (new_df["ct_Simple"] - les_df["ct_Simple"])
        new_df["ct_Simple_error"] = (new_df["ct_Simple"] - les_df["ct_Simple"]) / les_df["ct_Simple"] * 100
    else:
        new_df["Local Thrust Coefficient"] = calc_ctp(new_df, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb")
        new_df["ct_Turb"] = calc_ct(new_df, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb")
    # calculate power coefficent values
    new_df["Power"] = output_cp_vals * 0.5 * rho * np.pi * (D / 2)**2 * (new_df["UInf_Turb"])**3
    new_df["cp_Direct"] = output_cp_vals * (new_df["UInf_Turb"])**3
    new_df["cp_Turb"] = calc_cp(new_df, uinf_key = "UInf_Turb")
    new_df["cp_Ground"] = calc_cp(new_df, uinf_key = "UInf_Ground")
    new_df["cp_Ground_abs_error"] = (new_df["cp_Ground"] - les_df["cp_Ground"])
    return new_df



# %% [markdown]
# ### $C_T'$ Input UMM

# %%
df_umm_ctp = copy_df(avg_df_les, "UMM")
model_ctp = UnifiedMomentum()
file_name_ctp = "/Users/sky/src/HowlandLab/data/sim_16_all_runs_data_UMM_CTp.pkl"
umm_results_ctp = get_umm_values(df_umm_ctp, model_ctp, file_name_ctp, pre_calculated = pre_calculated, thrust_coeff_key = "Local Thrust Coefficient")

# %%
df_umm_ctp = calculate_umm_vals(df_umm_ctp, umm_results_ctp, avg_df_les, ctp_input = True)

# %%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey = False, figsize = (20, 4))
sns.scatterplot(ax = ax1, data = df_umm_ctp[df_umm_ctp.Movement == "Surge"], x = "Rounded_Phase", y = "cp_Direct", hue = "Amplitude", style = "Local Thrust Coefficient", legend = False)
sns.scatterplot(ax = ax2, data = df_umm_ctp[df_umm_ctp.Movement == "Surge"], x = "Rounded_Phase", y = "cp_Ground", hue = "Amplitude", style = "Local Thrust Coefficient", legend = False)
sns.scatterplot(ax = ax3, data = avg_df_les[avg_df_les.Movement == "Surge"], x = "Rounded_Phase", y = "cp_Ground", hue = "Amplitude", style = "Local Thrust Coefficient", legend = False)
sns.scatterplot(ax = ax4, data = df_umm_ctp[df_umm_ctp.Movement == "Surge"], x = "Rounded_Phase", y = "cp_Ground_abs_error", hue = "Amplitude", style = "Local Thrust Coefficient", legend = False)

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (16, 4))
sns.scatterplot(ax = ax1, data = df_umm_ctp[df_umm_ctp.Movement == "Surge"], x = "Rounded_Phase", y = "ct_Direct", hue = "Amplitude", style = "Local Thrust Coefficient", legend = False)
sns.scatterplot(ax = ax2, data = df_umm_ctp[df_umm_ctp.Movement == "Surge"], x = "Rounded_Phase", y = "ct_Simple", hue = "Amplitude", style = "Local Thrust Coefficient", legend = False)
sns.scatterplot(ax = ax3, data = avg_df_les[avg_df_les.Movement == "Surge"], x = "Rounded_Phase", y = "ct_Simple", hue = "Amplitude", style = "Local Thrust Coefficient", legend = False)

# %%
umm_stationary_low_ctp = model_ctp(1.33, yaw = 0.0, tilt = 0.0)
umm_stationary_low_ctp_udisk = (1 - umm_stationary_low_ctp.an)

# %% [markdown]
# ### $C_T$ Input UMM

# %%
low_ctp_les = avg_df_les[avg_df_les["Local Thrust Coefficient"] == 1.33]
high_ctp_les = avg_df_les[avg_df_les["Local Thrust Coefficient"] == 2]

# %%
df_umm_ct_low_ctp = copy_df(low_ctp_les, "UMM - $C_T$ Input ($C_T' = 1.33$ LES)")
df_umm_ct_high_ctp = copy_df(high_ctp_les, "UMM - $C_T$ Input ($C_T' = 2$ LES)")

# %%
model_ct = ThrustBasedUnified()
file_name_ct = "/Users/sky/src/HowlandLab/data/sim_16_all_runs_data_UMM_CT.pkl"
low_ctp_umm_results_ct = get_umm_values(df_umm_ct_low_ctp, model_ct, file_name_ct, pre_calculated = pre_calculated, thrust_coeff_key = "ct_Turb")
high_ctp_umm_results_ct = get_umm_values(df_umm_ct_high_ctp, model_ct, file_name_ct, pre_calculated = pre_calculated, thrust_coeff_key = "ct_Turb")

# %%
low_ctp_df_umm_ct = calculate_umm_vals(df_umm_ct_low_ctp, low_ctp_umm_results_ct, low_ctp_les, ctp_input = False)
high_ctp_df_umm_ct = calculate_umm_vals(df_umm_ct_high_ctp, high_ctp_umm_results_ct, high_ctp_les, ctp_input = False)

# %% [markdown]
# ## Plot Various Models

# %%
fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(2, 4, sharey = True, sharex = True, figsize = (10, 6))
fig.suptitle("Comparison of $a_n$, $C_T$ and $C_T'$ Calculations for Surge with $C_T' = 1.33$")
alpha = 0.5

mask_movement = (avg_df_les.Movement == "Surge") & (avg_df_les.Frequency == 0.8) & (avg_df_les.Amplitude == 0.8)
mask_ctp = avg_df_les["Local Thrust Coefficient"] == 1.33
les_small_df = avg_df_les[mask_movement & mask_ctp]
umm_ctp_small_df = df_umm_ctp[mask_movement & mask_ctp]
umm_ct_small_df = low_ctp_df_umm_ct[mask_movement]
classical_small_df = avg_classical_df[mask_movement & mask_ctp]

ax0.set_title("LES $C_T'$ Input")
ax0.set_ylabel(" ")
sns.lineplot(ax = ax0, data=les_small_df, x = "Rounded_Phase", y = "UTurb", alpha = alpha, label = "$u_{t, g}$")
sns.lineplot(ax = ax0, data=les_small_df, x = "Rounded_Phase", y = "UInf_Turb", alpha = alpha,  label = "$U_{\infty, t} = U_{\infty, g} - u_{t, g} $")
sns.lineplot(ax = ax0, data=les_small_df, x = "Rounded_Phase", y = "UDisk_Turb", alpha = alpha,  label = "$u_{d, t} = u_{d, g} - u_{t, g}$")
ax1.set_title("UMM $C_T'$ Input")
ax1.set_ylabel(" ")
sns.lineplot(ax = ax1, data=umm_ctp_small_df, x = "Rounded_Phase", y = "UTurb", alpha = alpha)
sns.lineplot(ax = ax1, data=umm_ctp_small_df, x = "Rounded_Phase", y = "UInf_Turb", alpha = alpha)
sns.lineplot(ax = ax1, data=umm_ctp_small_df, x = "Rounded_Phase", y = "UDisk_Turb", alpha = alpha)
ax2.set_title("UMM $C_T$ Input")
ax2.set_ylabel(" ")
sns.lineplot(ax = ax2, data=umm_ct_small_df, x = "Rounded_Phase", y = "UTurb", alpha = alpha)
sns.lineplot(ax = ax2, data=umm_ct_small_df, x = "Rounded_Phase", y = "UInf_Turb", alpha = alpha)
sns.lineplot(ax = ax2, data=umm_ct_small_df, x = "Rounded_Phase", y = "UDisk_Turb", alpha = alpha)

ax3.set_title("Classical Momentum")
ax3.set_ylabel(" ")

# plot LES
sns.lineplot(ax = ax4, data=les_small_df, x = "Rounded_Phase", y = "an_Turb", alpha = alpha)
sns.lineplot(ax = ax4, data=les_small_df, x = "Rounded_Phase", y = "ct_Turb")
sns.lineplot(ax = ax4, data=les_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient")
# plot UMM CT' Input
sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "an_Turb", alpha = alpha, label = "$a_{n, t} = 1 - (u_{d, t} / U_{\infty, t})$")
sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "ct_Turb", alpha = alpha, label = "$C_{T, t} = C_T' (u_{d, t} / U_{\infty, t})^2 $")
sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient",  alpha = alpha, label = "$C_{T, t}' = C_{T, t} / (u_{d, t} / U_{\infty, t})^2$")

sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "an_Direct", alpha = alpha, linestyle = "--", label = "$a_{n, \\text{direct}}$")
sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "ct_Direct", alpha = alpha, linestyle = "--", label = "$C_{T, \\text{direct}}$")
sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient Direct",  alpha = alpha, linestyle = "--", label = "$C_{T, \\text{direct}}'$")
# plot UMM CT Input
sns.lineplot(ax = ax6, data=umm_ct_small_df, x = "Rounded_Phase", y = "an_Turb", alpha = alpha)
sns.lineplot(ax = ax6, data=umm_ct_small_df, x = "Rounded_Phase", y = "ct_Turb", alpha = alpha)
sns.lineplot(ax = ax6, data=umm_ct_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient", alpha = alpha)

sns.lineplot(ax = ax6, data=umm_ct_small_df, x = "Rounded_Phase", y = "an_Direct", alpha = alpha, linestyle = "--")
sns.lineplot(ax = ax6, data=umm_ct_small_df, x = "Rounded_Phase", y = "ct_Direct", alpha = alpha, linestyle = "--")
sns.lineplot(ax = ax6, data=umm_ct_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient Direct",  alpha = alpha, linestyle = "--")
# Plot Classical Momentum Theory
sns.lineplot(ax = ax7, data=classical_small_df, x = "Rounded_Phase", y = "an_Turb", alpha = alpha)
sns.lineplot(ax = ax7, data=classical_small_df, x = "Rounded_Phase", y = "ct_Turb", alpha = alpha)
sns.lineplot(ax = ax7, data=classical_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient", alpha = alpha)

# add legends
fix_plot_legend(ax0, xOffset = 4.75, title = " ")
fix_plot_legend(ax5, xOffset = 3.75, title = " ")

# %%
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex = True, figsize = (6, 8))
mask_movement = (avg_df_les.Movement == "Surge") & (avg_df_les.Frequency == 0.8) & (avg_df_les.Amplitude == 0.8)
mask_ctp = avg_df_les["Local Thrust Coefficient"] == 2
les_small_df = avg_df_les[mask_movement & mask_ctp]
umm_ctp_small_df = df_umm_ctp[mask_movement & mask_ctp]

sns.lineplot(ax = ax0, data=les_small_df, x = "Rounded_Phase", y = "ct_Turb", alpha = alpha, label = "LES: $C_{T, t}$")
sns.lineplot(ax = ax0, data=umm_ctp_small_df, x = "Rounded_Phase", y = "ct_Turb", alpha = alpha, label = "UMM: $C_{T, t}$")

sns.lineplot(ax = ax1, data=les_small_df, x = "Rounded_Phase", y = "ct_Ground", alpha = alpha, label = "LES: $C_{T, g}$")
sns.lineplot(ax = ax1, data=umm_ctp_small_df, x = "Rounded_Phase", y = "ct_Ground", alpha = alpha, label = "UMM: $C_{T, g}$")

sns.lineplot(ax = ax2, data=les_small_df, x = "Rounded_Phase", y = "ct_Simple", alpha = alpha, label = "LES: $C_{T, s}$")
sns.lineplot(ax = ax2, data=umm_ctp_small_df, x = "Rounded_Phase", y = "ct_Simple", alpha = alpha, label = "UMM: $C_{T, s}$")

fix_plot_legend(ax0, xOffset = 0.5, title = " ")
fix_plot_legend(ax1, xOffset = 0.5, title = " ")
fix_plot_legend(ax2, xOffset = 0.5, title = " ")
ax0.set_yticks(np.linspace(0, 2.5, num = 6))


# %%
fig, ((ax0, ax1, ax2), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3, sharey = True, sharex = True, figsize = (20, 12))
fig.suptitle("Comparison of $a_n$ and $C_T$ Calculations for Surge with $C_T' = 1.33$, $A = 0.4$, $f = 0.4$")
alpha = 0.5

mask_movement = (avg_df_les.Movement == "Surge") & (avg_df_les.Frequency == 0.4) & (avg_df_les.Amplitude == 0.4)
mask_ctp = avg_df_les["Local Thrust Coefficient"] == 1.33
les_small_df = avg_df_les[mask_movement & mask_ctp]
umm_ctp_small_df = df_umm_ctp[mask_movement & mask_ctp]
classical_small_df = avg_classical_df[mask_movement & mask_ctp]

ax0.set_title("LES $C_T'$ Input")
ax0.set_ylabel(" ")
sns.lineplot(ax = ax0, data=les_small_df, x = "Rounded_Phase", y = "UTurb", alpha = alpha, label = "$u_{t, g}$")
sns.lineplot(ax = ax0, data=les_small_df, x = "Rounded_Phase", y = "UInf_Turb", alpha = alpha,  label = "$U_{\infty, t} = U_{\infty, g} - u_{t, g} $")
sns.lineplot(ax = ax0, data=les_small_df, x = "Rounded_Phase", y = "UDisk_Turb", alpha = alpha,  label = "$u_{d, t} = u_{d, g} - u_{t, g}$")
ax1.set_title("UMM $C_T'$ Input")
ax1.set_ylabel(" ")
sns.lineplot(ax = ax1, data=umm_ctp_small_df, x = "Rounded_Phase", y = "UTurb", alpha = alpha)
sns.lineplot(ax = ax1, data=umm_ctp_small_df, x = "Rounded_Phase", y = "UInf_Turb", alpha = alpha)
sns.lineplot(ax = ax1, data=umm_ctp_small_df, x = "Rounded_Phase", y = "UDisk_Turb", alpha = alpha)
ax2.set_title("Classical")
ax2.set_ylabel(" ")

# TURBINE FRAME
# plot LES
ax4.set_ylabel(" ")
sns.lineplot(ax = ax4, data=les_small_df, x = "Rounded_Phase", y = "an_Turb", alpha = alpha)
sns.lineplot(ax = ax4, data=les_small_df, x = "Rounded_Phase", y = "ct_Turb")
sns.lineplot(ax = ax4, data=les_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient")
# plot UMM CT' Input
sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "an_Turb", alpha = alpha, label = "$a_{n, t} = 1 - (u_{d, t} / U_{\infty, t})$")
sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "ct_Turb", alpha = alpha, label = "$C_{T, t} = C_T' (u_{d, t} / U_{\infty, t})^2 $")
sns.lineplot(ax = ax5, data=umm_ctp_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient",  alpha = alpha, label = "$C_{T, t}' = C_{T, t} / (u_{d, t} / U_{\infty, t})^2$")

# Plot Classical Momentum Theory
sns.lineplot(ax = ax6, data=classical_small_df, x = "Rounded_Phase", y = "an_Turb", alpha = alpha)
sns.lineplot(ax = ax6, data=classical_small_df, x = "Rounded_Phase", y = "ct_Turb", alpha = alpha)
sns.lineplot(ax = ax6, data=classical_small_df, x = "Rounded_Phase", y = "Local Thrust Coefficient", alpha = alpha)

# GROUND FRAME
ax7.set_ylabel(" ")
ax7.set_xlabel(" ")
ax8.set_xlabel(" ")
sns.lineplot(ax = ax7, data=les_small_df, x = "Rounded_Phase", y = "an_Ground", alpha = alpha)
sns.lineplot(ax = ax7, data=les_small_df, x = "Rounded_Phase", y = "ct_Ground")
# plot UMM CT' Input
sns.lineplot(ax = ax8, data=umm_ctp_small_df, x = "Rounded_Phase", y = "an_Ground", alpha = alpha, label = "$a_{n, g} = 1 - (u_{d, g} / U_{\infty, g})$")
sns.lineplot(ax = ax8, data=umm_ctp_small_df, x = "Rounded_Phase", y = "ct_Ground", alpha = alpha, label = "$C_{T, g} = C_T' (u_{d, g} / U_{\infty, g})^2 $")

# SIMPLE FRAME
# sns.lineplot(ax = ax9, data=les_small_df, x = "Rounded_Phase", y = "an_Simple", alpha = alpha)
sns.lineplot(ax = ax10, data=les_small_df, x = "Rounded_Phase", y = "ct_Simple")

# sns.lineplot(ax = ax10, data=umm_ctp_small_df, x = "Rounded_Phase", y = "an_Simple", alpha = alpha)
sns.lineplot(ax = ax11, data=umm_ctp_small_df, x = "Rounded_Phase", y = "ct_Simple")

# POWER
sns.lineplot(ax = ax13, data=les_small_df, x = "Rounded_Phase", y = "cp_Turb", alpha = alpha)
sns.lineplot(ax = ax14, data=umm_ctp_small_df, x = "Rounded_Phase", y = "cp_Turb")

# add legends
fix_plot_legend(ax0, xOffset = 3.8, title = " ")
fix_plot_legend(ax5, xOffset = 2.75, title = " ")
fix_plot_legend(ax8, xOffset = 2.75, title = " ")
# fig.subplots_adjust(wspace=0.4, hspace = 1)

# %%
avg_df_combo_ctp = pd.concat([avg_df_les, df_umm_ctp], ignore_index=True)
#combine first and last name column into new column, with space in between 
avg_df_combo_ctp['Model_Frequency'] = avg_df_combo_ctp['Model'].astype(str) + ': ' + avg_df_combo_ctp['Frequency'].astype(str)

# %%
surge_avg_df_combo_ctp = avg_df_combo_ctp[avg_df_combo_ctp.Movement == "Surge"]
pitch_avg_df_combo_ctp = avg_df_combo_ctp[avg_df_combo_ctp.Movement == "Pitch"]

# %% [markdown]
# ## Plot Period Mean Values
#
# Before digging into the structure, we want to plot the mean values of the dataframe over the period, rather than over the phase to start with. Here, I won't even have the box plot take into account the variation over the phase. This is *just* the mean.

# %%
palette = "GnBu"
cmap = sns.color_palette("Blues", as_cmap=True)
# Sample 4 darker shades (from 0.4 to 1.0)
dark_blues = [cmap(x) for x in [0.2, 0.4, 0.6, 0.8, 1.0]]

cmap = sns.color_palette("Oranges", as_cmap=True)
# Sample 4 darker shades (from 0.4 to 1.0)
dark_oranges = [cmap(x) for x in [0.2, 0.4, 0.6, 0.8, 1.0]]
error_palette = dark_blues
accent_line_color = "darkorange"
# sns.set_context("notebook") 
sns.set_theme(
    style="white",
    rc={
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.title_fontsize": 18,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
        "legend.labelspacing": 1,
    },
)

# %%
specific_surge = avg_df_combo_ctp[(avg_df_combo_ctp.Frequency == 1) & (avg_df_combo_ctp.Amplitude == 0.6) & (avg_df_combo_ctp.Model.str.startswith("LES")) & (avg_df_combo_ctp.Movement == "Surge") & (avg_df_combo_ctp["Local Thrust Coefficient"] == 1.33)]
np.mean(specific_surge["cp_Ground"])

# %%
avg_period_df_les = avg_df_combo_ctp.groupby(["Frequency", "Amplitude", "Model", 'Model_Frequency', "Movement", "Local Thrust Coefficient"], as_index=False).agg({
    "an_Turb": "mean",
    "an_Turb_error": "mean",
    "an_Turb_abs_error": "mean",
    "ct_Turb": "mean",
    "ct_Turb_error": "mean",
    "ct_Turb_abs_error": "mean",
    "cp_Turb": "mean",
    "UDisk_Turb": "mean",
    "UInf_Turb": "mean",
    "an_Ground": "mean",
    "an_Ground_error": "mean",
    "ct_Ground": "mean",
    "ct_Ground_error": "mean",
    "cp_Ground": "mean",
    "cp_Ground_abs_error": "mean",
    "an_Simple": "mean",
    "ct_Simple": "mean",
    "ct_Simple_abs_error": "mean",
    "ct_Simple_error": "mean",
    "UDisk_Ground": "mean",
    "UInf_Ground": "mean",
    "UTurb": "mean",
    "Tilt": "mean",
    "Power": "mean"
}).sort_values(by=["Model", "Frequency"], ascending = [False, True])
avg_period_df_les = avg_period_df_les.dropna()
avg_period_df_les = avg_period_df_les[avg_period_df_les.an_Turb < np.inf]
surge_avg_period_df = avg_period_df_les[avg_period_df_les["Movement"] == "Surge"]
pitch_avg_period_df = avg_period_df_les[avg_period_df_les["Movement"] == "Pitch"]

# %%
g = sns.catplot(
    data=surge_avg_period_df[surge_avg_period_df.Model.str.startswith("LES")],
    x="Amplitude",
    y="an_Turb",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

(ax1, ax2) = g.axes.flat
avg_val1 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 1.33)].an_Turb)
avg_val2 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 2)].an_Turb)
ax1.axhline(avg_val1, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot
ax2.axhline(avg_val2, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot

umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequencies")

g.figure.suptitle('Surging Turbine: Average $a_{n, t}$ vs Amplitude')

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$a_{n, t}$")
g._legend.set_bbox_to_anchor((1.0, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()

# %%
g = sns.catplot(
    data=surge_avg_period_df[surge_avg_period_df.Model.str.startswith("LES")],
    x="Amplitude",
    y="ct_Turb",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

(ax1, ax2) = g.axes.flat
avg_val1 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 1.33)].ct_Turb)
avg_val2 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 2)].ct_Turb)
ax1.axhline(avg_val1, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot
ax2.axhline(avg_val2, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot

umm_line = Line2D([], [], color=accent_line_color, label='UMM', linewidth = 2.5, ls="--")
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequency $(f)$")

g.figure.suptitle('Surging Turbine')

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_S$)", "Mean $C_{T, t}$")
g._legend.set_bbox_to_anchor((1.0, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()

# %%
g = sns.catplot(
    data=surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("LES"))],
    x="Amplitude",
    y="ct_Ground",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

g.figure.suptitle('Pitching Turbine')

for ctp, ax in g.axes_dict.items():
    sub_data = surge_avg_period_df[(surge_avg_period_df["Local Thrust Coefficient"] == ctp)& (surge_avg_period_df.Model.str.startswith("UMM"))]
    sns.stripplot(ax = ax, data = sub_data, x = "Amplitude", y = "ct_Ground",  hue="Frequency", dodge = True, palette=error_palette, edgecolor=accent_line_color, linewidth=2, marker = "v", s =8, legend = False)

umm_line = Line2D([], [], color=None, marker = "v", label='UMM', markerfacecolor="white", markeredgecolor = accent_line_color, linewidth=0, markeredgewidth=3, markersize=12)
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequency $(f)$")

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_S$)", "Mean $C_{T, g}$")
g._legend.set_bbox_to_anchor((1.0, 0.5))
g._legend.set_loc("center left")
plt.tight_layout()

# %%
g = sns.catplot(
    data=surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("LES")) & (surge_avg_period_df.Amplitude < 1)],
    x="Amplitude",
    y="ct_Simple",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

g.figure.suptitle('Surging Turbine')

for ctp, ax in g.axes_dict.items():
    sub_data = surge_avg_period_df[(surge_avg_period_df["Local Thrust Coefficient"] == ctp)& (surge_avg_period_df.Model.str.startswith("UMM") & (surge_avg_period_df.Amplitude < 1))]
    sns.stripplot(ax = ax, data = sub_data, x = "Amplitude", y = "ct_Simple",  hue="Frequency", dodge = True, palette=error_palette, edgecolor=accent_line_color, linewidth=2, marker = "v", s =8, legend = False)

umm_line = Line2D([], [], color=None, marker = "v", label='UMM', markerfacecolor="white", markeredgecolor = accent_line_color, linewidth=0, markeredgewidth=3, markersize=12)
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequency $(f)$")

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_S$)", "Mean $C_{T, s}$")
g._legend.set_bbox_to_anchor((1.0, 0.5))
g._legend.set_loc("center left")
plt.tight_layout()

# %%
surge_avg_period_df[(surge_avg_period_df.Amplitude == 0.6) & (surge_avg_period_df.Frequency == 1.0)][["cp_Ground"]]

# %%
g = sns.catplot(
    data=surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("LES")) & (surge_avg_period_df.Amplitude < 1)],
    x="Amplitude",
    y="cp_Ground",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

g.figure.suptitle('Surging Turbine')

for ctp, ax in g.axes_dict.items():
    sub_data = surge_avg_period_df[(surge_avg_period_df["Local Thrust Coefficient"] == ctp)& (surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df.Amplitude < 1)]
    sns.stripplot(ax = ax, data = sub_data, x = "Amplitude", y = "cp_Ground",  hue="Frequency", dodge = True, palette=error_palette, edgecolor=accent_line_color, linewidth=2, marker = "v", s =8, legend = False)

umm_line = Line2D([], [], color=None, marker = "v", label='UMM', markerfacecolor="white", markeredgecolor = accent_line_color, linewidth=0, markeredgewidth=3, markersize=12)
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequency $(f)$")

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_S$)", "Mean $C_{P, g}$")
g._legend.set_bbox_to_anchor((1.0, 0.5))
g._legend.set_loc("center left")
plt.tight_layout()

# %%
g = sns.catplot(
    data=surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("LES"))],
    x="Amplitude",
    y="UDisk_Ground",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

g.figure.suptitle('Pitching Turbine')

for ctp, ax in g.axes_dict.items():
    sub_data = surge_avg_period_df[(surge_avg_period_df["Local Thrust Coefficient"] == ctp)& (surge_avg_period_df.Model.str.startswith("UMM"))]
    sns.stripplot(ax = ax, data = sub_data, x = "Amplitude", y = "UDisk_Ground",  hue="Frequency", dodge = True, palette=error_palette, edgecolor=accent_line_color, linewidth=2, marker = "v", s =8, legend = False)

umm_line = Line2D([], [], color=None, marker = "v", label='UMM', markerfacecolor="white", markeredgecolor = accent_line_color, linewidth=0, markeredgewidth=3, markersize=12)
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequency $(f)$")

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_S$)", "Mean $u_d$")
g._legend.set_bbox_to_anchor((1.0, 0.5))
g._legend.set_loc("center left")
plt.tight_layout()

# %%
g = sns.catplot(
    data=surge_avg_period_df[surge_avg_period_df.Model.str.startswith("UMM")],
    x="Amplitude",
    y="an_Turb_error",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

g.figure.suptitle('Surging Turbine: Average $a_{n, t}$ Error vs Amplitude')

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$a_{n, t}$ Error [%]")
g._legend.set_bbox_to_anchor((1, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()

# %%
smaller_pitch_data = pitch_avg_period_df[(pitch_avg_period_df.Model.str.startswith("UMM")) & (pitch_avg_period_df.Amplitude < 20)]
g = sns.catplot(
    data=pitch_avg_period_df[(pitch_avg_period_df.Model.str.startswith("LES")) & (pitch_avg_period_df.Amplitude < 20)],
    x="Amplitude",
    y="an_Turb",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

g.figure.suptitle('Pitching Turbine: Average $a_n$ vs Amplitude')

for ctp, ax in g.axes_dict.items():
    sub_data = smaller_pitch_data[(smaller_pitch_data["Local Thrust Coefficient"] == ctp)]
    sns.stripplot(ax = ax, data = sub_data, x = "Amplitude", y = "an_Turb",  hue="Frequency", dodge = True, palette=error_palette, edgecolor=accent_line_color, linewidth=1.5, marker = "v", s =5, legend = False)

umm_line = Line2D([], [], color=None, marker = "v", label='UMM', markerfacecolor="white", markeredgecolor = accent_line_color, linewidth=0, markeredgewidth=1)
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "LES Frequencies")

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$a_n$")
g._legend.set_bbox_to_anchor((1.05, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()

# %%
smaller_pitch_data = pitch_avg_period_df[(pitch_avg_period_df.Model.str.startswith("UMM")) & (pitch_avg_period_df.Amplitude < 20)]
g = sns.catplot(
    data=pitch_avg_period_df[(pitch_avg_period_df.Model.str.startswith("LES")) & (pitch_avg_period_df.Amplitude < 20)],
    x="Amplitude",
    y="ct_Turb",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

g.figure.suptitle('Pitching Turbine')

for ctp, ax in g.axes_dict.items():
    sub_data = smaller_pitch_data[(smaller_pitch_data["Local Thrust Coefficient"] == ctp)]
    sns.stripplot(ax = ax, data = sub_data, x = "Amplitude", y = "ct_Turb",  hue="Frequency", dodge = True, palette=error_palette, edgecolor=accent_line_color, linewidth=2, marker = "v", s =8, legend = False)

umm_line = Line2D([], [], color=None, marker = "v", label='UMM', markerfacecolor="white", markeredgecolor = accent_line_color, linewidth=0, markeredgewidth=3, markersize=12)
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequency $(f)$")

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_P$)", "Mean $C_T$")
g._legend.set_bbox_to_anchor((1.0, 0.5))
g._legend.set_loc("center left")
plt.tight_layout()

# %%
g = sns.catplot(
    data=pitch_avg_period_df[(pitch_avg_period_df.Model.str.startswith("UMM")) & (pitch_avg_period_df.Amplitude < 20)],
    x="Amplitude",
    y="an_Turb_error",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="bar",
    palette=error_palette,
    sharex=False,
    sharey=True,
)

g.figure.suptitle('Pitching Turbine: Average $a_n$ Error vs Amplitude')

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$a_n$ Error [%]")
g._legend.set_bbox_to_anchor((1.05, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()

# %% [markdown]
# ## Plot Phase Mean Values
#
# Now that I have the straight up mean, I want to try with a boxplot that shows how the error and the value changes over the course of a phase, but without focusing on exactly at what stage of the phase the error is larger or smaller.

# %%
les_stationary_low_ctp_udisk = 0.758605

# %%
umm_stationary_low_ctp_udisk

# %%
surge_disk_data = surge_avg_df_combo_ctp[(surge_avg_df_combo_ctp.Amplitude < 1)].sort_values(by = ["Rounded_Phase"])
surge_disk_data["UDisk_Ground_Normalized"] = np.where(surge_disk_data.Model.str.startswith("LES"), surge_disk_data["UDisk_Ground"] / les_stationary_low_ctp_udisk, surge_disk_data["UDisk_Ground"] / umm_stationary_low_ctp_udisk)
surge_disk_data["UDisk_Turb_Normalized"] = np.where(surge_disk_data.Model.str.startswith("LES"), surge_disk_data["UDisk_Turb"] / les_stationary_low_ctp_udisk, surge_disk_data["UDisk_Turb"] / umm_stationary_low_ctp_udisk)

# %%
# LES_sub_data = surge_disk_data[surge_disk_data.Model.str.startswith("LES")]
# LES_sub_data["UDisk_Turb_Normalized"] = surge_disk_data["UDisk_Turb"] / les_stationary_low_ctp_udisk

# UMM_sub_data = surge_disk_data[surge_disk_data.Model.str.startswith("UMM")]
# UMM_sub_data["UDisk_Turb_Normalized"] = surge_disk_data["UDisk_Turb"] / umm_stationary_low_ctp_udisk

# %%
g = sns.relplot(
    data=surge_disk_data[(surge_disk_data.Model.str.startswith("LES")) & (surge_disk_data.Amplitude < 1) & (surge_disk_data["Local Thrust Coefficient"] < 2)].sort_values(by = ["Rounded_Phase"]),
    x="UInf_Turb",
    y="UDisk_Turb_Normalized",
    style = "Model",
    hue = "Frequency",
    col = "Amplitude",
    row = "Local Thrust Coefficient",
    kind="line",
    height=4,
    palette=error_palette,
    facet_kws={"sharey": True, "sharex": True},
    linewidth = 3,
    alpha = 0.7,
    sort=False
)
g.set_titles(row_template="$C_T'$ = {row_name}", col_template="Amplitude = {col_name}")
g.set_axis_labels("Relative Freestream ($u_{\infty, t}$)", "$u_{d, t} / u_\\text{fb}$", size = 20)
plt.tight_layout()
g.figure.suptitle("Disk Velocity vs Turbine Velocity in the Ground Frame", y = 1.05)

les_line = Line2D([], [], color=dark_blues[3], label='LES')
umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[0].get_legend_handles_labels()
handles = handles[:-1]
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [les_line, umm_line], loc="center left")
g._legend.set_bbox_to_anchor((0.925, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")

umm_data = surge_disk_data[surge_disk_data.Model.str.startswith("UMM")].sort_values(by = ["Rounded_Phase"])
for (row, col, _), data_subset in g.facet_data():
    ax = g.facet_axis(row, col)
    amp, ctp = data_subset["Amplitude"].unique(), data_subset["Local Thrust Coefficient"].unique()
    if (len(amp) > 0):
        umm_subset = umm_data[(umm_data["Amplitude"] == amp[0]) & (umm_data["Local Thrust Coefficient"] == ctp[0])]
        ax.axvline(1, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)
        ax.axhline(1, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)
        ax.plot(umm_subset["UInf_Turb"], umm_subset["UDisk_Turb_Normalized"], c = accent_line_color, linewidth = 3, alpha = 0.7, zorder = 1)

# %%
g = sns.relplot(
    data=surge_disk_data[(surge_disk_data.Model.str.startswith("LES")) &  (surge_avg_df_combo_ctp["Local Thrust Coefficient"] < 2)],
    x="UInf_Turb",
    y="UDisk_Ground_Normalized",
    style = "Model",
    hue = "Frequency",
    col = "Amplitude",
    row = "Local Thrust Coefficient",
    kind="line",
    palette=error_palette,
    facet_kws={"sharey": True, "sharex": True},
    linewidth = 3,
    alpha = 0.7,
    sort=False
)
g.set_titles(row_template="$C_T'$ = {row_name}", col_template="Amplitude = {col_name}")
g.set_axis_labels("Relative Freestream ($u_{\infty, t}$)", "$(u_{d, t} + u_{t, g})/u_{fb}$",  size = 20)
plt.tight_layout()
g.figure.suptitle(" ", y = 1.05)

les_line = Line2D([], [], color=dark_blues[3], label='LES')
umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[0].get_legend_handles_labels()
handles = handles[:-1]
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [les_line, umm_line], loc="center left")
g._legend.set_bbox_to_anchor((0.925, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")

umm_data = surge_disk_data[surge_disk_data.Model.str.startswith("UMM")].sort_values(by = ["Rounded_Phase"])
for (row, col, _), data_subset in g.facet_data():
    ax = g.facet_axis(row, col)
    amp, ctp = data_subset["Amplitude"].unique(), data_subset["Local Thrust Coefficient"].unique()
    if (len(amp) > 0):
        umm_subset = umm_data[(umm_data["Amplitude"] == amp[0]) & (umm_data["Local Thrust Coefficient"] == ctp[0])]
        ax.axvline(1, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)
        ax.axhline(1, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)
        ax.plot(umm_subset["UInf_Turb"], umm_subset["UDisk_Ground_Normalized"], c = accent_line_color, linewidth = 3, alpha = 0.7, zorder = 1)

# %%
g = sns.relplot(
    data=surge_disk_data[(surge_disk_data.Model.str.startswith("LES")) & (surge_disk_data.Amplitude < 1)].sort_values(by = ["Rounded_Phase"]),
    x="UTurb",
    y="UDisk_Turb_Normalized",
    style = "Model",
    hue = "Frequency",
    col = "Amplitude",
    row = "Local Thrust Coefficient",
    kind="line",
    height=4,
    palette=error_palette,
    facet_kws={"sharey": True, "sharex": True},
    linewidth = 3,
    alpha = 0.7,
    sort=False
)

g.set_titles(row_template="$C_T'$ = {row_name}", col_template="Amplitude = {col_name}")
g.set_axis_labels("Turbine Velocity ($u_{t, t}$)", "$\\frac{u_{d, t}}{u_{fb}}$")
plt.tight_layout()
g.figure.suptitle("Disk Velocity vs Normalized Freestream Velocity in the Turbine Frame", y = 1.05)

les_line = Line2D([], [], color=dark_blues[3], label='LES')
umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[0].get_legend_handles_labels()
handles = handles[:-1]
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [les_line, umm_line], loc="center left")
g._legend.set_bbox_to_anchor((0.925, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")

umm_data = surge_disk_data[surge_disk_data.Model.str.startswith("UMM")].sort_values(by = ["Rounded_Phase"])
for (row, col, _), data_subset in g.facet_data():
    ax = g.facet_axis(row, col)
    amp, ctp = data_subset["Amplitude"].unique(), data_subset["Local Thrust Coefficient"].unique()
    if (len(amp) > 0):
        umm_subset = umm_data[(umm_data["Amplitude"] == amp[0]) & (umm_data["Local Thrust Coefficient"] == ctp[0])]
        ax.axvline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)
        ax.plot(umm_subset["UTurb"], umm_subset["UDisk_Turb_Normalized"], c = accent_line_color, linewidth = 3, alpha = 0.7, label = (umm_subset["Model"]).unique()[0], zorder = 1)

# %%
g = sns.relplot(
    data=surge_avg_df_combo_ctp[(surge_avg_df_combo_ctp.Model.str.startswith("LES")) & (surge_avg_df_combo_ctp.Amplitude < 1)].sort_values(by = ["Rounded_Phase"]),
    x="UTurb",
    y="ct_Simple",
    style = "Model",
    hue = "Frequency",
    col = "Amplitude",
    row = "Local Thrust Coefficient",
    kind="line",
    height=4,
    palette=error_palette,
    facet_kws={"sharey": True, "sharex": True},
    linewidth = 3,
    alpha = 0.7,
    sort=False
)
g.set_titles(row_template="$C_T'$ = {row_name}", col_template="Amplitude = {col_name}")
g.set_axis_labels("Turbine Velocity ($u_{t, g}$)", "$C_{T, s}$")
plt.tight_layout()
g.figure.suptitle("$C_{T, s}$ vs Turbine Velocity", y = 1.05)

les_line = Line2D([], [], color=dark_blues[3], label='LES')
umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[0].get_legend_handles_labels()
handles = handles[:-1]
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [les_line, umm_line], loc="center left")
g._legend.set_bbox_to_anchor((0.925, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")

umm_data = surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Model.str.startswith("UMM")].sort_values(by = ["Rounded_Phase"])
for (row, col, _), data_subset in g.facet_data():
    ax = g.facet_axis(row, col)
    amp, ctp = data_subset["Amplitude"].unique(), data_subset["Local Thrust Coefficient"].unique()
    if (len(amp) > 0):
        umm_subset = umm_data[(umm_data["Amplitude"] == amp[0]) & (umm_data["Local Thrust Coefficient"] == ctp[0])]
        ax.axvline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)
        ax.plot(umm_subset["UTurb"], umm_subset["ct_Simple"], c = accent_line_color, linewidth = 3, alpha = 0.7, zorder = 1)

# %%
pitch_disk_data = pitch_avg_df_combo_ctp[(pitch_avg_df_combo_ctp.Amplitude < 20) & (pitch_avg_df_combo_ctp["Local Thrust Coefficient"] == 1.33)].sort_values(by = ["Rounded_Phase"])
pitch_disk_data["UDisk_Ground_Normalized"] = np.where(pitch_disk_data.Model.str.startswith("LES"), pitch_disk_data["UDisk_Ground"] / les_stationary_low_ctp_udisk, pitch_disk_data["UDisk_Ground"] / umm_stationary_low_ctp_udisk)
pitch_disk_data["UDisk_Turb_Normalized"] = np.where(pitch_disk_data.Model.str.startswith("LES"), pitch_disk_data["UDisk_Turb"] / les_stationary_low_ctp_udisk, pitch_disk_data["UDisk_Turb"] / umm_stationary_low_ctp_udisk)
pitch_disk_data["Tilt_degs"] = np.rad2deg(pitch_disk_data["Tilt"])

# %%
g = sns.relplot(
    data=pitch_disk_data[pitch_disk_data.Model.str.startswith("LES")].sort_values(by = ["Rounded_Phase"]),
    x="Tilt",
    y="UDisk_Turb_Normalized",
    style = "Model",
    hue = "Frequency",
    col = "Amplitude",
    row = "Local Thrust Coefficient",
    kind="line",
    height=4,
    palette=error_palette,
    facet_kws={"sharey": "row", "sharex": True},
    linewidth = 3,
    alpha = 0.7,
    sort=False
)

g.set_titles(row_template="$C_T'$ = {row_name}", col_template="Amplitude = {col_name}")
g.set_axis_labels("Turbine Tilt ($\phi_t$)", "Disk Velocity ($u_d$)")
plt.tight_layout()
g.figure.suptitle("Disk Velocity vs Turbine Tilt ", y = 1.05)

les_line = Line2D([], [], color=dark_blues[3], label='LES')
umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[0].get_legend_handles_labels()
handles = handles[:-1]
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [les_line, umm_line], loc="center left")
g._legend.set_bbox_to_anchor((0.925, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")

umm_data = pitch_disk_data[pitch_disk_data.Model.str.startswith("UMM")].sort_values(by = ["Rounded_Phase"])
for (row, col, _), data_subset in g.facet_data():
    ax = g.facet_axis(row, col)
    amp, ctp = data_subset["Amplitude"].unique(), data_subset["Local Thrust Coefficient"].unique()
    if (len(amp) > 0):
        umm_subset = umm_data[(umm_data["Amplitude"] == amp[0]) & (umm_data["Local Thrust Coefficient"] == ctp[0])]
        ax.axvline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)
        ax.plot(umm_subset["Tilt"], umm_subset["UDisk_Turb_Normalized"], c = accent_line_color, linewidth = 3, alpha = 0.7, label = (umm_subset["Model"]).unique()[0], zorder = 1)

# %%
g = sns.relplot(
    data=pitch_disk_data[(pitch_disk_data.Model.str.startswith("LES"))],
    x="Tilt_degs",
    y="UDisk_Turb_Normalized",
    style = "Model",
    hue = "Frequency",
    col = "Amplitude",
    row = "Local Thrust Coefficient",
    kind="line",
    palette=error_palette,
    facet_kws={"sharey": True, "sharex": True},
    linewidth = 3,
    alpha = 0.7,
    sort=False
)
g.set_titles(row_template="$C_T'$ = {row_name}", col_template="Amplitude = {col_name}")
g.set_axis_labels("Amplitude ($A_P$)", "Normalized Disk Velocity ($\\frac{u_{d}}{u_{fb}}$)")
plt.tight_layout()
g.figure.suptitle("Disk Velocity vs Tilt", y = 1.07)

les_line = Line2D([], [], color=dark_blues[3], label='LES')
umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[0].get_legend_handles_labels()
handles = handles[:-1]
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [les_line, umm_line], loc="center left")
g._legend.set_bbox_to_anchor((0.925, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")

umm_data = pitch_disk_data[pitch_disk_data.Model.str.startswith("UMM")].sort_values(by = ["Rounded_Phase"])
for (row, col, _), data_subset in g.facet_data():
    ax = g.facet_axis(row, col)
    amp, ctp = data_subset["Amplitude"].unique(), data_subset["Local Thrust Coefficient"].unique()
    if (len(amp) > 0):
        umm_subset = umm_data[(umm_data["Amplitude"] == amp[0]) & (umm_data["Local Thrust Coefficient"] == ctp[0])]
        ax.axvline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)
        ax.plot(umm_subset["Tilt_degs"], umm_subset["UDisk_Turb_Normalized"], c = accent_line_color, linewidth = 3, alpha = 0.7, zorder = 1)

# %%
surge_avg_df_lower_amp =  surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Amplitude < 1].sort_values(by = ["Model", "Frequency"], ascending = [False, True])
g = sns.catplot(
    data=surge_avg_df_lower_amp,
    x="Amplitude",
    y="an_Turb",
    hue="Model_Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    height=4,
    aspect=1.2,
    palette=palette,
    sharex=True,
    showmeans=True,
    meanprops={
        "marker": "x",        # Diamond shape
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)
(ax1, ax2) = g.axes.flat
avg_val1 = np.average(surge_avg_df_lower_amp[(surge_avg_df_lower_amp.Model.str.startswith("UMM")) & (surge_avg_df_lower_amp["Local Thrust Coefficient"] == 1.33)].an_Turb)
avg_val2 = np.average(surge_avg_df_lower_amp[(surge_avg_df_lower_amp.Model.str.startswith("UMM")) & (surge_avg_df_lower_amp["Local Thrust Coefficient"] == 2)].an_Turb)
ax1.axhline(avg_val1, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot
ax2.axhline(avg_val2, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot

g._legend.set_title("Model: Frequency")

g.figure.suptitle('Surging Turbine: $a_n$ vs Amplitude', y = 1.05)

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$a_n$")

# %%
g = sns.catplot(
    data=surge_avg_df_lower_amp[surge_avg_df_lower_amp.Model.str.startswith("UMM")],
    x="Amplitude",
    y="an_Turb_abs_error",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    height=4,
    aspect=1.2,
    palette=error_palette,
    sharex=True,
    showmeans=True,
    meanprops={
        "marker": "x",        # Diamond shape
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)
g.figure.suptitle('Surging Turbine: $a_{n, t}$ Error vs Amplitude', y = 1.05)

(ax1, ax2) = g.axes.flat
ax1.axhline(0, ls=":", c="darkgrey", linewidth = 2, zorder = 1, alpha = 0.75)   # line at y=20 across whole subplot
ax2.axhline(0, ls=":", c="darkgrey", linewidth = 2,  zorder = 1, alpha = 0.75)   # line at y=20 across whole subplot

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$a_{n, t}$ Error (UMM - LES)")

# %%
g = sns.catplot(
    data=surge_avg_df_lower_amp[surge_avg_df_lower_amp.Model.str.startswith("LES")],
    x="Amplitude",
    y="ct_Turb",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    height=4,
    aspect=1.2,
    palette=error_palette,
    sharex=True,
    showmeans=True,
    meanprops={
        "marker": "x",        # Diamond shape
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    },
    
)
g.figure.suptitle('Surging Turbine: $C_{T, t}$ vs Amplitude', y = 1.05)
g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$C_{T, t}$")


(ax1, ax2) = g.axes.flat
avg_val1 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 1.33)].ct_Turb)
avg_val2 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 2)].ct_Turb)
ax1.axhline(avg_val1, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot
ax2.axhline(avg_val2, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot

umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequency")
g._legend.set_bbox_to_anchor((0.9, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")


# %%
g = sns.catplot(
    data=surge_avg_df_lower_amp[surge_avg_df_lower_amp.Model.str.startswith("UMM")],
    x="Amplitude",
    y="ct_Turb_abs_error",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    palette=error_palette,
    sharex=False,
    sharey=False,
    showmeans=True,
    meanprops={
        "marker": "x",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)

(ax1, ax2) = g.axes.flat
# avg_val1 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 1.33)].ct_Turb)
# avg_val2 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 2)].ct_Turb)
ax1.axhline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)   # line at y=20 across whole subplot
ax2.axhline(0, ls="--", c="lightgrey", linewidth = 2.5,  zorder = 1)   # line at y=20 across whole subplot

# umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# # Grab existing legend handles and labels
# handles, labels = g.axes.flat[1].get_legend_handles_labels()
# print(handles, labels)
# # Update the FacetGrid legend to include the custom elements
# g._legend.remove()  # remove old legend
# g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequencies")

g.figure.suptitle('Surging Turbine')

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_S$)", "$C_{T, t}$ Error (UMM - LES)")
g._legend.set_bbox_to_anchor((1.0, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()


# %%
g = sns.catplot(
    data=surge_avg_df_lower_amp[surge_avg_df_lower_amp.Model.str.startswith("UMM")],
    x="Amplitude",
    y="ct_Simple_abs_error",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    palette=error_palette,
    sharex=False,
    sharey=False,
    showmeans=True,
    meanprops={
        "marker": "x",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)

(ax1, ax2) = g.axes.flat
# avg_val1 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 1.33)].ct_Turb)
# avg_val2 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 2)].ct_Turb)
ax1.axhline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)   # line at y=20 across whole subplot
ax2.axhline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)   # line at y=20 across whole subplot

# umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# # Grab existing legend handles and labels
# handles, labels = g.axes.flat[1].get_legend_handles_labels()
# print(handles, labels)
# # Update the FacetGrid legend to include the custom elements
# g._legend.remove()  # remove old legend
# g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequencies")

g.figure.suptitle('Surging Turbine')

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_S$)", "$C_{T, s}$ Error (UMM - LES)")
g._legend.set_bbox_to_anchor((1.0, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()

# %%
g = sns.catplot(
    data=surge_avg_df_lower_amp[surge_avg_df_lower_amp.Model.str.startswith("UMM")],
    x="Amplitude",
    y="cp_Ground_abs_error",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    palette=error_palette,
    sharex=False,
    sharey=True,
    showmeans=True,
    meanprops={
        "marker": "x",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)

(ax1, ax2) = g.axes.flat
# avg_val1 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 1.33)].ct_Turb)
# avg_val2 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 2)].ct_Turb)
ax1.axhline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)   # line at y=20 across whole subplot
# ax2.axhline(0, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot

umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequencies")

g.figure.suptitle('Surging Turbine')

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_S$)", "$C_{P}$ Error (UMM - LES)")
g._legend.set_bbox_to_anchor((1.0, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()

# %%
pitch_avg_df_lower_amp =  pitch_avg_df_combo_ctp[pitch_avg_df_combo_ctp.Amplitude < 20].sort_values(by = ["Model", "Frequency"], ascending = [False, True])
g = sns.catplot(
    data=pitch_avg_df_lower_amp,
    x="Amplitude",
    y="an_Turb",
    hue="Model_Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    palette=palette,
    sharey = True,
    sharex=True,
    showmeans=True,
    meanprops={
        "marker": "x",        # Diamond shape
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)

g.figure.suptitle('Pitching Turbine: $a_n$ vs Amplitude', y = 1.05)

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$a_n$")

# %%
g = sns.catplot(
    data=pitch_avg_df_lower_amp[pitch_avg_df_lower_amp.Model.str.startswith("UMM")],
    x="Amplitude",
    y="an_Turb_error",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    palette=error_palette,
    sharex=True,
    showmeans=True,
    meanprops={
        "marker": "x",        # Diamond shape
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)
g.figure.suptitle('Pitching Turbine: $a_n$ Error vs Amplitude', y = 1.05)

(ax1, ax2) = g.axes.flat
ax1.axhline(0, ls=":", c="darkgrey", linewidth = 2, zorder = 1, alpha = 0.75)   # line at y=20 across whole subplot
ax2.axhline(0, ls=":", c="darkgrey", linewidth = 2,  zorder = 1, alpha = 0.75)   # line at y=20 across whole subplot

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude", "$a_n$ Error [%]")

# %%
g = sns.catplot(
    data=pitch_avg_df_lower_amp[pitch_avg_df_lower_amp.Model.str.startswith("UMM")],
    x="Amplitude",
    y="ct_Turb_abs_error",
    hue="Frequency",
    col="Local Thrust Coefficient",
    kind="box",
    palette=error_palette,
    sharex=False,
    sharey=False,
    showmeans=True,
    meanprops={
        "marker": "x",
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)

(ax1, ax2) = g.axes.flat
# avg_val1 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 1.33)].ct_Turb)
# avg_val2 = np.average(surge_avg_period_df[(surge_avg_period_df.Model.str.startswith("UMM")) & (surge_avg_period_df["Local Thrust Coefficient"] == 2)].ct_Turb)
ax1.axhline(0, ls="--", c="lightgrey", linewidth = 2.5, zorder = 1)   # line at y=20 across whole subplot
# ax2.axhline(0, ls="--", c=accent_line_color, linewidth = 2.5)   # line at y=20 across whole subplot

umm_line = Line2D([], [], color=accent_line_color, label='UMM')
# Grab existing legend handles and labels
handles, labels = g.axes.flat[1].get_legend_handles_labels()
print(handles, labels)
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles) + [umm_line], loc="center left", title = "Frequencies")

g.figure.suptitle('Pitching Turbine')

g.set_titles(col_template="$C_T'$ = {col_name}")
g.set_axis_labels("Amplitude ($A_P$)", "$C_{T}$ Error (UMM - LES)")
g._legend.set_bbox_to_anchor((1.0, 0.5))  # (x, y) position relative to figure
g._legend.set_loc("center left")
plt.tight_layout()


# %%
les_data = surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Model.str.startswith("LES")]
umm_data = surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Model.str.startswith("UMM")]
fig, (ax0, ax1) = plot_4panel_an_ct(les_data, xkey = "Rounded_Phase", errorbar = "sd", pre_title = "Averaged Surging", sharey = "row", palette = error_palette, two_panel = True)
sns.lineplot(ax = ax0, data = umm_data[umm_data["Local Thrust Coefficient"] == 1.33], x = "Rounded_Phase", y = "an_Turb", hue = "Amplitude", style = "Frequency", legend = False, palette = dark_oranges)
sns.lineplot(ax = ax1, data = umm_data[umm_data["Local Thrust Coefficient"] == 2], x = "Rounded_Phase", y = "an_Turb", hue = "Amplitude", style = "Frequency", legend = False, palette = dark_oranges)
# sns.lineplot(ax = ax2, data = umm_data[umm_data["Local Thrust Coefficient"] == 1.33], x = "Rounded_Phase", y = "ct_Turb", hue = "Amplitude", style = "Frequency", legend = False, palette = dark_oranges)
# sns.lineplot(ax = ax3, data = umm_data[umm_data["Local Thrust Coefficient"] == 2], x = "Rounded_Phase", y = "ct_Turb", hue = "Amplitude", style = "Frequency", legend = False, palette = dark_oranges)

# ax0.set_ylim(0.1, 0.5)

# %%
les_data = surge_avg_df_combo_ctp[(surge_avg_df_combo_ctp.Model.str.startswith("LES")) & (surge_avg_df_combo_ctp.Amplitude < 1) & (surge_avg_df_combo_ctp.Frequency < 1)]
umm_data = surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Model.str.startswith("UMM") & (surge_avg_df_combo_ctp.Amplitude < 1) & (surge_avg_df_combo_ctp.Frequency < 1)]

fig, ax0 = plt.subplots(1, 1, sharey = True, sharex = True)

ax0.axvline(0.5, color='darkgrey', linestyle='--', linewidth=1, alpha=0.7)

alpha = 1

sns.lineplot(ax = ax0, data = les_data[(les_data["Local Thrust Coefficient"] == 1.33) & (les_data["Amplitude"] < 1)], x = "Rounded_Phase", y = "ct_Simple", hue = "Amplitude", style = "Frequency", palette = error_palette, alpha = alpha)

sns.lineplot(ax = ax0, data = umm_data[(umm_data["Local Thrust Coefficient"] == 1.33) & (umm_data["Amplitude"] < 1)], x = "Rounded_Phase", y = "ct_Simple", hue = "Amplitude", style = "Frequency", legend = False, palette = dark_oranges, alpha = alpha)
add_shading_4panel(les_data, "Rounded_Phase", (ax0,))

fix_plot_legend(ax0, xOffset = 0.75)
fig.suptitle("Surging Turbine $C_T$ vs Phase Fraction", y = 1.02, fontsize=18)

ax0.set_title("$C_T' = 1.33$", size = 16)
ax0.set_ylabel("$C_T$", size = 14)
ax0.set_xlabel("$t / T$", size = 14)
ax0.set_ylim(-0.25, 2.75)


# %%
plot_8panel_val(surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Frequency < 1], "Rounded_Phase", "an_Turb", None, "Averaged Surging", "t / T", "$a_n$", add_shading = True)
plot_8panel_val(pitch_avg_df_combo_ctp[pitch_avg_df_combo_ctp.Frequency < 1], "Rounded_Phase", "an_Turb", None, "Averaged Pitching", "t / T", "$a_n$", sharey = "row", add_shading = True)

# %%
from scipy.signal import find_peaks

offset_df = []
for (ct, freq, amp), group in pitch_avg_df_combo_ctp.groupby(["Local Thrust Coefficient", "Frequency", "Amplitude"]):
    umm_vals = group[group.Model.str.startswith("UMM")].sort_values(by = "Rounded_Phase")
    les_vals = group[group.Model.str.startswith("LES")].sort_values(by = "Rounded_Phase")
    x_vals = np.array(umm_vals.Rounded_Phase)
    troughs1, _ = find_peaks(umm_vals.an_Turb)
    troughs2, _ = find_peaks(les_vals.an_Turb)
    x_trough1 = x_vals[troughs1[0]]
    x_trough2 =x_vals[troughs2[0]]

    offset = x_trough2 - x_trough1

    temp_df = pd.DataFrame({
        "Local Thrust Coefficient": [ct],
        "Amplitude": [amp],
        "Frequency": [freq],
        "Offset": [offset],
    })
    offset_df.append(temp_df)
offset_df = pd.concat(offset_df, ignore_index=True)

# %%
sns.relplot(data = offset_df,kind = "line", col = "Local Thrust Coefficient", x = "Frequency", y = "Offset", hue = "Amplitude", palette=error_palette)

# %%
plot_8panel_val(surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Frequency < 1], "Rounded_Phase", "ct_Turb", None, "Averaged Surging", "t / T", "$C_T$", sharey = "row")
plot_8panel_val(pitch_avg_df_combo_ctp[pitch_avg_df_combo_ctp.Frequency < 1], "Rounded_Phase", "ct_Turb", None, "Averaged Pitching", "t / T", "$C_T$", sharey = "row")

# %%
plot_8panel_val(surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Model != "LES"], "Rounded_Phase", "an_Turb_error", None, "Error in Average Surging", "t / T", "$a_n$ Error [%]")
plot_8panel_val(pitch_avg_df_combo_ctp[pitch_avg_df_combo_ctp.Model != "LES"], "Rounded_Phase", "an_Turb_error", None, "Error in Average Pitching", "t / T", "$a_n$ Error [%]", sharey = "row")

# %%
plot_8panel_val(surge_avg_df_combo_ctp[surge_avg_df_combo_ctp.Model != "LES)"], "Rounded_Phase", "an_Turb_error", None, "Error in Average Surging", "t / T", "$a_n$ Error [%]", hue_key="Frequency", sharey = False)

# %%
surge_mask = (surge_avg_df_combo_ctp.Model != "LES)") & (surge_avg_df_combo_ctp.Amplitude < 0.8)
pitch_mask = (pitch_avg_df_combo_ctp.Model != "LES")
plot_8panel_val(surge_avg_df_combo_ctp[surge_mask], "Rounded_Phase", "ct_Turb_error", None, "Error in Average Surging", "t / T", "$C_T$ Error [%]", sharey = False)
plot_8panel_val(pitch_avg_df_combo_ctp[pitch_mask], "Rounded_Phase", "ct_Turb_error", None, "Error in Average Pitching", "t / T", "$C_T$ Error [%]", sharey = "row")

# %%
surge_df_umm_ctp = df_umm_ctp[df_umm_ctp.Movement == "Surge"]
pitch_df_umm_ctp = df_umm_ctp[df_umm_ctp.Movement == "Pitch"]

# %%
g = plot_8panel_val(surge_avg_df_les, "an_Turb", "ct_Turb", None, "Averaged Surging", "$a_n$", "$C_T$", add_shading = False)

for (ctp, freq), ax in g.axes_dict.items():
    sub_data = surge_df_umm_ctp[(surge_df_umm_ctp["Local Thrust Coefficient"] == ctp) & (surge_df_umm_ctp["Frequency"] == freq)]
    ax.scatter(x = sub_data["an_Turb"], y = sub_data["ct_Turb"], c = "k", zorder = 10, marker = "*", s = 120, label = "UMM")

# Grab existing legend handles and labels
handles, labels = g.axes.flat[0].get_legend_handles_labels()
# Update the FacetGrid legend to include the custom elements
g._legend.remove()  # remove old legend
g.add_legend(handles=list(handles), loc="center left")
g._legend.set_title(" ", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
g._legend.set_bbox_to_anchor((0.92, 0.5))  # (x, y) position relative to figure

# %%
g = plot_8panel_val(pitch_avg_df_combo_ctp, "an_Turb", "ct_Turb", None, "Averaged Pitching", "$a_n$", "$C_T$", "row", True, add_shading = False)

# %%
low_ctp_avg_df_ct = pd.concat([avg_df_les[avg_df_les["Local Thrust Coefficient"] == 1.33], low_ctp_df_umm_ct], ignore_index=True)
low_ctp_surge_avg_df_ct = low_ctp_avg_df_ct[low_ctp_avg_df_ct.Movement == "Surge"]
low_ctp_pitch_avg_df_ct = low_ctp_avg_df_ct[low_ctp_avg_df_ct.Movement == "Pitch"]

# %%
high_ctp_avg_df_ct = pd.concat([avg_df_les[avg_df_les["Local Thrust Coefficient"] == 2], high_ctp_df_umm_ct], ignore_index=True)
high_ctp_surge_avg_df_ct = high_ctp_avg_df_ct[high_ctp_avg_df_ct.Movement == "Surge"]
high_ctp_pitch_avg_df_ct = high_ctp_avg_df_ct[high_ctp_avg_df_ct.Movement == "Pitch"]

# %%
g = sns.relplot(
    data=low_ctp_surge_avg_df_ct.sort_values("Rounded_Phase"),
    x="Rounded_Phase",
    y="Local Thrust Coefficient",
    hue = "Amplitude",
    style = "Model",
    hue_order=get_hue_order(low_ctp_surge_avg_df_ct),
    col="Frequency",
    kind="line",
    palette="Set2",
    facet_kws={"sharey": False, "sharex": False},
    alpha = 0.6,
)

# %%
g = sns.relplot(
    data=high_ctp_surge_avg_df_ct.sort_values("Rounded_Phase"),
    x="Rounded_Phase",
    y="Local Thrust Coefficient",
    hue = "Amplitude",
    style = "Model",
    hue_order=get_hue_order(high_ctp_surge_avg_df_ct),
    col="Frequency",
    kind="line",
    palette="Set2",
    facet_kws={"sharey": False, "sharex": False},
    alpha = 0.6,
)

# %% [markdown]
# ## Examining the Surge Simulation's Induction Zone

# %%
df_surge_centerline = pd.read_csv("/Users/sky/src/HowlandLab/data/surge_centerline.csv")
df_surge_centerline = df_surge_centerline.rename(columns={'Thrust Coefficient': 'Local Thrust Coefficient'}) # disk velocity in the turbine frame of reference
df_surge_centerline = df_surge_centerline[((df_surge_centerline["Local Thrust Coefficient"] != 1.66) & (df_surge_centerline["Frequency"] < 1) & (df_surge_centerline["Amplitude"] < 1))]
df_surge_centerline = df_surge_centerline.dropna()
period = 1 / df_surge_centerline["Frequency"]
df_surge_centerline["Phase_Time"] = (df_surge_centerline["Time"] % period) / period
df_surge_centerline["Rounded_Phase_Time"] = np.round(df_surge_centerline["Phase_Time"], decimals = 8)
df_surge_centerline["Rounded_Phase_Time"] = np.where(df_surge_centerline["Rounded_Phase_Time"] == 1, 0, df_surge_centerline["Rounded_Phase_Time"])
df_surge_centerline["uVals_Turb"] = df_surge_centerline["uVals"] - df_surge_centerline["UTurb"]

# %%
small_df = df_surge_centerline[(df_surge_centerline.Frequency == 0.2) | (df_surge_centerline.Frequency == 0.6)]

g = sns.relplot(
    data=small_df,
    x="xVals",
    y="uVals",
    hue = "Amplitude",
    style = "Rounded_Phase_Time",
    col="Frequency",
    row = "Local Thrust Coefficient",
    kind="line",
    height=4,
    palette="Set2",
    facet_kws={"sharey": True, "sharex": True},
)
g.figure.suptitle('Surging Turbine Ground Perspective: $u_{x, y = 0, z = 0}$ vs $x$', y = 1.05)

g.set_titles(row_template="$C_T'$ = {row_name}",col_template="Frequency = {col_name}", )
g.set_axis_labels("$x$", "$u_{x, y = 0, z = 0}$")

# %%
small_df = df_surge_centerline[(df_surge_centerline.Frequency == 0.2) | (df_surge_centerline.Frequency == 0.6)]

g = sns.relplot(
    data=small_df,
    x="xVals",
    y="uVals_Turb",
    hue = "Amplitude",
    style = "Rounded_Phase_Time",
    col="Frequency",
    row = "Local Thrust Coefficient",
    kind="line",
    height=4,
    palette="Set2",
    facet_kws={"sharey": True, "sharex": True},
)

g.figure.suptitle('Surging Turbine Turbine Perspective: $u_{x, y = 0, z = 0}$ vs $x$', y = 1.05)

g.set_titles(row_template="$C_T'$ = {row_name}",col_template="Frequency = {col_name}", )
g.set_axis_labels("$x$", "$u_{x, y = 0, z = 0}$")


# %% [markdown]
# ## $a_n$ Analysis
#
# There are a lot of different ways to calculate $a_n$, so the first thing to do is to decide on what way we want to do that - and also to make sure that our definitions match across LES and the quasi-steady UMM. To check on this, I plotted all of the different definitions for a small subset of the data. As we can see below, it looks reasonable.

# %%
# def small_mask(df, f, A, ctp, t):
#     return df[(df.Frequency == f) & (df.Amplitude == A) & (df["Local Thrust Coefficient"] == ctp) & (df.Time > t)]

# f, A, ctp, t = 0.2, 0.4, 1.33, 290
# df_les_small_surge = small_mask(df_les, f, A, ctp, t)
# df_umm_ctp_small_surge = small_mask(df_umm_ctp, f, A, ctp, t)

# %%
# fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5), (ax6, ax7)) = plt.subplots(4, 2, sharey = True, sharex = True, figsize = (10, 8))
# fig.suptitle("Comparison of $a_n$ Calculations for Surge")

# ax0.set_title("Surge")
# ax0.set_ylabel(" ")
# sns.scatterplot(ax = ax0, data=df_les_small_surge, x = "Time", y = "UTurb", label = "$u_{t, g}$", color = "grey")
# ax1.set_title("$U_\infty$")
# ax1.set_ylabel(" ")
# sns.scatterplot(ax = ax1, data=df_les_small_surge, x = "Time", y = "UInf_Ground", label = "$U_{\infty, g}$", color = "lightblue")
# sns.scatterplot(ax = ax1, data=df_les_small_surge, x = "Time", y = "UInf_Turb", label = "$U_{\infty, t} = U_{\infty, g} - u_{t, g}$", color = "black")
# ax2.set_title("LES Velocities")
# ax2.set_ylabel(" ")
# sns.scatterplot(ax = ax2, data=df_les_small_surge, x = "Time", y = "UDisk_Ground", label = "$u_{d, g}$")
# sns.scatterplot(ax = ax2, data=df_les_small_surge, x = "Time", y = "UDisk_Turb", label = "$u_{d, t} = u_{d, g} - u_{t, g}$")
# ax3.set_title("UMM Velocities")
# ax3.set_ylabel(" ")
# sns.scatterplot(ax = ax3, data=df_umm_ctp_small_surge, x = "Time", y = "UDisk_Ground", label = "$u_{d, g}$", legend = False)
# sns.scatterplot(ax = ax3, data=df_umm_ctp_small_surge, x = "Time", y = "UDisk_Turb", label = "$u_{d, t}$", legend = False)
# ax4.set_title("LES $a_n$")
# ax4.set_ylabel(" ")
# sns.scatterplot(ax = ax4, data=df_les_small_surge, x = "Time", y = "an_Ground", label = "$a_{n, g} = 1 - (u_{d, g} / U_{\infty, g})$")
# sns.scatterplot(ax = ax4, data=df_les_small_surge, x = "Time", y = "an_Turb", label = "$a_{n, t} = 1 - (u_{d, t} / U_{\infty, t})$")
# sns.scatterplot(ax = ax4, data=df_les_small_surge, x = "Time", y = "an_Simple", label = "$a_{n, s} = 1 - (u_{d, t} / U_{\infty, g})$")
# ax5.set_title("UMM $a_n$")
# ax5.set_ylabel(" ")
# sns.scatterplot(ax = ax5, data=df_umm_ctp_small_surge, x = "Time", y = "an_Ground", label = "$a_{n, g} = 1 - (u_{d, g} / U_{\infty, g})$", legend = False)
# sns.scatterplot(ax = ax5, data=df_umm_ctp_small_surge, x = "Time", y = "an_Turb", label = "$a_{n, t} = 1 - (u_{d, t} / U_{\infty, t})$", legend = False)
# sns.scatterplot(ax = ax5, data=df_umm_ctp_small_surge, x = "Time", y = "an_Simple", label = "$a_{n, s} = 1 - (u_{d, t} / U_{\infty, g})$", legend = False)
# ax6.set_title("LES $C_T$")
# ax6.set_ylabel(" ")
# sns.scatterplot(ax = ax6, data=df_les_small_surge, x = "Time", y = "ct_Turb", label = "$C_{T, t}$")
# ax7.set_title("UMMa $C_T$")
# ax7.set_ylabel(" ")
# sns.scatterplot(ax = ax7, data=df_umm_ctp_small_surge, x = "Time", y = "ct_Turb", label = "$C_{T, t}", legend = False)

# fix_plot_legend(ax1, xOffset = 0.53, title = " ")
# fix_plot_legend(ax2, xOffset = 1.7, title = " ")
# fix_plot_legend(ax4, xOffset = 1.8, title = " ")
# fix_plot_legend(ax4, xOffset = 1.8, title = " ")

# %%
# f, A, ctp, t = 0.2, 8, 1.33, 290
# df_les_small_pitch = small_mask(df_les, f, A, ctp, t)
# df_umm_ctp_small_pitch = small_mask(df_umm_ctp, f, A, ctp, t)

# %%
# alpha = 0.5

# fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2, sharey = False, figsize = (12, 8))
# fig.suptitle("Comparison of $a_n$ Calculations for Pitch")

# ax0.set_title("Tilt")
# sns.scatterplot(ax = ax0, data=df_les_small_pitch, x = "Time", y = "Tilt", label = "$\phi$", color = "grey")
# ax0.set_ylabel(" ")

# ax1.set_title("$U_\infty$")
# sns.scatterplot(ax = ax1, data=df_les_small_pitch, x = "Time", y = "UInf_Ground", label = "$U_{\infty, g}$", color = "lightblue")
# sns.scatterplot(ax = ax1, data=df_les_small_pitch, x = "Time", y = "UInf_Turb", label = "$U_{\infty, t} = U_{\infty, g}\cos{\phi}$", color = "black")
# ax1.set_ylim(0.985, 1.005)
# ax1.set_ylabel(" ")
# ax2.set_title("LES Velocities")
# ax2.set_ylabel(" ")
# sns.scatterplot(ax = ax2, data=df_les_small_pitch, x = "Time", y = "UDisk_Ground", label = "$u_{d, g}$", alpha = alpha)
# sns.scatterplot(ax = ax2, data=df_les_small_pitch, x = "Time", y = "UDisk_Turb", label = "$u_{d, t}$", alpha = alpha)
# ax3.set_title("UMM Velocities")
# ax3.set_ylabel(" ")
# sns.scatterplot(ax = ax3, data=df_umm_ctp_small_pitch, x = "Time", y = "UDisk_Ground", label = "$u_{d, g}$", legend = False, alpha = alpha)
# sns.scatterplot(ax = ax3, data=df_umm_ctp_small_pitch, x = "Time", y = "UDisk_Turb", label = "$u_{d, t} = u_{d, g} - u_t $", legend = False, alpha = alpha)
# ax2.set_ylim(0.74, 0.76)
# ax3.set_ylim(0.74, 0.76)
# ax4.set_title("LES $a_n$ Calculations")
# ax4.set_ylabel(" ")
# sns.scatterplot(ax = ax4, data=df_les_small_pitch, x = "Time", y = "an_Ground", label = "$a_{n, g} = 1 - (u_{d, g} / U_{\infty, g})$", alpha = alpha)
# sns.scatterplot(ax = ax4, data=df_les_small_pitch, x = "Time", y = "an_Turb", label = "$a_{n, t} = 1 - (u_{d, t} / U_{\infty, t})$", alpha = alpha)
# # sns.scatterplot(ax = ax4, data=df_les_small_pitch, x = "Time", y = "an_Simple", label = "$a_{n, s} = 1 - (u_{d, t} / U_{\infty, g})$", alpha = alpha)

# ax5.set_title("LES $a_n$ Calculations")
# ax5.set_ylabel(" ")
# sns.scatterplot(ax = ax5, data=df_umm_ctp_small_pitch, x = "Time", y = "an_Ground", label = "$a_{n, g} = 1 - (u_{d, g} / U_{\infty, g})$", legend = False, alpha = alpha)
# sns.scatterplot(ax = ax5, data=df_umm_ctp_small_pitch, x = "Time", y = "an_Turb", label = "$a_{n, t} = 1 - (u_{d, t} / U_{\infty, t})$", legend = False, alpha = alpha)
# # sns.scatterplot(ax = ax5, data=df_umm_ctp_small_pitch, x = "Time", y = "an_Simple", label = "$a_{n, s} = 1 - (u_{d, t} / U_{\infty, g})$", legend = False, alpha = alpha)
# ax4.set_ylim(0.24, 0.26)
# ax5.set_ylim(0.24, 0.26)

# fix_plot_legend(ax2, xOffset = 1.45, title = " ")
# fix_plot_legend(ax4, xOffset = 1.75, title = " ")
# fig.subplots_adjust(hspace=0.5)

# %%
# g = sns.catplot(
#     data=df_combo[df_combo.Frequency > 0.1],
#     x="Amplitude",
#     y="an_Turb",
#     hue="Model Frequency",
#     col="Local Thrust Coefficient",
#     row="Movement",
#     kind="box",
#     height=4,
#     aspect=1.2,
#     palette="Set2",
#     sharex=False,
#     sharey="row",
#     showmeans=True,
#     meanprops={
#         "marker": "x",        # Diamond shape
#         "markerfacecolor": "black",
#         "markeredgecolor": "black",
#         "markersize": 4
#     }
# )

# g.figure.suptitle('$a_n$ vs Amplitude')

# g.set_titles(row_template="$C_T'$ = {col_name}", col_template="Movement = {row_name}")
# g.set_axis_labels("Amplitude", "$a_n$")
# g._legend.set_bbox_to_anchor((1.05, 0.5))  # (x, y) position relative to figure
# g._legend.set_loc("center left")
# plt.tight_layout()

# %%
# g = sns.catplot(
#     data=df_combo[(df_combo["Model"] == "UMM ($C_T'$)") & (df_combo["Frequency"] > 0.1)],
#     x="Amplitude",
#     y="an_Turb_error",
#     hue="Frequency",
#     col="Local Thrust Coefficient",
#     row="Movement",
#     kind="box",
#     height=4,
#     aspect=1.2,
#     palette="Set2",
#     sharex=False,
#     sharey="row",
#     showmeans=True,
#     meanprops={
#         "marker": "x",        # Diamond shape
#         "markerfacecolor": "black",
#         "markeredgecolor": "black",
#         "markersize": 4
#     }
# )

# g.figure.suptitle('$a_{n, t}$ Error vs Turbine Amplitude')

# g.set_titles(row_template="$C_T'$ = {col_name}", col_template="Movement = {row_name}")
# g.set_axis_labels("Amplitude [-]", "$a_{n, t}$ Error [%]")
# g._legend.set_bbox_to_anchor((1.05, 0.5))  # (x, y) position relative to figure
# g._legend.set_loc("center left")
# plt.tight_layout()

# %% [markdown]
# ## Period-Averaged Data
# At this point, we have a general idea of how the turbine-frame $a_n$ is distributed along amplitude and frequency (and $C_T'$ up to the Betz limit anyways). We will phase-average each of the runs to be able to plot smooth lines per run.

# %%
# kind = "linear"
# num_points = 1000
# grouped = df_combo.groupby(["Frequency", "Amplitude", "Model", "Movement", "Local Thrust Coefficient", "Phase"], as_index=False).agg({
#         "an_Turb": "mean",
#         "ct_Turb": "mean",
#         "UInf_Turb": "mean",
#         "UTurb": "mean",
#         "Tilt": "mean"
# })
# # Prepare output storage
# interp_results = []

# # Loop over each frequency group to interpolate separately
# for (freq, amp, model, movement, ctp), sub in grouped.groupby(["Frequency", "Amplitude", "Model", "Movement", "Local Thrust Coefficient"]):
#     # Ensure sorted by phase for interpolation
#     sub = sub.sort_values("Phase")
#     # Create interpolation function
#     f_an = interp1d(sub["Phase"], sub["an_Turb"], kind=kind, fill_value="extrapolate")
#     f_ct = interp1d(sub["Phase"], sub["ct_Turb"], kind=kind, fill_value="extrapolate")
#     f_uinf = interp1d(sub["Phase"], sub["UInf_Turb"], kind=kind, fill_value="extrapolate")
#     f_uturb = interp1d(sub["Phase"], sub["UTurb"], kind=kind, fill_value="extrapolate")
#     f_tilt = interp1d(sub["Phase"], sub["Tilt"], kind=kind, fill_value="extrapolate")
#     # Interpolated phases
#     phase_new = np.linspace(0, 1, num_points)
#     an_new = f_an(phase_new)
#     ct_new = f_ct(phase_new)
#     uinf_new = f_uinf(phase_new)
#     uturb_new = f_uturb(phase_new)
#     tilt_new = f_tilt(phase_new)

#     interp_results.append(pd.DataFrame({
#         "Frequency": freq,
#         "Amplitude": amp,
#         "Model": model,
#         "Movement": movement,
#         "Local Thrust Coefficient": ctp,
#         "Phase": phase_new,
#         "an_Turb": an_new,
#         "ct_Turb": ct_new,
#         "UInf_Turb": uinf_new,
#         "UTurb": uturb_new,
#         "Tilt": tilt_new,
#     }))

# phase_averaged_df = pd.concat(interp_results, ignore_index=True)
# phase_averaged_df = phase_averaged_df.sort_values(by = "Phase")

# %% [markdown]
# Since the UMM with $C_T$ input takes much longer to run, I will just run it for the phase-averged values!

# %%
# LES_phase_averaged_surge = phase_averaged_df[(phase_averaged_df["Model"] == "LES") & (phase_averaged_df["Movement"] == "Surge")]
# umm_ct_phase_averaged_surge = copy_df(LES_phase_averaged_surge, "UMM - $C_T$ Input")

# %%
# model_ct = ThrustBasedUnified()
# file_name_ct = "/Users/sky/src/HowlandLab/data/sim_16_phase_averaged_runs_data_UMM_CT.pkl"
# umm_results_ct = get_umm_values(umm_ct_phase_averaged_surge, model_ct, file_name_ct, thrust_coeff_key = "ct_Turb", pre_calculated = pre_calculated)

# %%
# umm_ct_phase_averaged_surge["UInf_Ground"] = 1
# umm_ct_phase_averaged_surge = calculate_umm_vals(umm_ct_phase_averaged_surge, umm_results_ct)

# %%
# def add_shading(g, df, phase_key = "Phase"):
#     upwind_uturb = df[(df["UTurb"] < 0) & (df["Frequency"] == 0.2) & (df["Amplitude"] == 0.2) & (df["Local Thrust Coefficient"] == 2)]
#     x1, x2 = np.min(upwind_uturb[phase_key]), np.max(upwind_uturb[phase_key])
#     for frequency, ax in g.axes_dict.items():
#         # vertical lines at x1 and x2
#         ax.axvline(x1, color='darkgrey', linestyle='--', linewidth=1, alpha=0.7)
#         ax.axvline(x2, color='darkgrey', linestyle='--', linewidth=1, alpha=0.7)
#         # shaded vertical region between x1 and x2
#         ax.axvspan(x1, x2, color='lightgrey', alpha=0.2)
#     # Create new legend element
#     region_heading = Line2D([], [], color='none', label='Regions')
#     shade_element = Patch(facecolor='lightgrey', label='$U_{\infty, t} > U_{\infty, g}$ ($u_{t, g} < 0$)')
#     # Grab existing legend handles and labels
#     handles, labels = g.axes.flat[0].get_legend_handles_labels()
#     # Update the FacetGrid legend to include the custom elements
#     g._legend.remove()  # remove old legend
#     g.add_legend(handles=list(handles) + [region_heading, shade_element], loc="center left")

# %%
# surge_low_ctp_df = phase_averaged_df[(phase_averaged_df["Local Thrust Coefficient"] == 1.33) & (phase_averaged_df["Movement"] == "Surge")]
# g = sns.relplot(
#     data=surge_low_ctp_df,
#     x="Phase",
#     y="an_Turb",
#     style="Model",
#     hue = "Amplitude",
#     col="Frequency",
#     col_wrap= 2,
#     kind="line",
#     height=4,
#     palette="Set2",
#     facet_kws={"sharey": "row", "sharex": True},
# )

# # add_shading(g, x1 = 0.25, x2 = 0.75)
# g.figure.suptitle("Surge Motion $a_n$ vs Phase", fontsize=22, y = 1.02)
# g.set_titles(col_template="Frequency = {col_name}", size = 16)
# g.set_axis_labels("t / T [-]", "$a_{n, t}$ [-]", fontsize = 16)
# [ax.tick_params(labelsize=18) for ax in g.axes.flat]

# g._legend.set_title(" ", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
# g._legend.set_bbox_to_anchor((1.04, 0.5))  # (x, y) position relative to figure
# plt.tight_layout()

# %%
# surge_low_ctp_df = phase_averaged_df[(phase_averaged_df["Local Thrust Coefficient"] == 1.33) & (phase_averaged_df["Movement"] == "Surge")]
# g = sns.relplot(
#     data=surge_low_ctp_df,
#     x="Phase",
#     y="ct_Turb",
#     style="Model",
#     hue = "Amplitude",
#     col="Frequency",
#     col_wrap= 2,
#     kind="line",
#     height=4,
#     palette="Set2",
#     facet_kws={"sharey": "row", "sharex": True},
# )

# add_shading(g, x1 = 0.25, x2 = 0.75)

# g.figure.suptitle("Surge Motion $C_T$ vs Phase", fontsize=22, y = 1.02)
# g.set_titles(col_template="Frequency = {col_name}", size = 16)
# g.set_axis_labels("t / T", "$C_{T, t}$ [-]", fontsize = 16)
# [ax.tick_params(labelsize=18) for ax in g.axes.flat]
# g._legend.set_title(" ", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
# g._legend.set_bbox_to_anchor((1.04, 0.5))  # (x, y) position relative to figure
# plt.tight_layout()

# %%
# phase_averaged_df = phase_averaged_df.sort_values(by = "Phase")
# g = sns.relplot(
#     data=phase_averaged_df[(phase_averaged_df["Local Thrust Coefficient"] == 1.33) & (phase_averaged_df["Movement"] == "Surge")],
#     x="Phase",
#     y="an_Turb",
#     style="Model",
#     hue = "Frequency",
#     col="Amplitude",
#     col_wrap= 2,
#     kind="line",
#     height=4,
#     palette="Set2",
#     facet_kws={"sharey": "row", "sharex": True},
# )
# add_shading(g, x1 = 0.25, x2 = 0.75)

# g.set_titles(col_template="Amplitude = {col_name}", size = 16)
# g.set_axis_labels("t / T", "$a_{n, t}$ [-]", fontsize = 16)
# g._legend.set_title("", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
# g._legend.set_bbox_to_anchor((1.04, 0.5))  # (x, y) position relative to figure
# g._legend.set_loc("center left")
# [ax.tick_params(labelsize=18) for ax in g.axes.flat]
# g.figure.suptitle("Surge Motion $a_n$ vs Phase", fontsize=22, y = 1.02)
# plt.tight_layout()

# %%
# phase_averaged_df = phase_averaged_df.sort_values(by = "Phase")
# g = sns.relplot(
#     data=phase_averaged_df[(phase_averaged_df["Local Thrust Coefficient"] == 1.33) & (phase_averaged_df["Movement"] == "Surge")],
#     x="Phase",
#     y="ct_Turb",
#     style="Model",
#     hue = "Frequency",
#     col="Amplitude",
#     col_wrap= 2,
#     kind="line",
#     height=4,
#     palette="Set2",
#     facet_kws={"sharey": "row", "sharex": True},
# )
# add_shading(g, x1 = 0.25, x2 = 0.75)

# g.set_titles(col_template="Amplitude = {col_name}", size = 16)
# g.set_axis_labels("t / T", "$C_{T, t}$ [-]", fontsize = 16)
# g._legend.set_title("", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
# g._legend.set_bbox_to_anchor((1.04, 0.5))  # (x, y) position relative to figure
# g._legend.set_loc("center left")
# [ax.tick_params(labelsize=18) for ax in g.axes.flat]
# g.figure.suptitle("Surge Motion $C_T$ vs Phase", fontsize=22, y = 1.02)
# plt.tight_layout()

# %%
# df_combo_pitch["Tilt_Degs"] = np.rad2deg(df_combo_pitch.Tilt)

# %%
# g = sns.relplot(
#     data=df_combo_pitch,
#     x="Tilt_Degs",
#     y="an_Turb",
#     style="Model",
#     hue = "Local Thrust Coefficient",
#     col="Frequency",
#     row="Amplitude",
#     kind="scatter",
#     height=4,
#     palette="Set2",
#     facet_kws={"sharey": "row", "sharex": True},
#     s = 60,
# )

# g.set_titles(row_template="Amplitude = {row_name}", col_template="Frequency = {col_name}", size = 16)
# g.set_axis_labels("$/phi$", "$a_{n, t}$ [-]", fontsize = 16)
# g._legend.set_title("$C_T '$ and Model", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
# g._legend.set_bbox_to_anchor((1.04, 0.5))  # (x, y) position relative to figure
# g._legend.set_loc("center left")
# [ax.tick_params(labelsize=18) for ax in g.axes.flat]
# g.figure.suptitle("Pitch Degrees $a_n$ vs Turbine Motion", fontsize=22, y = 1.02)
# plt.tight_layout()

# %%
# from scipy.fft import fft, fftfreq

# fft_records = []
# for (ct, move, freq, amp, model), group in df_combo.groupby(["Local Thrust Coefficient", "Movement", "Frequency", "Amplitude", "Model"]):
#     an_vals = group["an_Turb"].values
#     y = an_vals - np.mean(an_vals)
#     N = len(y)
#     T = group["Time"].iloc[1] - group["Time"].iloc[0]  # sampling interval

#     yf = fft(y)
#     xf = fftfreq(N, T)[:N//2]       # positive freqs only
#     power = 2.0 / N * np.abs(yf[0:N//2])

#     temp_df = pd.DataFrame({
#         "fft_freq": xf,
#         "fft_power": power,
#         "Local Thrust Coefficient": ct,
#         "Movement": move,
#         "Amplitude": amp,
#         "Frequency": freq,
#         "Model": model,
#     })
#     fft_records.append(temp_df)

# fft_df = pd.concat(fft_records, ignore_index=True)

# %%
# fft_surge_df = fft_df[fft_df["Movement"] == "Surge"]
# fft_pitch_df = fft_df[fft_df["Movement"] == "Pitch"]

# %%
# g = sns.relplot(
#     data=fft_surge_df[fft_surge_df["Local Thrust Coefficient"] == 1.33],
#     x="fft_freq",
#     y="fft_power",
#     hue="Model",
#     style = "Model",
#     col="Amplitude",
#     row="Frequency",
#     kind="line",
#     height=4,
#     aspect=1.5,
#     palette="Set2",
#     facet_kws={"sharey": "row", "sharex": False},
#     errorbar= None,
#     linewidth=3.5,
#     alpha = 0.5
# )

# g.set_titles(row_template="Frequency = {row_name}", col_template="Amplitude = {col_name}", size = 16)
# g.set_axis_labels("Frequency (Hz)", "FFT Power", fontsize = 16)
# g._legend.set_title("Model", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
# g._legend.set_bbox_to_anchor((1.04, 0.5))  # (x, y) position relative to figure
# g._legend.set_loc("center left")
# [ax.tick_params(labelsize=18) for ax in g.axes.flat]
# g.figure.suptitle("FFT of $a_n$ for Surge Motion at $C_T' = 1.33$", fontsize=22, y = 1.02)
# plt.tight_layout()
# g.set(xlim=(0, 5))

# %%
# g = sns.relplot(
#     data=fft_pitch_df[fft_pitch_df["Local Thrust Coefficient"] == 1.33],
#     x="fft_freq",
#     y="fft_power",
#     hue="Model",
#     style = "Model",
#     col="Amplitude",
#     row="Frequency",
#     kind="line",
#     height=4,
#     aspect=1.5,
#     palette="Set2",
#     facet_kws={"sharey": "row", "sharex": False},
#     errorbar= None,
#     linewidth=3.5,
#     alpha = 0.5
# )

# g.set_titles(row_template="Frequency = {row_name}", col_template="Amplitude = {col_name}", size = 16)
# g.set_axis_labels("Frequency (Hz)", "FFT Power", fontsize = 16)
# g._legend.set_title("Model", prop={'size': 18}); [t.set_fontsize(16) for t in g._legend.get_texts()]
# g._legend.set_bbox_to_anchor((1.04, 0.5))  # (x, y) position relative to figure
# g._legend.set_loc("center left")
# [ax.tick_params(labelsize=18) for ax in g.axes.flat]
# g.figure.suptitle("FFT of $a_n$ for Pitch Motion at $C_T' = 1.33$", fontsize=22, y = 1.02)
# plt.tight_layout()
# g.set(xlim=(0, 5))

# %%
