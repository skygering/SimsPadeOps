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

# %% [markdown]
# # Vertification of UMM with Tilt

# %%
import analysis.lib.analysis_utils as au
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import analysis.lib.quick_metadata_plots as mplts
import analysis.lib.analysis_utils as au
import padeopsIO as pio
import pandas as pd
import seaborn as sns
import streamtube
import itertools

# %%
from UnifiedMomentumModel.Momentum import UnifiedMomentum
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw

# %%
linewidth = 3.5
palette =['tab:orange', 'tab:green', 'tab:blue', 'tab:purple']

# %%
# %matplotlib inline

# %% [markdown]
# ## UMM Results

# %% [markdown]
# ### Get UMM Data

# %%
n_ct = 3
n_angles = 10
def get_umm_df(n_ct, n_angles):
    n_angles_sqrd = n_angles**2
    ctprimes = [0] * n_angles_sqrd * n_ct
    yaws = [0] * n_angles_sqrd * n_ct
    tilts = [0] * n_angles_sqrd * n_ct
    eff_angles = [0] * n_angles_sqrd * n_ct
    cps = [0] * n_angles_sqrd * n_ct
    ans = [0] * n_angles_sqrd * n_ct
    du4s = [0] * n_angles_sqrd * n_ct
    dv4s = [0] * n_angles_sqrd * n_ct
    dw4s = [0] * n_angles_sqrd * n_ct
    model = UnifiedMomentum()
    ct_yaw_tilt_combos = itertools.product((1, 1.33, 4), np.linspace(0, 30, num = n_angles), np.linspace(0, 30, num = n_angles))
    for (i, (ctp, y, t)) in enumerate(ct_yaw_tilt_combos):
        ctprimes[i] = ctp
        yaws[i] = y
        tilts[i] = t
        y_rad, t_rad = np.deg2rad(y), np.deg2rad(t)
        eff_angles[i] = np.rad2deg(calc_eff_yaw(y_rad, t_rad))
        sol = model(Ctprime = ctp, yaw = y_rad, tilt = t_rad)
        cps[i] = sol.Cp
        ans[i] = sol.an
        du4s[i] = 1 - sol.u4
        dv4s[i] = 0 - sol.v4
        dw4s[i] = 0 - sol.w4

    umm_data = {"Ctprime": ctprimes,
        "yaw": yaws,
        "tilt": tilts,
        "angle": eff_angles,
        "Cp": cps,
        "an": ans,
        "du4": du4s,
        "dv4": dv4s,
        "dw4": dw4s,
        }
    umm_df = pd.DataFrame(umm_data)
    umm_df = umm_df.sort_values(by=['Ctprime', 'angle', 'yaw'], ascending=True)
    return umm_df


# %%
umm_df = get_umm_df(n_ct, n_angles)
umm_df

# %%
limited_tilt_umm_df = umm_df[(umm_df["tilt"] % 10 == 0) & (umm_df["tilt"] <= 30)]
limited_yaw_umm_df = umm_df[(umm_df["yaw"] % 10 == 0) & (umm_df["yaw"] <= 30)]
limited_yaw_tilt_umm_df = limited_tilt_umm_df[(limited_tilt_umm_df["yaw"] % 10 == 0) & (limited_tilt_umm_df["yaw"] <= 30)]


# %% [markdown]
# ### Plot UMM Data

# %% [markdown]
# #### By Yaw and Colored by Tilt

# %%
def cp_plots(limited_tilt_umm_df, palette = palette):
    # cp_fig, (cp_ax0, cp_ax1, cp_ax2) = plt.subplots(3, 1, sharey = False, figsize = (3, 10), dpi = 300)
        
    label_size = 22
    title_size = 24

    cp_fig, cp_ax1 = plt.subplots(1, 1, sharey = False)
    # cp_fig.suptitle("Mean $C_p$ vs Yaw ($\gamma$) by Tilt ($\phi$):", size = title_size, y = 1.02)

    # cp_ax0.set_title("$C_T' = 1$", size = 14)
    # cp_ax0.set_ylabel('$C_P$', size = 14)
    # cp_ax0.set_xlabel('Yaw ($\gamma$)', size = 14)
    # cp_ax0.tick_params(axis='both', which='major', labelsize=12)
    # cp_ax0.set_ylim(0, 1)

    # cp_ax1.set_title("$C_T' = 1.33$", size = label_size)
    cp_ax1.set_ylabel('$C_P$', size = label_size)
    cp_ax1.set_xlabel('Yaw ($\gamma$)', size = label_size)
    cp_ax1.tick_params(axis='both', which='major', labelsize=label_size - 2)
    cp_ax1.set_ylim(0, 1)

    # cp_ax2.set_title("$C_T' = 4$", size = label_size)
    # cp_ax2.set_ylabel(' ')
    # cp_ax2.set_xlabel('Yaw ($\gamma$)', size = label_size)
    # cp_ax2.tick_params(axis='both', which='major', size = label_size - 2)
    # cp_ax2.set_ylim(0, 1)

    # sns.lineplot(ax = cp_ax0, data = limited_tilt_umm_df[limited_tilt_umm_df["Ctprime"] == 1],  x = "yaw", y = "Cp", hue = "tilt", linestyle = "dashed", linewidth = linewidth, palette = palette)
    sns.lineplot(ax = cp_ax1, data = limited_tilt_umm_df[limited_tilt_umm_df["Ctprime"] == 1.33],  x = "yaw", y = "Cp", hue = "tilt", linestyle = "dashed", linewidth = linewidth, palette = palette)#legend = False)
    # sns.lineplot(ax = cp_ax2, data = limited_tilt_umm_df[limited_tilt_umm_df["Ctprime"] == 4],  x = "yaw", y = "Cp", hue = "tilt", linestyle = "dashed", linewidth = linewidth, palette = palette, legend = False)

    leg = cp_ax1.legend(title='Tilt ($\phi$)')
    bb = leg.get_bbox_to_anchor().transformed(cp_ax1.transAxes.inverted())
    xOffset = 0.3
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform = cp_ax1.transAxes)
    # plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

    leg.get_title().set_fontsize(label_size)
    for text in leg.get_texts():
        text.set_fontsize(label_size - 2)

    cp_fig.set_size_inches(5.5, 5.5)
    cp_fig.set_dpi(300)

    # return cp_fig, (cp_ax0, cp_ax1, cp_ax2)
    return cp_fig, cp_ax1


# %%
cmap = sns.color_palette("Blues", as_cmap=True)
# Sample 4 darker shades (from 0.4 to 1.0)
dark_blues = [cmap(x) for x in [0.4, 0.6, 0.8, 1.0]]

cmap = sns.color_palette("Oranges", as_cmap=True)
# Sample 4 darker shades (from 0.4 to 1.0)
dark_oranges = [cmap(x) for x in [0.2, 0.4, 0.6, 0.8, 1.0]]

# %%
cp_plots(limited_tilt_umm_df, palette = dark_blues)


# %%
def plot_vels(limited_tilt_umm_df, limited_yaw_umm_df, ctprime = [1, 1.33, 4]):

    vels_fig0, axes0 = plt.subplots(1, 3, dpi = 300, sharey = True, figsize = (8, 4))
    vels_fig0.suptitle(f"$\delta u_4$ vs Yaw ($\gamma$) by Tilt ($\phi$)", size = 16, y = 1.04)
    for i, ax in enumerate(axes0):
        ctp = ctprime[i]
        ax.set_title(f"$C_T' = {ctp}$", size = 16)
        ax.set_xlabel('Yaw ($\\gamma$)', size = 16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        add_legend = True if i == 0 else False
        sns.lineplot(ax = ax, data = limited_tilt_umm_df[limited_tilt_umm_df["Ctprime"] == ctp],  x = "yaw", y = "du4", hue = "tilt", linestyle = "dashed", linewidth = linewidth, palette = palette, legend = add_legend)
        if add_legend:
            ax.set_ylabel('$\\delta u_4$', size = 16)
            ax.legend(title='Tilt ($\phi$)', title_fontsize=14, fontsize=12)
        else:
            ax.set_ylabel(' ')
    vels_fig0.subplots_adjust(wspace=0.25)

    vels_fig1, axes1 = plt.subplots(1, 3, dpi = 300, sharey = True, figsize = (8, 4))
    vels_fig1.suptitle(f"$\delta v_4$ vs Yaw ($\gamma$) by Tilt ($\phi$)", size = 16, y = 1.04)
    for i, ax in enumerate(axes1):
        ctp = ctprime[i]
        ax.set_title(f"$C_T' = {ctp}$", size = 16)
        ax.set_xlabel('Yaw ($\\gamma$)', size = 16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        add_legend = True if i == 0 else False
        sns.lineplot(ax = ax, data = limited_tilt_umm_df[limited_tilt_umm_df["Ctprime"] == ctp],  x = "yaw", y = "dv4", hue = "tilt", linestyle = "dashed", linewidth = linewidth, palette = palette, legend = add_legend)
        if add_legend:
            ax.set_ylabel('$\\delta v_4$', size = 16)
            ax.legend(title='Tilt ($\phi$)', title_fontsize=14, fontsize=12)
        else:
            ax.set_ylabel(' ')
    vels_fig1.subplots_adjust(wspace=0.25)

    vels_fig2, axes2 = plt.subplots(1, 3, dpi = 300, sharey = True, figsize = (8, 4))
    vels_fig2.suptitle(f"$\delta w_4$ vs Yaw ($\gamma$) by Tilt ($\phi$)", size = 16, y = 1.04)
    for i, ax in enumerate(axes2):
        ctp = ctprime[i]
        ax.set_title(f"$C_T' = {ctp}$", size = 16)
        ax.set_xlabel('Yaw ($\\gamma$)', size = 16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        add_legend = True if i == 0 else False
        sns.lineplot(ax = ax, data = limited_tilt_umm_df[limited_tilt_umm_df["Ctprime"] == ctp],  x = "yaw", y = "dw4", hue = "tilt", linestyle = "dashed", linewidth = linewidth, palette = palette, legend = add_legend)
        if add_legend:
            ax.set_ylabel('$\\delta w_4$', size = 16)
            ax.legend(title='Tilt ($\phi$)', title_fontsize=14, fontsize=12)
        else:
            ax.set_ylabel(' ')
    vels_fig2.subplots_adjust(wspace=0.25)

    vels_fig3, axes3 = plt.subplots(1, 3, dpi = 300, sharey = True, figsize = (8, 4))
    vels_fig3.suptitle(f"$\delta v_4$ vs Tilt ($\phi$) by Yaw ($\gamma$)", size = 16, y = 1.04)
    for i, ax in enumerate(axes3):
        ctp = ctprime[i]
        ax.set_title(f"$C_T' = {ctp}$", size = 16)
        ax.set_xlabel('Tilt ($\\phi$)', size = 16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        add_legend = True if i == 0 else False
        sns.lineplot(ax = ax, data = limited_yaw_umm_df[limited_yaw_umm_df["Ctprime"] == ctp],  x = "tilt", y = "dv4", hue = "yaw", linestyle = "dashed", linewidth = linewidth, palette = palette, legend = add_legend)
        if add_legend:
            ax.set_ylabel('$\\delta v_4$', size = 16)
            ax.legend(title='Yaw ($\gamma$)', title_fontsize=14, fontsize=12)
        else:
            ax.set_ylabel(' ')
    vels_fig3.subplots_adjust(wspace=0.25)

    vels_fig4, axes4 = plt.subplots(1, 3, dpi = 300, sharey = True, figsize = (8, 4))
    vels_fig4.suptitle(f"$\delta w_4$ vs Tilt ($\phi$) by Yaw ($\gamma$)", size = 16, y = 1.04)
    for i, ax in enumerate(axes4):
        ctp = ctprime[i]
        ax.set_title(f"$C_T' = {ctp}$", size = 16)
        ax.set_xlabel('Tilt ($\\phi$)', size = 16)
        ax.tick_params(axis='both', which='major', labelsize=12)
        add_legend = True if i == 0 else False
        sns.lineplot(ax = ax, data = limited_yaw_umm_df[limited_yaw_umm_df["Ctprime"] == ctp],  x = "tilt", y = "dw4", hue = "yaw", linestyle = "dashed", linewidth = linewidth, palette = palette, legend = add_legend)
        if add_legend:
            ax.set_ylabel('$\\delta w_4$', size = 16)
            ax.legend(title='Yaw ($\gamma$)', title_fontsize=14, fontsize=12)
        else:
            ax.set_ylabel(' ')
    vels_fig4.subplots_adjust(wspace=0.25)

    return (vels_fig0, axes0), (vels_fig1, axes1), (vels_fig2, axes2), (vels_fig3, axes3), (vels_fig4, axes4)


# %%
(vels_fig0, vels_ax0), (vels_fig1, vels_ax1), (vels_fig2, vels_ax2), (vels_fig3, axes3), (vels_fig4, axes4) = plot_vels(limited_tilt_umm_df, limited_yaw_umm_df)

# %% [markdown]
# ## Get LES PadéOps Data

# %%
data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "B_0001_Files")
rows, fields = mplts.get_sim_varied_params(sim_folder)
les_ids, les_cT, les_budget, les_yaw, les_tilt, les_filterWidth = zip(*rows)

# %%
run_folder = au.get_run_folder(sim_folder, 48)
sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")

# %%
ds = sim.slice(field_terms=["u", "v", "w", "x", "y", "z"], xlim = [-5, 20], ylim = [-2.5, 2.5], zlim = [-2.5, 2.5])
ds.u

# %%
ds = sim.slice(budget_terms=["ubar", "vbar", "wbar"], xlim = [-5, 20], ylim = [-2.5, 2.5], zlim = [-2.5, 2.5])
ds.ubar


# %%
def get_wake_vels(sim, filterwidth, budget):
    if budget:
        ds_budget = sim.slice(budget_terms=["ubar", "vbar", "wbar"], xlim = [-5, 20], ylim = [-2.5, 2.5], zlim = [-2.5, 2.5])
        u, v, w = ds_budget.ubar, ds_budget.vbar, ds_budget.wbar
        ds_field = sim.slice(field_terms=["x", "y", "z"], xlim = [-5, 20], ylim = [-2.5, 2.5], zlim = [-2.5, 2.5])
        x, y, z = ds_field.x, ds_field.y, ds_field.z
    else:
        ds = sim.slice(field_terms=["u", "v", "w", "x", "y", "z"], xlim = [-5, 20], ylim = [-2.5, 2.5], zlim = [-2.5, 2.5])
        x, y, z, u, v, w = ds.x, ds.y, ds.z, ds.u, ds.v, ds.w
    stream = streamtube.Streamtube(x, y, z, u, v, w)
    stream.compute_streamtube(R = 0.3)  # D/2 - filterwidth
    stream.compute_mask(R = 0.5 - filterwidth)
    dumax, dvmax, dwmax = 0, 0, 0
    for i in range(stream.mask.shape[0]):
        mask = stream.mask[i, :, :]
        umean = np.average(u[i, :, :], weights = mask)
        vmean = np.average(v[i, :, :], weights = mask)
        wmean = np.average(w[i, :, :], weights = mask)
        dumax = (1 - umean) if np.abs(1 - umean) > np.abs(dumax) else dumax
        dvmax = (0 - vmean) if np.abs(0 - vmean) > np.abs(dvmax) else dvmax
        dwmax = (0 - wmean) if np.abs(0 - wmean) > np.abs(dwmax) else dwmax
    return dumax, dvmax, dwmax


# %%
def get_les_df(CT_vals, budget_vals, yaw_vals, tilt_vals, filterwidth_vals):
    nruns = len(yaw_vals)
    ctprimes = [0] * nruns
    budgets = [True] * nruns
    yaws = [0] * nruns
    tilts = [0] * nruns
    eff_angles = [0] * nruns
    cps = [0] * nruns
    udisks = [0] * nruns
    du4s = [0] * nruns
    dv4s = [0] * nruns
    dw4s = [0] * nruns

    for i in range(0, nruns):
        run_folder = au.get_run_folder(sim_folder, i)
        sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")
        
        ctprimes[i] = float(CT_vals[i])
        yaws[i] = float(yaw_vals[i])
        tilts[i] = float(tilt_vals[i])
        y_rad, t_rad = np.deg2rad(yaws[i]), np.deg2rad(tilts[i])
        eff_angles[i] = np.rad2deg(calc_eff_yaw(y_rad, t_rad))

        mean_dt = sim.get_dt()
        power = sim.read_turb_power("all", turb=1)[round(100 / mean_dt):]
        udisk = sim.read_turb_uvel("all", turb=1)[round(100 / mean_dt):]
        avg_udisk = np.mean(udisk)
        udisks[i] = avg_udisk
        avg_power = np.mean(power)
        cps[i] = au.power_to_Cp(avg_power)
        budgets[i] = True if budget_vals[i] == "True" else False
        dumax, dvmax, dwmax = get_wake_vels(sim, float(filterwidth_vals[i]), budgets[i])
        du4s[i], dv4s[i], dw4s[i] = dumax, dvmax, dwmax


    les_data = {"Ctprime": ctprimes,
        "budget": budgets,
        "yaw": yaws,
        "tilt": tilts,
        "angle": eff_angles,
        "Cp": cps,
        "du4": du4s,
        "dv4": dv4s,
        "dw4": dw4s,
        "udisk": udisks,
        "an": 1 - np.array(udisks)
        }
    les_df = pd.DataFrame(les_data)
    les_df = les_df.sort_values(by=['Ctprime', 'angle', 'yaw'], ascending=True)
    return les_df

# %%
les_df = get_les_df(les_cT, les_budget, les_yaw, les_tilt, les_filterWidth)
les_df

# %%
stationary_case = les_df[(les_df.tilt == 0) & (les_df.yaw == 0) & (les_df.Ctprime == 1.33)]
stationary_case["udisk"]**2 * stationary_case.Ctprime

# %% [markdown]
# ### Plot LES Data

# %%
plt.rcParams["text.usetex"] = False

# %%
cp_fig, cp_ax1 = cp_plots(limited_tilt_umm_df,  palette = dark_blues)
handles_umm, labels_umm = cp_ax1.get_legend_handles_labels()

# sns.scatterplot(ax = cp_ax0, data = les_df[les_df["Ctprime"] == 1],  x = "yaw", y = "Cp", hue = "tilt", palette = dark_blues, s = 60, edgecolors = "k", zorder = 5)

# leg = cp_ax0.legend(title='Tilt ($\phi$)')
# bb = leg.get_bbox_to_anchor().transformed(cp_ax0.transAxes.inverted())
# xOffset = 0.5
# bb.x0 += xOffset
# bb.x1 += xOffset
# leg = leg.set_bbox_to_anchor(bb, transform = cp_ax0.transAxes)

sns.scatterplot(ax = cp_ax1, data = les_df[les_df["Ctprime"] == 1.33],  x = "yaw", y = "Cp", hue = "tilt", palette = dark_blues, s = 60, edgecolors = "k",  zorder = 5)#, legend = False)
# sns.scatterplot(ax = cp_ax2, data = les_df[(les_df["Ctprime"] == 4) & (les_df["budget"] == False)],  x = "yaw", y = "Cp", hue = "tilt", palette = dark_blues, s = 60, edgecolors = "k", legend = False,  zorder = 5)
handles_les, labels_les = cp_ax1.get_legend_handles_labels()

# Now remove duplicates — seaborn tends to accumulate handles
# So keep only unique ones (based on label text)
unique = dict(zip(labels_les, handles_les))
handles_all = list(unique.values())
labels_all = list(unique.keys())

# Build your combined legend manually
from matplotlib.legend import Legend

# Create dummy text entries for headers
import matplotlib.lines as mlines

umm_label = mlines.Line2D([], [], color='none', label='UMM', linewidth=0)
les_label = mlines.Line2D([], [], color='none', label='LES', linewidth=0)

# Combine with spacing
combined_handles = [umm_label] + handles_umm[:] + [les_label] + handles_les[4:]
combined_labels = [h.get_label() for h in combined_handles]

# Create legend
leg = cp_ax1.legend(combined_handles, combined_labels,
                    title = "Tilt ($\\phi$)",
                    handlelength=2.5, loc='center left',
                    frameon=False)

label_size = 22
title_size = 24

leg.get_title().set_fontweight('bold')
leg.get_title().set_fontsize(label_size)

# Make the headers bold (UMM and LES)
for text in leg.get_texts():
    if text.get_text() in ['UMM', 'LES']:
        text.set_fontweight('bold')
    text.set_fontsize(label_size - 2)

# Optional: make sure header text aligns cleanly
leg._legend_box.align = "left"

# Adjust legend box position if desired
bb = leg.get_bbox_to_anchor().transformed(cp_ax1.transAxes.inverted())
xOffset = 1
bb.x0 += xOffset
bb.x1 += xOffset
leg.set_bbox_to_anchor(bb, transform=cp_ax1.transAxes)

cp_ax1.set_title(cp_ax1.get_title(), fontsize=title_size)
cp_ax1.set_xlabel(cp_ax1.get_xlabel(), fontsize=label_size)
cp_ax1.set_ylabel(cp_ax1.get_ylabel(), fontsize=label_size)
cp_ax1.tick_params(axis='both', which='major', labelsize=label_size - 2)
# cp_fig._suptitle.set_fontsize(title_size + 2)
cp_ax1.set_ylim(0.25, 0.75)

# %%
tilt_only_umm = limited_tilt_umm_df[(limited_tilt_umm_df.Ctprime == 1.33) & (limited_tilt_umm_df.yaw == 0)]
tilt_only_umm["udisk"] = (1 - tilt_only_umm.an) * np.cos(np.deg2rad(tilt_only_umm.tilt))
tilt_only_les = les_df[(les_df.Ctprime == 1.33) & (les_df.yaw == 0)]
sns.scatterplot(data = tilt_only_les, x = "tilt", y = "udisk", color = "blue", label = "LES")
f = sns.scatterplot(data = tilt_only_umm, x = "tilt", y = "udisk", color = "orange", marker = "v", label = "UMM")
f.set_title("Static Tilt: $u_d$ vs $\phi$")
f.set_xlabel("Tilt ($\phi$)")
f.set_ylabel("$u_d$")

# %%
cp_fig, (cp_ax0, cp_ax1, cp_ax2) = cp_plots(limited_tilt_umm_df)

sns.scatterplot(ax = cp_ax0, data = les_df[les_df["Ctprime"] == 1],  x = "yaw", y = "an", hue = "tilt", palette = palette, s = 60, edgecolors = "k", zorder = 5)

leg = cp_ax0.legend(title='Tilt ($\phi$)')
bb = leg.get_bbox_to_anchor().transformed(cp_ax0.transAxes.inverted())
xOffset = 2.8
bb.x0 += xOffset
bb.x1 += xOffset
leg = leg.set_bbox_to_anchor(bb, transform = cp_ax0.transAxes)

sns.scatterplot(ax = cp_ax1, data = les_df[les_df["Ctprime"] == 1.33],  x = "yaw", y = "an", hue = "tilt", palette = palette, s = 60, edgecolors = "k", legend = False,  zorder = 5)
sns.scatterplot(ax = cp_ax2, data = les_df[(les_df["Ctprime"] == 4) & (les_df["budget"] == False)],  x = "yaw", y = "an", hue = "tilt", palette = palette, s = 60, edgecolors = "k", legend = False,  zorder = 5)

# %%
(vels_fig0, vels_ax0), (vels_fig1, vels_ax1), (vels_fig2, vels_ax2), (vels_fig3, vels_ax3), (vels_fig4, vels_ax4) = plot_vels(limited_tilt_umm_df, limited_yaw_umm_df)

ctprimes = [1, 1.33, 4]
for (i, ax) in enumerate(vels_ax0):
    legend = True if i == 0 else False
    budget = False if ctprimes[i] == 1 else True
    df = les_df[(les_df["Ctprime"] == ctprimes[i]) & (les_df["budget"] == budget)]
    sns.scatterplot(ax = ax, data = df,  x = "yaw", y = "du4", hue = "tilt", palette = palette, s = 60, edgecolors = "k", zorder = 5, legend = legend)
leg0 = vels_ax0[0].legend(title='Tilt ($\phi$)', title_fontsize=14, fontsize=12)
bb = leg0.get_bbox_to_anchor().transformed(vels_ax0[0].transAxes.inverted())
xOffset = 3.25
bb.x0 += xOffset
bb.x1 += xOffset
leg0 = leg0.set_bbox_to_anchor(bb, transform = vels_ax0[0].transAxes)

for (i, ax) in enumerate(vels_ax1):
    legend = True if i == 0 else False
    budget = False if ctprimes[i] == 1 else True
    df = les_df[(les_df["Ctprime"] == ctprimes[i]) & (les_df["budget"] == budget)]
    sns.scatterplot(ax = ax, data = df,  x = "yaw", y = "dv4", hue = "tilt", palette = palette, s = 60, edgecolors = "k", zorder = 5, legend = legend)
leg1 = vels_ax1[0].legend(title='Tilt ($\phi$)', title_fontsize=14, fontsize=12)
bb = leg1.get_bbox_to_anchor().transformed(vels_ax1[0].transAxes.inverted())
xOffset = 3.25
bb.x0 += xOffset
bb.x1 += xOffset
leg1 = leg1.set_bbox_to_anchor(bb, transform = vels_ax1[0].transAxes)
 
for (i, ax) in enumerate(vels_ax2):
    legend = True if i == 0 else False
    budget = False if ctprimes[i] == 1 else True
    df = les_df[(les_df["Ctprime"] == ctprimes[i]) & (les_df["budget"] == budget)]
    sns.scatterplot(ax = ax, data = df,  x = "yaw", y = "dw4", hue = "tilt", palette = palette, s = 60, edgecolors = "k", zorder = 5, legend = legend)
leg2 = vels_ax2[0].legend(title='Tilt ($\phi$)', title_fontsize=14, fontsize=12)
bb = leg2.get_bbox_to_anchor().transformed(vels_ax2[0].transAxes.inverted())
xOffset = 3.25
bb.x0 += xOffset
bb.x1 += xOffset
leg2 = leg2.set_bbox_to_anchor(bb, transform = vels_ax2[0].transAxes)

for (i, ax) in enumerate(vels_ax3):
    legend = True if i == 0 else False
    budget = False if ctprimes[i] == 1 else True
    df = les_df[(les_df["Ctprime"] == ctprimes[i]) & (les_df["budget"] == budget)]
    sns.scatterplot(ax = ax, data = df,  x = "tilt", y = "dv4", hue = "yaw", palette = palette, s = 60, edgecolors = "k", zorder = 5, legend = legend)
leg3 = vels_ax3[0].legend(title='Yaw ($\gamma$)', title_fontsize=14, fontsize=12)
bb = leg3.get_bbox_to_anchor().transformed(vels_ax3[0].transAxes.inverted())
xOffset = 3.25
bb.x0 += xOffset
bb.x1 += xOffset
leg3 = leg3.set_bbox_to_anchor(bb, transform = vels_ax3[0].transAxes)

for (i, ax) in enumerate(vels_ax4):
    legend = True if i == 0 else False
    budget = False if ctprimes[i] == 1 else True
    df = les_df[(les_df["Ctprime"] == ctprimes[i]) & (les_df["budget"] == budget)]
    sns.scatterplot(ax = ax, data = df,  x = "tilt", y = "dw4", hue = "yaw", palette = palette, s = 60, edgecolors = "k", zorder = 5, legend = legend)
leg4 = vels_ax4[0].legend(title='Yaw ($\gamma$)', title_fontsize=14, fontsize=12)
bb = leg4.get_bbox_to_anchor().transformed(vels_ax4[0].transAxes.inverted())
xOffset = 3.25
bb.x0 += xOffset
bb.x1 += xOffset
leg4 = leg4.set_bbox_to_anchor(bb, transform = vels_ax4[0].transAxes)

# %% [markdown]
# ## Quantify Error

# %%
compare_umm_df = limited_yaw_tilt_umm_df[limited_yaw_tilt_umm_df["Ctprime"] == 1]
compare_les_df = les_df[les_df["Ctprime"] == 1]

cp_err_vals = []
du4_err_vals = []
dv4_err_vals = []
dw4_err_vals = []
for row_idx, row in compare_umm_df.iterrows():
    y, t = row["yaw"], row["tilt"]
    les_row = compare_les_df[(compare_les_df["tilt"] == t) & (compare_les_df["yaw"] == y)]
    cp_err_vals.append(((row["Cp"] - les_row["Cp"]) / les_row["Cp"]).item() * 100)
    du4_err_vals.append(((row["du4"] - les_row["du4"]) / les_row["du4"]).item() * 100)
    if y == 0:
        dv4_err_vals.append(np.nan)
    else:
        dv4_err_vals.append(((row["dv4"] - les_row["dv4"]) / les_row["dv4"]).item() * 100)
    if t == 0:
        dw4_err_vals.append(np.nan)
    else:
        dw4_err_vals.append(((row["dw4"] - les_row["dw4"]) / les_row["dw4"]).item() * 100 )

compare_umm_df["cp_err"] = cp_err_vals
compare_umm_df["du4_err"] = du4_err_vals
compare_umm_df["dv4_err"] = dv4_err_vals
compare_umm_df["dw4_err"] = dw4_err_vals


# %%
def plot_error_barplot(data, err_col, err_name):
    fig, ax0 = plt.subplots(dpi = 300)
    ax0.set_title(f"{err_name} vs Yaw ($\gamma$) by Tilt ($\phi$) for $C_T' = 1$", size = 16)
    ax0.set_ylabel(f"{err_name} (%)", size = 16)
    ax0.set_xlabel('Yaw ($\\gamma$)', size = 16)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    sns.barplot(data, x="yaw", y=err_col, hue="tilt", palette = palette)
    ax0.legend(title='Tilt ($\phi$)', title_fontsize=14, fontsize=12)
    return


# %%
plot_error_barplot(compare_umm_df, "cp_err", "$C_p$ Error")

# %%
plot_error_barplot(compare_umm_df, "du4_err", "$\delta u_4$ Error")

# %%
plot_error_barplot(compare_umm_df, "dv4_err", "$\delta v_4$ Error")

# %%
plot_error_barplot(compare_umm_df, "dw4_err", "$\delta w_4$ Error")

# %%
for ctp in [1]:  # note that compare_umm_df only has CT' = 1 data in it
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey = True, sharex = True, figsize = (10, 4), dpi = 300)
    fig.suptitle(f"Velocity Deficit Errors vs Effective Angle for $C_T' = {ctp}$", size = 18)

    ax0.set_ylabel("$\delta u_4$ Error (%)", size = 16)
    ax1.set_ylabel("$\delta v_4$ Error (%)", size = 16)
    ax2.set_ylabel("$\delta w_4$ Error (%)", size = 16)

    ax0.set_xlabel('Effective Angle', size = 16)
    ax1.set_xlabel('Effective Angle', size = 16)
    ax2.set_xlabel('Effective Angle', size = 16)

    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    sns.scatterplot(ax = ax0, data = compare_umm_df, x="angle", y="du4_err", hue="yaw", style = "tilt", palette = palette, s = 80)
    sns.scatterplot(ax = ax1, data = compare_umm_df, x="angle", y="dv4_err", hue="yaw", style = "tilt", palette = palette, legend = False, s = 100)
    sns.scatterplot(ax = ax2, data = compare_umm_df, x="angle", y="dw4_err", hue="yaw", style = "tilt", palette = palette, legend = False, s = 100)
    leg = ax0.legend(title_fontsize=14, fontsize=12)
    bb = leg.get_bbox_to_anchor().transformed(ax0.transAxes.inverted())
    xOffset = 2.9
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg = leg.set_bbox_to_anchor(bb, transform = ax0.transAxes)
