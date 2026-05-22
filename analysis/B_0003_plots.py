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

from mitwindfarm import Uniform, Layout
from mitwindfarm.windfarm import Windfarm, CurledWindfarm
from mitwindfarm.Rotor import UnifiedAD_TI
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw, eff_yaw_inv_rotation
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from matplotlib.lines import Line2D


# %%
def umm_vs_les_plots(xlim, ylim, zlim):
    sim_B3_all_folder = os.path.join(au.DATA_PATH, "B_0003_Files")
    n_sims = 3
    n1, n2 = 0, 0
    for i in range(n_sims):
        try:
            run_folder = au.get_run_folder(sim_B3_all_folder, i)
            sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")
            ds = sim.slice(budget_terms=['ubar'], xlim=xlim, ylim=ylim, zlim=zlim)
            if i == 0:
                (n1, n2) = np.shape(ds.ubar)
                les_ubar_data = np.ones((n2, n1, n_sims))
            les_ubar_data[:, :, i] = ds.ubar.T
        except:
            continue

    base_windfield = Uniform(TIamb=0.035)  # 3.5% ambient TI
    wf_curled = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=base_windfield,
        solver_kwargs=dict(
            dy=1/10,
            dz=1/10,
            integrator="scipy_rk23",  # see mitwindfarm.utils.integrate
            k_model="k-l",  # alternatives: "const", "2021"
            verbose=False,
            use_r4 = False,
        ),
    )
    wf_gauss = Windfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=base_windfield,
        TIamb= 0.035
    )
    layout = Layout([0], [0], [0])

    yaw3 = np.radians(20)
    tilt3 = np.radians(20)
    eff_yaw = calc_eff_yaw(yaw3, tilt3)
    yaws, tilts = [eff_yaw, 0, yaw3], [0, eff_yaw, tilt3]
    # get grid points
    if len(np.shape(xlim)) == 0:
        ny, nz = n1, n2
        _y, _z = np.linspace(*ylim, ny), np.linspace(*zlim, nz)
        ymesh, zmesh = np.meshgrid(_y, _z)
        xmesh = np.full_like(ymesh, xlim)
        extent=[*ylim, *zlim]
        xlabel = f"$y/D$"
        ylabel = f"$z/D$"
        filename = "/UMM_LES_validation_yz.png"
        title_num = xlim
        title_str = "x"
        figsize=(10, 10)
        y_offset = 0.025
    elif len(np.shape(ylim)) == 0:
        nx, nz = n1, n2
        _x, _z = np.linspace(*xlim, nx), np.linspace(*zlim, nz)
        xmesh, zmesh = np.meshgrid(_x, _z)
        ymesh = np.full_like(zmesh, ylim)
        extent=[*xlim, *zlim]
        xlabel = f"$x/D$"
        ylabel = f"$z/D$"
        filename = "/UMM_LES_validation_xz.png"
        title_num = ylim
        title_str = "y"
        figsize=(10, 6)
        y_offset = 0.05
    else:
        nx, ny = n1, n2
        _x, _y = np.linspace(*xlim, nx), np.linspace(*ylim, ny)
        xmesh, ymesh = np.meshgrid(_x, _y)
        zmesh = np.full_like(ymesh, zlim)
        extent=[*xlim, *ylim]
        xlabel = f"$x/D$"
        ylabel = f"$y/D$"
        filename = "/UMM_LES_validation_xy.png"
        title_num = zlim
        title_str = "z"
        figsize=(10, 6)
        y_offset = 0.05
    # get solutions to all set point combos
    curled_umm_ubar = []
    gauss_umm_ubar = []
    ctprime = 1.33
    for yaw, tilt in zip(yaws, tilts):
        setpoints = [(ctprime, yaw, tilt)]
        sol_c = wf_curled(layout, setpoints)
        wsp_c = sol_c.windfield.wsp(xmesh, ymesh, zmesh)
        curled_umm_ubar.append(wsp_c)

        sol_g = wf_gauss(layout, setpoints)
        wsp_g = sol_g.windfield.wsp(xmesh, ymesh, zmesh)
        gauss_umm_ubar.append(wsp_g)

    fig, axes = plt.subplots(3, 3, figsize=figsize, dpi = 300)
    fig.suptitle(f"Validation of Curled Wake Model Under Tilt and Yaw at ${title_str} = {title_num}D$ with $3.5\\%$ TI\n $C_T' = 1.33$, $\lambda = 7$, and $\\theta_p = 0$", size = 14)
    # Flatten the axes array for easy iteration
    (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = axes.ravel()
    # (ax0, ax1, ax2, ax3, ax4, ax5) = axes.ravel()

    # Plot each dataset with imshow, keeping track of the last image
    les_vmin = min(d.min() for d in les_ubar_data)
    les_vmax = max(d.max() for d in les_ubar_data)
    # Plot each dataset with imshow, keeping track of the last image
    umm_vmin = min(d.min() for d in curled_umm_ubar)
    umm_vmax = max(d.max() for d in curled_umm_ubar)

    vmin = np.minimum(les_vmin, umm_vmin)
    vmax = np.minimum(les_vmax, umm_vmax)

    ax0.set_title(r"LES")
    cmap = "Blues_r"
    im = ax0.imshow(les_ubar_data[:, :, 0], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")
    ax1.set_title(r"LES")
    im = ax1.imshow(les_ubar_data[:, :, 1], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")
    ax2.set_title(r"LES")
    im = ax2.imshow(les_ubar_data[:, :, 2], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")
    ax3.set_title(r"Curled Wake")
    im = ax3.imshow(curled_umm_ubar[0], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")
    ax4.set_title(r"Curled Wake")
    im = ax4.imshow(curled_umm_ubar[1], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")
    ax5.set_title(r"Curled Wake")
    im = ax5.imshow(curled_umm_ubar[2], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")

    ax6.set_title(r"Gaussian Wake")
    im = ax6.imshow(gauss_umm_ubar[0], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")
    ax7.set_title(r"Gaussian Wake")
    im = ax7.imshow(gauss_umm_ubar[1], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")
    ax8.set_title(r"Gaussian Wake")
    im = ax8.imshow(gauss_umm_ubar[2], vmin=vmin, vmax=vmax, cmap=cmap, extent=extent, origin="lower")

    ax0.set_ylabel(ylabel)
    ax3.set_ylabel(ylabel)
    # ax3.set_xlabel(xlabel)
    # ax4.set_xlabel(xlabel)
    # ax5.set_xlabel(xlabel)
    ax6.set_ylabel(ylabel)
    ax6.set_xlabel(xlabel)
    ax7.set_xlabel(xlabel)
    ax8.set_xlabel(xlabel)

    fig.subplots_adjust(left=0.1, right=0.9, hspace=0.1, wspace=0.3)
    fig.canvas.draw()

    # === Compute figure coordinates ===
    bbox_left = axes[0, 0].get_position()
    x_left = bbox_left.x0
    x_label = x_left - 0.08  # further left than y-labels

    # --- Top row: LES ---
    y_top_row = axes[0, 0].get_position()
    y_top_center = y_top_row.y0 + y_top_row.height / 2

    fig.text(x_label, y_top_center, "LES",
            va="center", ha="center", rotation="vertical",
            fontsize=14, fontweight="bold")

    # --- Bottom two rows: MITWindFarm ---
    y_row2 = axes[1, 0].get_position()
    y_row3 = axes[2, 0].get_position()

    y_bottom = y_row3.y0
    y_top = y_row2.y0 + y_row2.height
    y_center = (y_bottom + y_top) / 2

    fig.text(x_label, y_center, "MITWindFarm",
            va="center", ha="center", rotation="vertical",
            fontsize=14, fontweight="bold")
    
# === Add COLUMN labels above each column ===
    def add_column_label(label, col_axes, y_offset=0.025):
        """
        label: text to display
        col_axes: list of axes objects in this column (e.g., axes[:, j])
        y_offset: fraction of figure height to offset above the top row
        """
        # Get horizontal center of column
        x0 = col_axes[0].get_position().x0
        x1 = col_axes[0].get_position().x0 + col_axes[0].get_position().width
        x_center = (x0 + x1) / 2

        # Get top y-position of top subplot in this column
        y_top = col_axes[0].get_position().y0 + col_axes[0].get_position().height
        y_label = y_top + y_offset

        # Place text slightly above
        fig.text(x_center,y_label, label,
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        

#     # Example column labels
    col_labels = ["$28 ^\circ$ Yaw", "$28 ^\circ$ Tilt", "$20 ^\circ$ Yaw & Tilt"]

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.05)
    cbar.set_label(r"$\overline{u}/U$", rotation=90, labelpad=15)

    for j, label in enumerate(col_labels):
        add_column_label(label, axes[:, j], y_offset = y_offset)

    filename = sim_B3_all_folder + filename
    fig.savefig(filename, dpi = 300)

# %%
xlim, ylim, zlim = 4, (-2, 2), (-2, 2)
umm_vs_les_plots(xlim, ylim, zlim)

# %%
xlim, ylim, zlim = (-5, 15), 0, (-2, 2)
umm_vs_les_plots(xlim, ylim, zlim)

# %%
xlim, ylim, zlim = (-5, 15), (-2, 2), 0
umm_vs_les_plots(xlim, ylim, zlim)


# %%
def umm_vs_les_poster_plots():
    xlim = [-2, 15]
    ylim = [-2, 2]
    zlim = [-2, 2]
    sim_B3_all_folder = os.path.join(au.DATA_PATH, "B_0003_Files")
    run_folder = au.get_run_folder(sim_B3_all_folder, 2)
    sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")

    ds_xy = sim.slice(budget_terms=['ubar'], xlim=xlim, ylim=ylim, zlim=0)
    (n1, n2) = np.shape(ds_xy.ubar)
    les_ubar_xy_data = ds_xy.ubar.T

    ds_xz = sim.slice(budget_terms=['ubar'], xlim=xlim, ylim=0, zlim=zlim)
    les_ubar_xz_data = ds_xz.ubar.T


    base_windfield = Uniform(TIamb=0.035)  # 3.5% ambient TI
    wf_curled = CurledWindfarm(
        rotor_model=UnifiedAD_TI(),
        base_windfield=base_windfield,
        solver_kwargs=dict(
            dy=1/10,
            dz=1/10,
            integrator="scipy_rk23",  # see mitwindfarm.utils.integrate
            k_model="k-l",  # alternatives: "const", "2021"
            verbose=False,
            use_r4 = False,
        ),
    )
    
    layout = Layout([0], [0], [0])
    yaw = np.radians(20)
    tilt = np.radians(20)
    ctprime = 1.33
    setpoints = [(ctprime, yaw, tilt)]
    sol_c = wf_curled(layout, setpoints)

    _x, _y = np.linspace(*xlim, n1), np.linspace(*ylim, n2)
    xmesh, ymesh = np.meshgrid(_x, _y)
    zmesh = np.full_like(ymesh, 0)
    curled_umm_ubar_xy = sol_c.windfield.wsp(xmesh, ymesh, zmesh)

    _x, _z = np.linspace(*xlim, n1), np.linspace(*zlim, n2)
    xmesh, zmesh = np.meshgrid(_x, _z)
    ymesh = np.full_like(zmesh, 0)
    curled_umm_ubar_xz = sol_c.windfield.wsp(xmesh, ymesh, zmesh)

    fig, axes = plt.subplots(2, 2, figsize=(10, 4.25), dpi = 300)
    # Flatten the axes array for easy iteration
    (ax0, ax1, ax2, ax3) = axes.ravel()

    # Plot each dataset with imshow, keeping track of the last image
    les_vmin = np.minimum(np.min(les_ubar_xy_data), np.min(les_ubar_xz_data))
    les_vmax = np.maximum(np.max(les_ubar_xy_data), np.max(les_ubar_xz_data))
    # Plot each dataset with imshow, keeping track of the last image
    umm_vmin = np.minimum(np.min(curled_umm_ubar_xy), np.min(curled_umm_ubar_xz))
    umm_vmax = np.maximum(np.max(curled_umm_ubar_xy), np.max(curled_umm_ubar_xz))

    vmin = np.minimum(les_vmin, umm_vmin)
    vmax = np.minimum(les_vmax, umm_vmax)
    cmap = "Blues_r"
    
    label_size = 22
    title_size = 24

    # fig.suptitle(f"Validation of MITWindFarm with $20^\\circ$ Yaw and Tilt\n $C_T' = 1.33$, $\lambda = 7$, and $\\theta_p = 0$", size = title_size + 2)
    ax0.set_title(r"MITWindFarm", size = title_size)
    ax0.set_ylabel(f"$y/D$", size = label_size)
    # ax0.set_xlabel(f"$x/D$", size = label_size)
    im = ax0.imshow(curled_umm_ubar_xy, vmin=vmin, vmax=vmax, cmap=cmap, extent=[*xlim, *ylim], origin="lower")
    ax0.tick_params(axis='both', which='major', labelsize=label_size - 2)

    ax1.set_title(r"LES", size = title_size)
    # ax1.set_ylabel(f"$y/D$", size = label_size)
    # ax1.set_xlabel(f"$x/D$", size = label_size)
    im = ax1.imshow(les_ubar_xy_data, vmin=vmin, vmax=vmax, cmap=cmap,  extent=[*xlim, *ylim], origin="lower")
    ax1.tick_params(axis='both', which='major', labelsize=label_size - 2)

    ax2.set_title(r"MITWindFarm", size = title_size)
    ax2.set_ylabel(f"$z/D$", size = label_size)
    ax2.set_xlabel(f"$x/D$", size = label_size)
    im = ax2.imshow(curled_umm_ubar_xz, vmin=vmin, vmax=vmax, cmap=cmap,  extent=[*xlim, *zlim], origin="lower")
    ax2.tick_params(axis='both', which='major', labelsize=label_size - 2)

    ax3.set_title(r"LES", size = title_size)
    ax3.set_ylabel(f"$z/D$", size = label_size)
    ax3.set_xlabel(f"$x/D$", size = label_size)
    im = ax3.imshow(les_ubar_xz_data, vmin=vmin, vmax=vmax, cmap=cmap,  extent=[*xlim, *zlim], origin="lower")
    ax3.tick_params(axis='both', which='major', labelsize=label_size - 2)

    fig.subplots_adjust(left=0.1, right=0.9, hspace=0.4, wspace=0.3)

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.05)
    cbar.set_label(r"$\overline{u}/U$", rotation=90, labelpad=15, size = label_size)
    cbar.ax.tick_params(labelsize=label_size - 2)

    fig.set_size_inches(14, 5.5)
    fig.set_dpi(300)

    filename = sim_B3_all_folder + "/poster_mitwindfarm.jpg"
    fig.savefig(filename, dpi = 300)

# %%
umm_vs_les_poster_plots()

# %%
