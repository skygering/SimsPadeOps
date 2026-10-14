# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: simspadeops-YbNxeWqo-py3.11
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import os
from pathlib import Path
import padeopsIO as pio
import numpy as np
import pandas as pd
import math
from UnifiedMomentumModel.Momentum import UnifiedMomentum
import seaborn as sns
from matplotlib.lines import Line2D

# %%
import lib.quick_metadata_plots as mplts
import lib.analysis_utils as au

# %%
data_path = Path(au.DATA_PATH)
sim_folder = os.path.join(au.DATA_PATH, "F_0035_Files")

rows, fields = mplts.get_sim_varied_params(sim_folder)
ids,surge_freq,cT,useCTprime,use_cT_timeseries,cT_timeseries_file = zip(*rows)

# %%
surge_freq = [float(f) if i < 5 else 0.0 for i, f in enumerate(surge_freq)]

# %%
domain_file = Path("/scratch/10264/sgering/SimsPadeOps/simulations/F_0029_SU.csv")
df_dom = pd.read_csv(domain_file)
nx, ny, nz = np.unique(df_dom["nx"]), np.unique(df_dom["ny"]), np.unique(df_dom["nz"])
Lx, Ly, Lz = np.unique(df_dom["Lx"]), np.unique(df_dom["Ly"]), np.unique(df_dom["Lz"])
dt_solver = float(np.unique(df_dom["dt"])[0])

h = np.sqrt((Lx[0] / nx[0])**2 + (Ly[0] / ny[0])**2 + (Lz[0] / nz[0])**2)
filterWidth = 1.5 * h
useCorrection = True

# %%
arr = np.asarray(useCTprime)              # list/series -> numpy array
norm = np.char.lower(np.char.strip(arr.astype(str))) # normalize case/whitespace
useCTprime = norm == "true"  

arr = np.asarray(use_cT_timeseries)              # list/series -> numpy array
norm = np.char.lower(np.char.strip(arr.astype(str))) # normalize case/whitespace
use_cT_timeseries = norm == "true"  

# %%
ct_timeseries_df = pd.read_csv(
    "/scratch/10264/sgering/DataPadeOps/F_0035_Files/timeseries_ct_sin.csv",
    delimiter = " ",
    header=None,
    names=["Time", "CT"]
)

ct_prime_timeseries_df = pd.read_csv(
    "/scratch/10264/sgering/DataPadeOps/F_0035_Files/timeseries_ctp_sin.csv",
    delimiter = " ",
    header=None,
    names=["Time", "CT_prime"]
)

# %%
fig_ct, ax_ct = plt.subplots(figsize=(10, 5))
fig_cp, ax_cp = plt.subplots(figsize=(10, 5))


# %%

# -------------------------------------------------------------------
# Assumes these already exist in your environment:
#   - au (analysis utils with get_run_folder/load_simulation_timeseries)
#   - _to_bool
#   - ids, f, useCTprime, use_cT_timeseries, cT_timeseries_file
#   - sim_35_all_folder, transient_time, Delta, UInf, rho, D, dpi, tidx_min
# -------------------------------------------------------------------

def shapiro_correct_ct_mode(
    u_les,
    CT,
    Delta,
    R,
    Uinf=1.0,
    eps=1e-12,
    clip_discriminant=False
):
    """
    Self-consistent Shapiro post-correction for CT-mode runs.

    Solves:
        u_corr = M * u_les
        M = 1 / (1 + beta * CT')
        CT' = CT * Uinf^2 / u_corr^2
        beta = Delta / (4 * R * sqrt(3*pi))

    => quadratic:
        u_corr^2 - u_les*u_corr + beta*(CT*Uinf^2) = 0

    Physical root:
        u_corr = 0.5 * (u_les + sqrt(u_les^2 - 4*beta*CT*Uinf^2))
    """
    u0 = np.asarray(u_les, dtype=float)
    scalar_in = (u0.ndim == 0)
    u0a = np.atleast_1d(u0)

    beta = Delta / (4.0 * R * np.sqrt(3.0 * np.pi))
    K = np.asarray(CT, dtype=float) * (Uinf ** 2)  # scalar or array CT

    disc = u0a**2 - 4.0 * beta * K
    if clip_discriminant:
        disc_eff = np.maximum(disc, 0.0)
        valid = np.ones_like(disc_eff, dtype=bool)
    else:
        disc_eff = disc
        valid = (disc_eff >= 0.0)

    u_corr = np.full_like(u0a, np.nan, dtype=float)
    u_corr[valid] = 0.5 * (u0a[valid] + np.sqrt(disc_eff[valid]))  # + root

    CTprime_corr = K / np.maximum(u_corr**2, eps)
    M_corr = np.where(np.abs(u0a) > eps, u_corr / u0a, np.nan)

    if scalar_in:
        return u_corr.item(), CTprime_corr.item(), M_corr.item()
    return u_corr, CTprime_corr, M_corr


def _norm(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    if s < eps:
        return x - np.nanmean(x)
    return (x - np.nanmean(x)) / s


def make_group_plots(
    group_name,
    moving_flag,  # False=stationary, True=surging
    ids, f, useCTprime, use_cT_timeseries, cT_timeseries_file,
    sim_35_all_folder, transient_time,
    Delta, UInf, rho, D, dpi, tidx_min
):
    # Group summary figures
    fig_u, ax_u = plt.subplots(figsize=(10, 4), dpi=dpi)
    fig_ctp, ax_ctp = plt.subplots(figsize=(10, 4), dpi=dpi)
    fig_cp, ax_cp = plt.subplots(figsize=(10, 4), dpi=dpi)
    fig_ct, ax_ct = plt.subplots(figsize=(10, 4), dpi=dpi)

    plotted_any = False
    out = Path(sim_35_all_folder)

    for i, id_str in enumerate(ids):
        freq = 0.0 if i == 5 else float(f[i])  # your special-case override
        moving_turbine = (freq != 0.0)

        # split by stationary/surging
        if moving_turbine != moving_flag:
            continue

        useCTP = useCTprime[i]
        ct_timeseries = use_cT_timeseries[i]
        ct_file = cT_timeseries_file[i]

        run_folder = au.get_run_folder(sim_35_all_folder, i)
        data = au.load_simulation_timeseries(
            run_folder, id_str, freq, spinup_time=transient_time
        )

        time = data["Time"]
        tidx = data["TIDX"]
        uturb = data["UTurb"]
        udisk = data["UDisk"]
        power = data["Power"]

        # Build CT / CT' and apply post-correction only in CT mode
        if ct_timeseries:
            df = pd.read_csv(ct_file, header=None, delimiter=",")
            full_df = df[1].to_numpy(dtype=float)
            n = len(time)
            cts = full_df[-n:]

            if useCTP:
                ct_label = r"Sinusoidal Timeseries $C_T'$ where $\overline{C_T'} = 1.66$"
                ctp = cts
                ct = ctp * (udisk**2) / (UInf**2)
                linestyle = "--"
            else:
                ct_label = r"Sinusoidal Timeseries $C_T$ where $\overline{C_T} = 0.829$"
                ct = cts
                udisk, ctp, M = shapiro_correct_ct_mode(udisk, ct, Delta, R=0.5, Uinf=UInf)
                power = M * power
                linestyle = "-"
        else:
            if useCTP:
                ctp = np.full_like(udisk, 2.0, dtype=float)
                ct_label = r"Constant $C_T' = 2$"
                ct = ctp * (udisk**2) / (UInf**2)
                linestyle = "--"
            else:
                ct = np.full_like(udisk, 8/9, dtype=float)
                ct_label = r"Constant $C_T = 8/9$"
                udisk, ctp, M = shapiro_correct_ct_mode(udisk, ct, Delta, R=0.5, Uinf=UInf)
                power = M * power
                linestyle = "-"

        cp = power / (0.5 * rho * math.pi * (D/2.0)**2 * (UInf**3))

        surge_label = "Sinusoidal Surge" if moving_turbine else "Stationary"
        label = f"{surge_label},\n {ct_label}"

        # Group summary lines
        ax_u.plot(tidx, uturb, label=label, linestyle=linestyle, linewidth=2)
        ax_ctp.plot(tidx, ctp, label=label, linestyle=linestyle, linewidth=2)
        ax_cp.plot(tidx, cp, label=label, linestyle=linestyle, linewidth=2)
        ax_ct.plot(tidx, ct, label=label, linestyle=linestyle, linewidth=2)

        plotted_any = True

        # -----------------------------------------------------------
        # Per-index alignment plot: UTurb, UDisk, CT on same axis
        # (normalized so phase alignment is easy to see)
        # -----------------------------------------------------------
        fig_align, ax_align = plt.subplots(figsize=(11, 4), dpi=dpi)
        ax_align.plot(tidx, _norm(uturb), lw=2.0, label="UTurb (norm)", color="C0")
        ax_align.plot(tidx, _norm(udisk), lw=2.0, label="UDisk (norm)", color="C1")
        ax_align.plot(tidx, _norm(ct),    lw=2.0, label="CT (norm)",    color="C2", linestyle="--")
        ax_align.set_xlabel("TIDX")
        ax_align.set_ylabel("Normalized value")
        ax_align.set_title(f"{group_name} alignment: motion, disk velocity, CT\n{id_str}")
        ax_align.grid(alpha=0.25)
        ax_align.legend(loc="upper left", ncol=3)
        fig_align.tight_layout()

        tag = "surging" if moving_flag else "stationary"
        fig_align.savefig(out / f"sim34_alignment_{tag}_{i:02d}_{id_str}.png",
                          dpi=dpi, bbox_inches="tight")
        plt.close(fig_align)

    def finish_axis(ax, ylabel, title):
        ax.set_xlabel("TIDX")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        if plotted_any:
            ax.legend(title="Simulation", loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    finish_axis(ax_u, "UTurb", f"{group_name}: Turbine velocity UTurb (TIDX > 15000)")
    finish_axis(ax_ctp, "CT'", f"{group_name}: CT' (TIDX > 15000)")
    finish_axis(ax_cp, "Cp", f"{group_name}: Turbine Cp (TIDX > {tidx_min})")
    finish_axis(ax_ct, "Ct", f"{group_name}: Turbine Ct (TIDX > {tidx_min})")

    fig_u.tight_layout()
    fig_ctp.tight_layout()
    fig_cp.tight_layout()
    fig_ct.tight_layout()

    tag = "surging" if moving_flag else "stationary"
    fig_u.savefig(out / f"sim34_UTurb_{tag}.png", dpi=dpi, bbox_inches="tight")
    fig_ctp.savefig(out / f"sim34_CTprime_{tag}.png", dpi=dpi, bbox_inches="tight")
    fig_cp.savefig(out / f"sim34_Cp_{tag}.png", dpi=dpi, bbox_inches="tight")
    fig_ct.savefig(out / f"sim34_Ct_{tag}.png", dpi=dpi, bbox_inches="tight")

    plt.close(fig_u)
    plt.close(fig_ctp)
    plt.close(fig_cp)
    plt.close(fig_ct)


# -------------------------
# Run both categories
# -------------------------
UInf = 1.0
rho = 1.0 # TODO: check
transient_time = 75
D = 1
dpi = 300
tidx_min = 15000
make_group_plots(
    "Stationary", False,
    ids, surge_freq, useCTprime, use_cT_timeseries, cT_timeseries_file,
    sim_folder, transient_time,
    filterWidth, UInf, rho, D, dpi, tidx_min
)

make_group_plots(
    "Surging", True,
    ids, surge_freq, useCTprime, use_cT_timeseries, cT_timeseries_file,
    sim_folder, transient_time,
    filterWidth, UInf, rho, D, dpi, tidx_min
)

print(f"Saved grouped plots in: {Path(sim_folder)}")
