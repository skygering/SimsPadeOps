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

sim_folder = os.path.join(au.DATA_PATH, "F_0038_SU_Files")
metadata_df = pd.read_csv("/scratch/10264/sgering/SimsPadeOps/simulations/F_0038_metadata.csv")
# File has NO header row; assign your own column names
ctp_timeseries_df = pd.read_csv(
    "/scratch/10264/sgering/DataPadeOps/F_0038_SU_Files/timeseries_cT_blend.csv",
    delimiter = " ",
    header=None,
    names=["Time", "CT_prime"]
)

# %%
rows, fields = mplts.get_sim_varied_params(sim_folder)
ids,_,use_motion_timeseries,use_cT_timeseries = zip(*rows)

# %%
arr = np.asarray(use_motion_timeseries)              # list/series -> numpy array
norm = np.char.lower(np.char.strip(arr.astype(str))) # normalize case/whitespace
use_motion_timeseries = norm == "true"  

arr = np.asarray(use_cT_timeseries)              # list/series -> numpy array
norm = np.char.lower(np.char.strip(arr.astype(str))) # normalize case/whitespace
use_cT_timeseries = norm == "true"  

# %%
freq = metadata_df["v_f"]

# %%
re_film = False
if re_film: 
    for i, id_str in enumerate(ids):
        print(id_str)
        run_folder = os.path.join(sim_folder, f"Sim_{id_str}")
        run_folder = mplts.plot_instantaneous_field(sim_folder, i, tidx = "all", field = "u", dpi = 300)
        mplts.film_instantaneous_field(run_folder, fps = 10, video_name = "u_stability.mp4")

# %%
fig_power, ax_power = plt.subplots(figsize=(10, 5))
fig_udisk, ax_udisk = plt.subplots(figsize=(10, 5))
fig_ut, ax_ut = plt.subplots(figsize=(10, 5))

base_ut = None
base_udisk = None
base_power = None

all_df = []
for i, id_str in enumerate(ids):
    run_folder = os.path.join(sim_folder, f"Sim_{id_str}")
    data = au.load_simulation_timeseries(run_folder, id_str, freq[0], spinup_time=75)

    t = data["Time"]
    ut = data["UTurb"]
    p = data["Power"]
    ud = data["UDisk"]

    mts = use_motion_timeseries[i]
    ctts = use_cT_timeseries[i]
    label = f"ID {id_str} | motion_ts={mts}, cT_ts={ctts}"

    df = pd.DataFrame({
        "id": i,
        "use_motion_timeseries": mts,
        "use_cT_timeseries": ctts,
        **data
    })
    all_df.append(df)

    if i == 0:
        base_ut = data["UTurb"]
        base_udisk = data["UDisk"]
        base_power = data["Power"]

    ax_power.plot(t, p - base_power, label=label, lw=1.5)
    ax_udisk.plot(t, ud - base_udisk, label=label, lw=1.5)
    ax_ut.plot(t, ut - base_ut, label=label, lw=1.5)

all_df = pd.concat(all_df, ignore_index=True)

# Configure all axes BEFORE showing
ax_power.set_title("Power by simulation")
ax_power.set_xlabel("Time")
ax_power.set_ylabel("Power")
ax_power.set_xlim(85, 90)
ax_power.legend(fontsize=8, ncol=2)

ax_udisk.set_title("UDisk by simulation")
ax_udisk.set_xlabel("Time")
ax_udisk.set_ylabel("UDisk")
ax_udisk.legend(fontsize=8, ncol=2)

ax_ut.set_title("UTurb by simulation")
ax_ut.set_xlabel("Time")
ax_ut.set_ylabel("UTurb")
ax_ut.legend(fontsize=8, ncol=2)

# Tight layout per figure
fig_power.tight_layout()
fig_udisk.tight_layout()
fig_ut.tight_layout()

# Show once
plt.show()

# %%
UInf = 1
D = 1
rho = 1

all_df["URel"] = UInf - all_df["UTurb"]
all_df.rename(columns={"Power": "LES_Power", "UDisk": "LES_UDisk"}, inplace=True)
all_df["LES_CP"] =  all_df["LES_Power"] / (0.5 * rho * math.pi * (D/2)**2 * (UInf)**3)

# %%
stationary_ct_lookup = metadata_df["CT_prime_mean"][0]
ct_ts = ctp_timeseries_df[["Time", "CT_prime"]].copy()
ct_ts

# %%
all_df = all_df.sort_values("Time").copy()
all_df = pd.merge_asof(
    all_df,
    ct_ts,
    on="Time",
    direction="nearest",   # or "backward" if you prefer
    # tolerance=0.05        # optional, if you want max allowed time mismatch
)

# %%
all_df.rename(columns={"CT_prime": "CTprime_ts"}, inplace=True)
all_df["CTprime_stationary"] = stationary_ct_lookup
all_df["CT_prime"] = np.where(
    all_df["use_cT_timeseries"].astype(bool),
    all_df["CTprime_ts"],
    all_df["CTprime_stationary"]
)

all_df["LES_CT"] =  all_df["CT_prime"] * all_df["LES_UDisk"]**2 / (UInf)**2

# %%
for i in range(2, 4):
    openfast_data = all_df[all_df["id"] == i].copy()
    print(np.unique(openfast_data["use_cT_timeseries"]))

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()  # second y-axis

    # Left axis
    sns.lineplot(data=openfast_data, x="Time", y="UTurb", ax=ax1, color="C0", lw=2)

    # Right axis
    sns.lineplot(data=openfast_data, x="Time", y="CT_prime", ax=ax2, color="C1", lw=2, ls="--")

    ax1.set_xlabel("Time")
    ax1.set_xlim(75, 85)
    ax1.set_ylabel("UTurb", color="C0")
    ax2.set_ylabel("CT_prime", color="C1")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2.tick_params(axis="y", labelcolor="C1")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    plt.show()

# %%
all_df.keys()

# %%
plot_df = all_df[all_df["id"] > 1].copy()

x0, x1 = 95, 100  # visible range
mean_df = plot_df[(plot_df["Time"] >= x0) & (plot_df["Time"] <= x1)]  # mean over shown window

# Keep color mapping consistent between line + mean line
hue_order = sorted(mean_df["use_motion_timeseries"].dropna().unique())  # [False, True]
palette = dict(zip(hue_order, sns.color_palette("deep", len(hue_order))))

ax = sns.lineplot(
    data=plot_df,
    x="Time",
    y="LES_UDisk",
    hue="use_motion_timeseries",
    hue_order=hue_order,
    palette=palette,
    lw=2,
    errorbar=None
)
ax.set_xlim(x0, x1)

# Horizontal mean line per hue group
means = mean_df.groupby("use_motion_timeseries")["LES_UDisk"].mean()
for grp, m in means.items():
    ax.hlines(m, x0, x1, colors=palette[grp], linestyles="--", linewidth=2)

# %%
model = UnifiedMomentum()

# choose precision based on your data (e.g., 6–10)
all_df["CT_key"] = all_df["CT_prime"].round(8)
all_df["CT_key"].dropna().unique()

# %%
model = UnifiedMomentum()

# choose precision based on your data (e.g., 6–10)
all_df["CT_key"] = all_df["CT_prime"].round(8)

# 1) Solve UMM once per unique CT'
umm_cache = {}
for ctk in all_df["CT_key"].dropna().unique():
    umm_cache[ctk] = model(ctk, yaw=0.0, tilt=0.0)

# %%
# 2) Map cached solution objects back to each row
all_df["umm_sol"] = all_df["CT_key"].map(umm_cache)

# 3) Compute outputs row-wise using cached solution
#    (list comprehensions are usually faster/cleaner than .apply for this)
UMM_UDisk = []
UMM_CT = []
UMM_CP = []

for row in all_df.itertuples(index=False):
    sol = row.umm_sol
    urel = row.URel

    # handle missing CT' safely
    if sol is None or (isinstance(urel, float) and np.isnan(urel)):
        UMM_UDisk.append(np.nan)
        UMM_CT.append(np.nan)
        UMM_CP.append(np.nan)
        continue

    ud = (1 - sol.an) * urel
    UMM_UDisk.append(ud)
    UMM_CT.append(sol.Ct * urel**2)
    UMM_CP.append(sol.Cp * urel**3)

all_df["UMM_UDisk"] = UMM_UDisk
all_df["UMM_CT"] = UMM_CT
all_df["UMM_CP"] = UMM_CP
all_df["UMM_Power"] = all_df["UMM_CP"] * (0.5 * rho * math.pi * (D/2)**2 * UInf**3)

# %%
all_df["Diff_UDisk"] = all_df["LES_UDisk"] - all_df["UMM_UDisk"]
all_df["Diff_CT"] = all_df["LES_CT"] - all_df["UMM_CT"]
all_df["Diff_CP"] = all_df["LES_CP"] - all_df["UMM_CP"]
all_df["Diff_Power"] = all_df["LES_Power"] - all_df["UMM_Power"]

# %%
all_df.keys()

# %%
sns.relplot(data = all_df[all_df["Time"] > 90], kind="line", x = "Time", y = "LES_CP", hue = "use_cT_timeseries", col = "use_motion_timeseries")

# %%
sns.relplot(data = all_df[all_df["Time"] > 95], kind="line", x = "Time", y = "UMM_CP", hue = "use_motion_timeseries", col = "use_cT_timeseries")

# %%
sns.relplot(data = all_df[all_df["Time"] > 75], kind="line", x = "Time", y = "UTurb", col = "use_cT_timeseries", row = "use_motion_timeseries")

# %%
# columns to average
diff_cols = [c for c in all_df.columns if c.startswith("Diff_")]

# one row per id
time_avg = (
    all_df
    .groupby("id", as_index=False)[diff_cols]
    .mean()
)

print(time_avg.head())

# %%
all_df["Phase"]

# %%
all_df.keys()

# %%
ax = sns.lineplot(data=all_df[all_df["id"] > 1], x="Time", y="Diff_CT", hue = "use_motion_timeseries", lw=2)
ax.set_xlim(95, 100)

# %%
