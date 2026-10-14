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
def num(x):
    return pd.to_numeric(x, errors="coerce")  # bad parses -> NaN


# %%
data_path = Path(au.DATA_PATH)

# --- F_0032 ---
sim_high_f_folder = os.path.join(au.DATA_PATH, "F_0032_SU_Files")
rows_32, fields_32 = mplts.get_sim_varied_params(sim_high_f_folder)
ids_32, dt_32, surge_freq_32, tstop_32, n_hrs_32 = zip(*rows_32)


# --- F_0029 ---
sim_reg_f_folder = os.path.join(au.DATA_PATH, "F_0029_SU_Files")
rows_29, fields_29 = mplts.get_sim_varied_params(sim_reg_f_folder)
(ids_29, cT_29, surge_amp_29, surge_freq_29, nx_29, ny_29, nz_29,
    Lx_29, Ly_29, Lz_29, dt_29, tstop_29, n_hrs_29, filterWidth_29, useCorrection_29
) = zip(*rows_29)

# %%
df32 = pd.DataFrame({
    "id": ids_32,
    "surge_freq": num(surge_freq_32),
    "surge_amplitude": 0.3,
    "cT": 2.0, 
    "nx": num(nx_29[0]), "ny": num(ny_29[0]), "nz": num(nz_29[0]),
    "Lx": num(Lx_29[0]), "Ly": num(Ly_29[0]), "Lz": num(Lz_29[0]),
    "dt": num(dt_32),
    "filterWidth": num(filterWidth_29[0]),
    # leave non-numeric as-is:
    "useCorrection": useCorrection_29[0],
})

df29 = pd.DataFrame({
    "id": ids_29,
    "cT": num(cT_29),
    "surge_amplitude": num(surge_amp_29),
    "surge_freq": num(surge_freq_29),
    "nx": num(nx_29), "ny": num(ny_29), "nz": num(nz_29),
    "Lx": num(Lx_29), "Ly": num(Ly_29), "Lz": num(Lz_29),
    "dt": num(dt_29),
    "filterWidth": num(filterWidth_29),
    # leave non-numeric as-is:
    "useCorrection": useCorrection_29,
})
# keep dt = 0.005 only
df32 = df32[df32["dt"] != 0.005]
df29 = df29[df29["dt"] == 0.005]

# keep only Av = 0.3
df29 = df29[np.isclose(df29["surge_amplitude"], 0.3)]

# keep only CT = 2
df29 = df29[np.isclose(df29["cT"], 2.0)]

# %%
df29

# %%
df32


# %%
def load_runs_to_timeseries(df_runs, sim_folder, spinup_time):
    dfs = []
    for r in df_runs.itertuples(index=False):
        try:
            id_str = r.id
            run_folder = os.path.join(sim_folder, f"Sim_{id_str}")

            data = au.load_simulation_timeseries(
                run_folder,
                id_str,
                r.surge_freq,
                spinup_time=spinup_time
            )

            meta = {
                "id": int(r.id),
                "CT_prime": float(r.cT),  # or r.CT_prime if that is your column name
                "Surge_Amplitude": float(r.surge_amplitude),
                "Pitch_Amplitude": 0.0,
                "Frequency": float(r.surge_freq),
                "nx": float(r.nx),
                "ny": float(r.ny),
                "nz": float(r.nz),
                "Lx": float(r.Lx),
                "Ly": float(r.Ly),
                "Lz": float(r.Lz),
                "dt": float(r.dt),
                "filterWidth": float(r.filterWidth),
                "useCorrection": r.useCorrection,
            }

            df_temp = au.build_sim_dataframe(data, meta)
            dfs.append(df_temp)
        except Exception as e:
            print(f"Error occurred in {id_str}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# %%
# make column names consistent first (important)
# df29 and df32 should both have: id, cT, surge_amplitude, surge_freq, nx,ny,nz,Lx,Ly,Lz,dt,filterWidth,useCorrection

ts29 = load_runs_to_timeseries(df29, sim_reg_f_folder, 75)
ts32 = load_runs_to_timeseries(df32, sim_high_f_folder, 75)

final_surge_df = pd.concat([ts29, ts32], ignore_index=True, sort=False)

# %%
final_surge_df

# %%
final_surge_df = final_surge_df.rename(columns={"UDisk": "LES_UDisk", "Power": "LES_Power", "DeltaX": "XTurb"})
final_surge_df.keys()

# %%
UInf = 1.0
D = 1.0
rho = 1.0
A = math.pi * (D/2)**2
Pnorm = 0.5 * rho * A * UInf**3

final_surge_df["URel"] = UInf - final_surge_df["UTurb"]
omega = 2 * np.pi * final_surge_df["Frequency"]

final_surge_df["LES_CP"] = final_surge_df["LES_Power"] / Pnorm

model = UnifiedMomentum()
CT_prime = 2.00
umm_sol = model(CT_prime, yaw=0.0, tilt=0.0)

def get_umm_cp(row):
    return umm_sol.Cp * row.URel**3

final_surge_df["UMM_CP"] = final_surge_df.apply(get_umm_cp, axis=1)
final_surge_df["UMM_Power"] = final_surge_df["UMM_CP"] * Pnorm
final_surge_df["CP_diff"] = final_surge_df["LES_CP"] - final_surge_df["UMM_CP"]

# %%
final_surge_df.keys()

# %%
run_cols = ["id", "CT_prime", "Surge_Amplitude", "Frequency", "dt"]

time_avg_df = (
    final_surge_df
    .groupby(run_cols, as_index=False)
    .agg(
        LES_CP_time_mean=("LES_CP", "mean"),
        UMM_CP_time_mean=("UMM_CP", "mean"),
        CP_diff_time_mean=("CP_diff", "mean"),
    )
)

# %%
time_avg_df


# %%
def phase_average_cp(df, nperiods=8, n_bins=100):
    df = df.copy()

    cols_to_avg = ["LES_CP", "UMM_CP", "CP_diff"]
    phase_grid = np.round(np.linspace(0, 1, n_bins, endpoint=False), 4)
    period = df["Period"]

    # select middle periods (skip first transient)
    unique_periods = np.unique(period)
    if len(unique_periods) < nperiods + 2:
        # fallback: use all except first if not enough
        valid_periods = unique_periods[1:] if len(unique_periods) > 1 else unique_periods
    else:
        valid_periods = unique_periods[1:(nperiods+1)]

    df = df[df["Period"].isin(valid_periods)].copy()

    # bin phase
    dphi = 1.0 / n_bins
    p = np.mod(df["Phase"].to_numpy(), 1.0)
    phase_bin = np.round(np.round(p / dphi) * dphi, 4)
    phase_bin = np.mod(phase_bin, 1.0)
    df["Phase_bin"] = phase_bin

    # grouped stats
    g = df.groupby("Phase_bin", observed=True)[cols_to_avg].agg(["mean", "std"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    g = g.reset_index().rename(columns={"Phase_bin": "Phase"})

    # enforce full grid
    g = g.set_index("Phase").reindex(phase_grid).reset_index()

    # metadata
    meta_cols = [
        "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
        "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
    ]
    for c in meta_cols:
        g[c] = df[c].iloc[0]

    return g


# %%
phase_avg_list = []
for _, sub in final_surge_df.groupby(run_cols):
    phase_avg_list.append(phase_average_cp(sub, nperiods=8, n_bins=100))

phase_avg_df = pd.concat(phase_avg_list, ignore_index=True)

# %%
phase_avg_df

# %%
extrema_df = (
    phase_avg_df
    .groupby(run_cols, as_index=False)
    .agg(
        LES_CP_phase_max=("LES_CP_mean", "max"),
        UMM_CP_phase_max=("UMM_CP_mean", "max"),
        LES_CP_phase_min=("LES_CP_mean", "min"),
        UMM_CP_phase_min=("UMM_CP_mean", "min"),
    )
)

extrema_df["max_diff_extrema"] = extrema_df["LES_CP_phase_max"] - extrema_df["UMM_CP_phase_max"]
extrema_df["min_diff_extrema"] = extrema_df["LES_CP_phase_min"] - extrema_df["UMM_CP_phase_min"]
extrema_df["amp_diff_extrema"] = (
    (extrema_df["LES_CP_phase_max"] - extrema_df["LES_CP_phase_min"])
    - (extrema_df["UMM_CP_phase_max"] - extrema_df["UMM_CP_phase_min"])
)

# %%
summary_df = time_avg_df.merge(extrema_df, on=run_cols, how="inner")

# %%
summary_df

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plot_long = summary_df.melt(
    id_vars="Frequency",
    value_vars=["CP_diff_time_mean", "max_diff_extrema"],
    var_name="Metric",
    value_name="Value"
)

plot_long["Metric"] = plot_long["Metric"].replace({
    "CP_diff_time_mean": r"$\overline{\Delta C_P}$",
    "max_diff_extrema": r"$\Delta\max(C_P)$",
})

sns.set_theme(style="whitegrid", context="talk")

fig, ax = plt.subplots(figsize=(9, 6), dpi = 300)

sns.lineplot(
    data=plot_long,
    x="Frequency",
    y="Value",
    hue="Metric",
    style="Metric",
    dashes=False,
    linewidth=3.2,
    ax=ax,
    legend=True
)

sns.scatterplot(
    data=plot_long,
    x="Frequency",
    y="Value",
    hue="Metric",
    # style="Metric",
    s=220,
    edgecolor="black",
    linewidth=0.8,
    ax=ax,
    legend=False
)

ax.axhline(0, color="black", lw=1.4, alpha=0.7)
ax.set_xlabel(r"$f_r$", fontsize=22, fontweight="normal")
ax.set_ylabel(r"$\Delta C_P$", fontsize=22, fontweight="normal")

# --- FORCE tick label sizes ---
ax.tick_params(axis="both", which="major", labelsize=18)
# ax.tick_params(axis="both", which="minor", labelsize=16)

# --- FORCE legend font sizes ---
leg = ax.legend(title="", loc="best", frameon=True)
for txt in leg.get_texts():
    txt.set_fontsize(18)
if leg.get_title() is not None:
    leg.get_title().set_fontsize(18)

plt.tight_layout()
plt.show()

# %%
