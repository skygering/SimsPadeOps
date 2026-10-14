# ---
# jupyter:
#   jupytext:
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

# --- F_0029 ---
sims_29_folder = os.path.join(au.DATA_PATH, "F_0029_SU_Files")
rows_29, fields_29 = mplts.get_sim_varied_params(sims_29_folder)
(ids_29, cT_29, surge_amp_29, surge_freq_29, nx_29, ny_29, nz_29,
    Lx_29, Ly_29, Lz_29, dt_29, tstop_29, n_hrs_29, filterWidth_29, useCorrection_29
) = zip(*rows_29)

# --- F_0032 ---
sims_32_folder = os.path.join(au.DATA_PATH, "F_0032_SU_Files")
rows_32, fields_32 = mplts.get_sim_varied_params(sims_32_folder)
ids_32, dt_32, surge_freq_32, tstop_32, n_hrs_32 = zip(*rows_32)

# --- F_0036 ---
sims_36_folder = os.path.join(au.DATA_PATH, "F_0036_SU_Files")
rows_36, fields_36 = mplts.get_sim_varied_params(sims_36_folder)
ids_36, cT_36, surge_amplitude_36, surge_freq_36, nx_36, ny_36, nz_36, Lx_36, Ly_36, Lz_36, dt_36, tstop_36, n_hrs_36, filterWidth_36, useCorrection_36 = zip(*rows_36)

# --- F_0039 ---
sims_39_folder = os.path.join(au.DATA_PATH, "F_0039_SU_Files")
rows_39, fields_39 = mplts.get_sim_varied_params(sims_39_folder)
ids_39, cT_39, surge_amplitude_39, surge_freq_39, nx_39, ny_39, nz_39, Lx_39, Ly_39, Lz_39, dt_39, tstop_39, n_hrs_39, filterWidth_39, useCorrection_39 = zip(*rows_39)

# --- F_0040 ---
sims_40_folder = os.path.join(au.DATA_PATH, "F_0040_SU_Files")
rows_40, fields_40 = mplts.get_sim_varied_params(sims_40_folder)
ids_40, cT_40, surge_amplitude_40, surge_freq_40, nx_40, ny_40, nz_40, Lx_40, Ly_40, Lz_40, dt_40, tstop_40, n_hrs_40, filterWidth_40, useCorrection_40 = zip(*rows_40)


# %%
def num(x):
    return pd.to_numeric(x, errors="coerce")  # bad parses -> NaN


# %%
df29 = pd.DataFrame({
    "id": ids_29,
    "cT": num(cT_29),
    "surge_amplitude": num(surge_amp_29),
    "surge_freq": num(surge_freq_29),
    "nx": num(nx_29), "ny": num(ny_29), "nz": num(nz_29),
    "Lx": num(Lx_29), "Ly": num(Ly_29), "Lz": num(Lz_29),
    "dt": num(dt_29),
    "filterWidth": num(filterWidth_29),
    "useCorrection": useCorrection_29,
})

df32 = pd.DataFrame({
    "id": ids_32,
    "surge_freq": num(surge_freq_32),
    "surge_amplitude": 0.3,
    "cT": 2.0, 
    "nx": num(nx_29[0]), "ny": num(ny_29[0]), "nz": num(nz_29[0]),
    "Lx": num(Lx_29[0]), "Ly": num(Ly_29[0]), "Lz": num(Lz_29[0]),
    "dt": num(dt_32),
    "filterWidth": num(filterWidth_29[0]),
    "useCorrection": useCorrection_29[0],
})

df36 = pd.DataFrame({
    "id": ids_36,
    "surge_freq": num(surge_freq_36),
    "surge_amplitude": num(surge_amplitude_36),
    "cT": num(cT_36), 
    "nx": num(nx_36[0]), "ny": num(ny_36[0]), "nz": num(nz_36[0]),
    "Lx": num(Lx_36[0]), "Ly": num(Ly_36[0]), "Lz": num(Lz_36[0]),
    "dt": num(dt_36),
    "filterWidth": num(filterWidth_36[0]),
    "useCorrection": useCorrection_36[0],
})

# df39 = pd.DataFrame({
#     "id": ids_39,
#     "surge_freq": num(surge_freq_39),
#     "surge_amplitude": num(surge_amplitude_39),
#     "cT": num(cT_39), 
#     "nx": num(nx_39[0]), "ny": num(ny_39[0]), "nz": num(nz_39[0]),
#     "Lx": num(Lx_39[0]), "Ly": num(Ly_39[0]), "Lz": num(Lz_39[0]),
#     "dt": num(dt_39),
#     "filterWidth": num(filterWidth_39[0]),
#     "useCorrection": useCorrection_39[0],
# })

# df40 = pd.DataFrame({
#     "id": ids_40,
#     "surge_freq": num(surge_freq_40),
#     "surge_amplitude": num(surge_amplitude_40),
#     "cT": num(cT_40), 
#     "nx": num(nx_40[0]), "ny": num(ny_40[0]), "nz": num(nz_40[0]),
#     "Lx": num(Lx_40[0]), "Ly": num(Ly_40[0]), "Lz": num(Lz_40[0]),
#     "dt": num(dt_40),
#     "filterWidth": num(filterWidth_40[0]),
#     "useCorrection": useCorrection_40[0],
# })


# keep dt ! 0.005 only
df32 = df32[df32["dt"] != 0.005]
# keep dt = 0.005 only
df29 = df29[df29["dt"] == 0.005]
# remove data that is numerically wrong
# Remove known bad df36 cases
bad_cases = [
    (0.2, 0.75, 1.0),
    (0.3, 0.25, 1.0),
    (0.4, 0.25, 1.0),
    (0.4, 1.50, 1.0),
]

mask_bad = pd.Series(False, index=df36.index)
for A, f, ct in bad_cases:
    mask_bad |= (
        np.isclose(df36["surge_amplitude"], A) &
        np.isclose(df36["surge_freq"], f) &
        np.isclose(df36["cT"], ct)
    )

df36 = df36.loc[~mask_bad].copy()
print(f"Removed {mask_bad.sum()} rows from df36")


# %%
def load_runs_to_timeseries(df_runs, sim_folder, spinup_time, nbins = 100):
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
            amp, freq = float(r.surge_amplitude), float(r.surge_freq)
            meta = {
                "id": int(r.id),
                "CT_prime": float(r.cT),  # or r.CT_prime if that is your column name
                "Surge_Amplitude": amp,
                "Pitch_Amplitude": 0.0,
                "Frequency": freq,
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
print("Loading 29")
ts29 = load_runs_to_timeseries(df29, sims_29_folder, 75)
print("Loading 32")
ts32 = load_runs_to_timeseries(df32, sims_32_folder, 75)
print("Loading 36")
ts36 = load_runs_to_timeseries(df36, sims_36_folder, 0)
# print("Loading 39")
# ts39 = load_runs_to_timeseries(df39, sims_39_folder, 0)
# print("Loading 40")
# ts40 = load_runs_to_timeseries(df40, sims_40_folder, 75)

# %%
print("Combining")
final_surge_df = pd.concat([ts29, ts32, ts36], ignore_index=True, sort=False)

# %%
final_surge_df.keys()
final_surge_df = final_surge_df[final_surge_df["Surge_Amplitude"] < 0.5]
final_surge_df = final_surge_df[final_surge_df["Surge_Amplitude"] != 0.05]
final_surge_df = final_surge_df[final_surge_df["Frequency"] <= 2.0]

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

final_surge_df["LES_CT"] =  final_surge_df["CT_prime"] * final_surge_df["LES_UDisk"]**2 / (UInf)**2
final_surge_df["LES_CP"] = final_surge_df["LES_Power"] / Pnorm


# %%
model = UnifiedMomentum()
CT_primes = [1.00, 1.33, 2.00, 2.33]
umm_sols = {round(k,2): model(k, yaw=0.0, tilt=0.0) for k in CT_primes}

def get_umm_sol(row):
    return umm_sols[row.CT_prime]
def get_umm_ct(row):
    sol = get_umm_sol(row)
    return sol.Ct * row.URel**2
def get_umm_cp(row):
    sol = get_umm_sol(row)
    return sol.Cp * row.URel**3

final_surge_df["UMM_CT"] = final_surge_df.apply(get_umm_ct, axis=1)
final_surge_df["UMM_CP"] = final_surge_df.apply(get_umm_cp, axis=1)
final_surge_df["CT_diff"] = final_surge_df["LES_CT"] - final_surge_df["UMM_CT"]
final_surge_df["CP_diff"] = final_surge_df["LES_CP"] - final_surge_df["UMM_CP"]

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# pick one CT_prime value (replace with what you want)
ct_val = 1 # example

dfp = final_surge_df.copy()
dfp = dfp[dfp['CT_prime'] == ct_val].copy()
# dfp = dfp[dfp['Surge_Amplitude'] >= 0.1]
# dfp = dfp[dfp['Frequency'] == 1.5]
dfp = dfp[dfp['Time'] > 75]
dfp = dfp[dfp['Time'] < 85]

# If float matching is tricky, use this instead:
# dfp = dfp[np.isclose(dfp['CT_prime'], ct_val, atol=1e-6)].copy()

if dfp.empty:
    print(f"No rows found for CT_prime = {ct_val}")
else:
    # dfp['time'] = (dfp['Period']) + dfp['Phase']

    # Sort + optional averaging over duplicate (amp,freq,time)
    dfp = dfp.sort_values(['Surge_Amplitude', 'Frequency', 'Period', 'Phase'])
    plot_df = (
        dfp.groupby(['Surge_Amplitude', 'Frequency', 'Time', "Phase"], as_index=False)
           [['LES_CP', 'UMM_CP', 'CP_diff']]
           .mean()
    )

    # Long form for 3 side-by-side panels
    long_df = plot_df.melt(
        id_vars=['Surge_Amplitude', 'Frequency', 'Time', "Phase"],
        value_vars=['LES_CP', 'UMM_CP', 'CP_diff'],
        var_name='metric',
        value_name='value'
    )

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.relplot(
        data=long_df,
        x='Phase',
        y='value',
        hue='Frequency',   # color
        kind='scatter',
        col='metric',
        row = "Surge_Amplitude",
        col_order=['LES_CP', 'UMM_CP'],
        facet_kws={'sharey': False},
        aspect=2,
        palette = "tab10"
    )

    g.set_axis_labels("Time (period + phase)", "Value")
    g.fig.suptitle(f"CT_prime = {ct_val}", y=1.03)
    plt.show()

# %%
# identify stationary rows
stationary_mask = (final_surge_df["Frequency"] == 0) & (final_surge_df["Surge_Amplitude"] == 0)

# compute reference values (one per CT')
ref_df = (
    final_surge_df[stationary_mask]
    .groupby("CT_prime")[[
        "LES_CP", "LES_CT", "UMM_CP", "UMM_CT",
    ]]
    .mean()
    .rename(columns=lambda c: f"{c}_ref")
    .reset_index()
)

final_surge_df = final_surge_df.merge(ref_df, on="CT_prime", how="left", suffixes=("", "_ref"))

# %%
for col in ["LES_CP", "LES_CT", "UMM_CP", "UMM_CT"]:
    final_surge_df[col+"_norm"] = final_surge_df[col] / final_surge_df[f"{col}_ref"]

final_surge_df["CT_diff_norm"] = final_surge_df["LES_CT_norm"] - final_surge_df["UMM_CT_norm"]
final_surge_df["CP_diff_norm"] = final_surge_df["LES_CP_norm"] - final_surge_df["UMM_CP_norm"]


# %%
# def phase_average_periods_interp(df, nperiods=8, n_bins=100):
#     import numpy as np
#     import pandas as pd

#     cols_interp = [
#         "LES_CP", "LES_CT", "UMM_CP", "UMM_CT", "CT_diff", "CP_diff",
#         "LES_CP_norm", "LES_CT_norm", "UMM_CP_norm", "UMM_CT_norm",
#         "CT_diff_norm", "CP_diff_norm"
#     ]
#     phase_grid = np.linspace(0, 1, n_bins, endpoint=False)

#     # copy + sort
#     work = df.sort_values("Time").copy()

#     # stationary case
#     surge_amp = work["Surge_Amplitude"].iloc[0]
#     pitch_amp = work["Pitch_Amplitude"].iloc[0]

#     if surge_amp == 0 and pitch_amp == 0:
#         out = pd.DataFrame({
#             "Phase": np.tile(phase_grid, nperiods),
#             "Period": np.repeat(np.arange(nperiods), n_bins),
#         })
#         for col in cols_interp:
#             out[col] = work[col].mean()

#         meta_cols = [
#             "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
#             "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
#         ]
#         for c in meta_cols:
#             out[c] = work[c].iloc[0]
#         return out

#     # If Period missing, compute from phase wrap
#     if "Period" not in work.columns:
#         ph_tmp = np.mod(work["Phase"].values, 1.0)
#         per = np.zeros(len(ph_tmp), dtype=int)
#         per[1:] = np.cumsum(np.diff(ph_tmp) < 0)
#         work["Period"] = per

#     # use ALL period values, not iloc[0]
#     unique_periods = np.sort(work["Period"].dropna().unique())

#     if len(unique_periods) < nperiods + 2:
#         raise ValueError(f"Need at least {nperiods + 2} detected periods; got {len(unique_periods)}")

#     valid_periods = unique_periods[1:(nperiods + 1)]  # periods 1..8 if nperiods=8
#     work = work[work["Period"].isin(valid_periods)].copy()

#     # interpolate each selected period
#     interp_periods = []
#     for p, g in work.groupby("Period", sort=True):
#         g = g.sort_values("Phase").copy()
#         ph = np.mod(g["Phase"].values, 1.0)

#         # ensure monotonic for np.interp
#         order = np.argsort(ph)
#         ph = ph[order]

#         out = {"Phase": phase_grid, "Period": np.full(n_bins, int(p), dtype=int)}

#         # periodic extension
#         ph_ext = np.concatenate([ph - 1.0, ph, ph + 1.0])

#         for col in cols_interp:
#             vals = g[col].values[order]
#             vals_ext = np.concatenate([vals, vals, vals])
#             out[col] = np.interp(phase_grid, ph_ext, vals_ext)

#         interp_periods.append(pd.DataFrame(out))

#     final_df = pd.concat(interp_periods, ignore_index=True)

#     meta_cols = [
#         "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
#         "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
#     ]
#     for c in meta_cols:
#         final_df[c] = work[c].iloc[0]

#     return final_df

# %%
def phase_average_periods_binned(df, nperiods=8, n_bins=100):
    df = df.copy()

    cols_to_avg = [
        "LES_CP", "LES_CT", "UMM_CP", "UMM_CT", "CT_diff", "CP_diff",
        "LES_CP_norm", "LES_CT_norm", "UMM_CP_norm", "UMM_CT_norm",
        "CT_diff_norm", "CP_diff_norm"
    ]

    # phase grid
    phase_grid = np.round(np.linspace(0, 1, n_bins, endpoint=False), 4)
    dphi = 1.0 / n_bins

    # stationary case
    surge_amp = df["Surge_Amplitude"].iloc[0]
    pitch_amp = df["Pitch_Amplitude"].iloc[0]
    if surge_amp == 0 and pitch_amp == 0:
        out = pd.DataFrame({
            "Phase": np.tile(phase_grid, nperiods),
            "Period": np.repeat(np.arange(1, nperiods + 1), n_bins),
        })
        for c in cols_to_avg:
            out[c] = df[c].mean()

        meta_cols = [
            "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
            "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
        ]
        for c in meta_cols:
            out[c] = df[c].iloc[0]
        return out

    # If Period missing, infer by phase wrap
    if "Period" not in df.columns:
        ph_tmp = np.mod(df["Phase"].values, 1.0)
        per = np.zeros(len(ph_tmp), dtype=int)
        per[1:] = np.cumsum(np.diff(ph_tmp) < 0) + 1
        df["Period"] = per

    # choose periods: skip first transient period
    unique_periods = np.sort(df["Period"].dropna().unique())
    if len(unique_periods) < nperiods + 1:
        # fallback: use as many as possible after first
        valid_periods = unique_periods[1:]
    else:
        valid_periods = unique_periods[1:(nperiods + 1)]

    df = df[df["Period"].isin(valid_periods)].copy()

    # phase binning
    p = np.mod(df["Phase"].to_numpy(), 1.0)
    phase_bin = np.round(np.round(p / dphi) * dphi, 4)
    phase_bin = np.mod(phase_bin, 1.0)
    df["Phase_bin"] = phase_bin

    # group by period + phase_bin
    g = (
        df.groupby(["Period", "Phase_bin"], observed=True)[cols_to_avg]
          .mean()
          .reset_index()
          .rename(columns={"Phase_bin": "Phase"})
    )

    # enforce full phase grid for every valid period
    full_index = pd.MultiIndex.from_product(
        [np.sort(valid_periods), phase_grid],
        names=["Period", "Phase"]
    )
    g = (
        g.set_index(["Period", "Phase"])
         .reindex(full_index)
         .reset_index()
    )

    # metadata
    meta_cols = [
        "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
        "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
    ]
    for c in meta_cols:
        g[c] = df[c].iloc[0]

    return g


# %%
def apply_phase_binned_all_runs(df_all, nperiods=8, n_bins=100):
    out = []
    for run_id, g in df_all.groupby("id", sort=True):
        g = g.sort_values("Time").copy()   # important for correct period detection
        try:
            g_interp = phase_average_periods_binned(g, nperiods=nperiods, n_bins=n_bins)
            out.append(g_interp)
        except Exception as e:
            print(f"[skip] id={run_id}: {e}")

    if not out:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


# %%
df_interp_all = apply_phase_binned_all_runs(final_surge_df, nperiods=8, n_bins=100)

# %%
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # pick one CT_prime value (replace with what you want)
# ct_val = 1 # example

# dfp = df_interp_all.copy()
# dfp = dfp[dfp['CT_prime'] == ct_val].copy()
# # dfp = dfp[dfp['Surge_Amplitude'] >= 0.1]
# dfp = dfp[dfp['Frequency'] == 0.25]

# # If float matching is tricky, use this instead:
# # dfp = dfp[np.isclose(dfp['CT_prime'], ct_val, atol=1e-6)].copy()

# if dfp.empty:
#     print(f"No rows found for CT_prime = {ct_val}")
# else:
#     # Build continuous time = (Period-1) + Phase_norm
#     phase_max = dfp['Phase'].max()
#     if phase_max > 1.5:   # likely degrees
#         dfp['Phase_norm'] = dfp['Phase'] / 360.0
#     else:
#         dfp['Phase_norm'] = dfp['Phase']

#     if dfp['Period'].min() == 1:
#         dfp['time'] = (dfp['Period'] - 1) + dfp['Phase_norm']
#     else:
#         dfp['time'] = dfp['Period'] + dfp['Phase_norm']

#     # Sort + optional averaging over duplicate (amp,freq,time)
#     dfp = dfp.sort_values(['Surge_Amplitude', 'Frequency', 'Period', 'Phase_norm'])
#     plot_df = (
#         dfp.groupby(['Surge_Amplitude', 'Frequency', 'time'], as_index=False)
#            [['LES_CP', 'UMM_CP', 'CP_diff']]
#            .mean()
#     )

#     # Long form for 3 side-by-side panels
#     long_df = plot_df.melt(
#         id_vars=['Surge_Amplitude', 'Frequency', 'time'],
#         value_vars=['LES_CP', 'UMM_CP', 'CP_diff'],
#         var_name='metric',
#         value_name='value'
#     )

#     sns.set_theme(style="whitegrid", context="talk")
#     g = sns.relplot(
#         data=long_df,
#         x='time',
#         y='value',
#         hue='Surge_Amplitude',   # color
#         style='Frequency',       # style
#         kind='line',
#         col='metric',
#         col_order=['LES_CP', 'UMM_CP'],
#         col_wrap = 1,
#         facet_kws={'sharey': False},
#         aspect=2  
#     )

#     g.set_axis_labels("Time (period + phase)", "Value")
#     g.fig.suptitle(f"CT_prime = {ct_val}", y=1.03)
#     plt.show()

# %%
case_cols = ['Surge_Amplitude', 'Frequency', 'CT_prime']

# -----------------------------
# 1) Mean of existing diff cols
# -----------------------------
diff_cols = [c for c in df_interp_all.columns if 'diff' in c.lower()]

mean_diff = (
    df_interp_all.groupby(case_cols, as_index=False)[diff_cols]
      .mean()
)

# %%
les_mean_ref = (
    df_interp_all.groupby(case_cols, as_index=False)[['LES_CP','LES_CT','LES_CP_norm','LES_CT_norm']]
      .mean()
      .rename(columns={
          'LES_CP': 'LES_CP_mean_ref',
          'LES_CT': 'LES_CT_mean_ref',
          'LES_CP_norm': 'LES_CP_norm_mean_ref',
          'LES_CT_norm': 'LES_CT_norm_mean_ref',
      })
)

# %%
# --------------------------------------------
# 2) Period-averaged difference in PERIOD MAX
# --------------------------------------------
pairs = [
    ('LES_CP', 'UMM_CP', 'CP'),
    ('LES_CT', 'UMM_CT', 'CT'),
    ('LES_CP_norm', 'UMM_CP_norm', 'CP_norm'),
    ('LES_CT_norm', 'UMM_CT_norm', 'CT_norm'),
]

# Max per period for each signal
period_max = (
    df_interp_all.groupby(case_cols + ['Period'], as_index=False)
      [[c for p in pairs for c in p[:2]]]
      .max()
)

# Difference in maximum for each period, then average across periods
for les_col, umm_col, tag in pairs:
    period_max[f'diff_max_{tag}'] =  period_max[les_col] - period_max[umm_col]

diff_max_cols = [f'diff_max_{tag}' for _, _, tag in pairs]

period_avg_diff_max = (
    period_max.groupby(case_cols, as_index=False)[diff_max_cols]
              .mean()
)


# %%
les_max_ref = (
    period_max.groupby(case_cols, as_index=False)[['LES_CP','LES_CT','LES_CP_norm','LES_CT_norm']]
              .mean()
              .rename(columns={
                  'LES_CP': 'LES_CP_max_ref',
                  'LES_CT': 'LES_CT_max_ref',
                  'LES_CP_norm': 'LES_CP_norm_max_ref',
                  'LES_CT_norm': 'LES_CT_norm_max_ref',
              })
)


# %%
# Merge all
out = mean_diff.merge(period_avg_diff_max, on=case_cols, how='outer')
out = out.merge(les_mean_ref, on=case_cols, how='left')
out = out.merge(les_max_ref, on=case_cols, how='left')

eps = 1e-12

# Percent for mean-diff columns (if present)
if 'CP_diff' in out.columns:
    out['CP_diff_percent'] = 100 * out['CP_diff'] / (out['LES_CP_mean_ref'].abs() + eps)
if 'CT_diff' in out.columns:
    out['CT_diff_percent'] = 100 * out['CT_diff'] / (out['LES_CT_mean_ref'].abs() + eps)
if 'CP_diff_norm' in out.columns:
    out['CP_diff_norm_percent'] = 100 * out['CP_diff_norm'] / (out['LES_CP_norm_mean_ref'].abs() + eps)
if 'CT_diff_norm' in out.columns:
    out['CT_diff_norm_percent'] = 100 * out['CT_diff_norm'] / (out['LES_CT_norm_mean_ref'].abs() + eps)

# Percent for period-averaged diff-in-max
out['diff_max_CP_percent']      = 100 * out['diff_max_CP']      / (out['LES_CP_max_ref'].abs() + eps)
out['diff_max_CT_percent']      = 100 * out['diff_max_CT']      / (out['LES_CT_max_ref'].abs() + eps)
out['diff_max_CP_norm_percent'] = 100 * out['diff_max_CP_norm'] / (out['LES_CP_norm_max_ref'].abs() + eps)
out['diff_max_CT_norm_percent'] = 100 * out['diff_max_CT_norm'] / (out['LES_CT_norm_max_ref'].abs() + eps)

# --------------------------------------------
# 4) Final dataframe with requested structure
# --------------------------------------------
final_cols = [
    'Surge_Amplitude', 'Frequency', 'CT_prime',

    # mean differences
    'CP_diff', 'CT_diff', 'CP_diff_norm', 'CT_diff_norm',
    'CP_diff_percent', 'CT_diff_percent', 'CP_diff_norm_percent', 'CT_diff_norm_percent',

    # period-averaged diff in maximum
    'diff_max_CP', 'diff_max_CT', 'diff_max_CP_norm', 'diff_max_CT_norm',
    'diff_max_CP_percent', 'diff_max_CT_percent',
    'diff_max_CP_norm_percent', 'diff_max_CT_norm_percent',
]

final_cols = [c for c in final_cols if c in out.columns]
final_df = out[final_cols].sort_values(case_cols).reset_index(drop=True)

print(final_df.head())

# %%
import os
import seaborn as sns
import matplotlib.pyplot as plt

# DataFrame assumed to be your summary output (one row per case)
# e.g., final_df from previous step
plot_df = final_df.copy()
# plot_df = plot_df[plot_df["CT_prime"] == 2]

metrics = [
    'CP_diff', 'CT_diff', 'CP_diff_norm', 'CT_diff_norm',
    'CP_diff_percent', 'CT_diff_percent', 'CP_diff_norm_percent', 'CT_diff_norm_percent',

    # period-averaged diff in maximum
    'diff_max_CP', 'diff_max_CT', 'diff_max_CP_norm', 'diff_max_CT_norm',
    'diff_max_CP_percent', 'diff_max_CT_percent',
    'diff_max_CP_norm_percent', 'diff_max_CT_norm_percent',
]

# Optional: ensure numeric types
for c in ['CT_prime', 'Surge_Amplitude', 'Frequency'] + metrics:
    if c in plot_df.columns:
        plot_df[c] = pd.to_numeric(plot_df[c], errors='coerce')

# Optional: order CT_prime so facets are consistently sorted
ct_order = sorted(plot_df['CT_prime'].dropna().unique())

# Output folder
outdir = "relplots_4x4_by_metric"
os.makedirs(outdir, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

for ycol in metrics:
    if ycol not in plot_df.columns:
        print(f"Skipping missing column: {ycol}")
        continue

    g = sns.relplot(
        data=plot_df,
        x='Surge_Amplitude',
        y=ycol,
        hue='Frequency',
        col='CT_prime',
        kind='line',         # use 'scatter' if you prefer points only
        marker='o',
        dashes=False,
        facet_kws={'sharey': True, 'sharex': True},
        col_order=ct_order,
        height=3.0,
        aspect=1.0,
        palette='viridis',

    )

    g.fig.subplots_adjust(top=0.90)
    g.fig.suptitle(f"{ycol} vs Surge_Amplitude (hue=Frequency, facets=CT_prime)", fontsize=14)

    # axis labels
    g.set_axis_labels("Surge Amplitude", ycol)

    # Save figure
    fname = f"{ycol}_facet4x4.png"
    g.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(g.fig)

print(f"Saved plots to: {outdir}")

# %%
