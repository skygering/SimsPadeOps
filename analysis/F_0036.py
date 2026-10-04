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

# --- F_0036 ---
sim_low_ctp_folder = os.path.join(au.DATA_PATH, "F_0036_SU_Files")
rows_36, fields_36 = mplts.get_sim_varied_params(sim_low_ctp_folder)
ids_36 ,cT_36 ,surge_amplitude_36 ,surge_freq_36 ,nx_36 ,ny_36 ,nz_36 ,Lx_36 ,Ly_36 ,Lz_36 ,dt_36 ,tstop_36 ,n_hrs_36 ,filterWidth_36 ,useCorrection_36 = zip(*rows_36)


# %%
def num(x):
    return pd.to_numeric(x, errors="coerce")  # bad parses -> NaN


# %%
df36 = pd.DataFrame({
    "id": ids_36,
    "surge_freq": num(surge_freq_36),
    "surge_amplitude": 0.3,
    "cT": 2.0, 
    "nx": num(nx_36[0]), "ny": num(ny_36[0]), "nz": num(nz_36[0]),
    "Lx": num(Lx_36[0]), "Ly": num(Ly_36[0]), "Lz": num(Lz_36[0]),
    "dt": num(dt_36),
    "filterWidth": num(filterWidth_36[0]),
    # leave non-numeric as-is:
    "useCorrection": useCorrection_36[0],
})


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
ts36 = load_runs_to_timeseries(df36, sim_low_ctp_folder, 75)

# %%
ts36

# %%
r = df36.iloc[0]
id_str = r.id
run_folder = os.path.join(sim_low_ctp_folder, f"Sim_{id_str}")

sim = pio.BudgetIO(run_folder, padeops=True, runid=0, normalize_origin="turbine")

power = sim.read_turb_power("all", turb=1)
uvel = sim.read_turb_uvel("all", turb=1)

# %%
n_dumped = min(len(power), len(uvel))
power, uvel = power[:n_dumped], uvel[:n_dumped]

# %%
power, uvel

# %%
log_paths = au.get_all_log_paths(run_folder, id_str)

# %%
log_paths

# %%
log0 = au._parse_one_log(log_paths[0])["log"]
log1 = au._parse_one_log(log_paths[1])["log"]

# %%
log0["uturb"][19995:20005]

# %%
log1["TIDX"][:5]

# %%
log1["uturb"][:5]

# %%
log_paths = au.get_all_log_paths(run_folder, id_str)
if len(log_paths) == 1:
    log = au._parse_one_log(log_paths[0])["log"]
else:
    infos = [au._parse_one_log(p) for p in log_paths]
    log = au._concat_logs_prefer_last(infos)

# %%
log["Time"]

# %%
log["TIDX"]

# %%
time = np.asarray(log["Time"])[:n_dumped]
tidx = np.asarray(log["TIDX"])[:n_dumped]
