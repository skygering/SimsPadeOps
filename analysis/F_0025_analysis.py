import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts
import padeopsIO as pio
import math
from scipy.stats import describe
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import padeopsIO.utils.io_utils as io

def phase_average_periods(df):
    phase_grid = np.round(np.linspace(0, 0.99, 100), decimals = 2)
    surge_amp = df["Surge_Amplitude"].iloc[0]
    pitch_amp = df["Pitch_Amplitude"].iloc[0]
    cols_interp = ["Tilt","UTurb","DeltaX","Power","UDisk"]
    # stationary
    if surge_amp == 0 and pitch_amp == 0:
        final_df = pd.DataFrame({"Phase": phase_grid})
        final_df = pd.DataFrame({"Phase": np.tile(phase_grid, 8)})
        for col in cols_interp:
            mean_val = df[col].mean()
            std_val = df[col].std()
            # final_df[f"{col}_mean"] = np.full(100, mean_val)
            # final_df[f"{col}_std"] = np.full(100, std_val)
            final_df[f"{col}"] = np.full(800, mean_val)
            final_df[f"{col}"] = np.full(800, std_val)

        meta_cols = ["id","CT_prime","Surge_Amplitude", "Pitch_Amplitude", "Frequency",
                     "nx","ny","nz","Lx","Ly","Lz","dt",
                     "filterWidth","useCorrection"]
        for c in meta_cols:
            final_df[c] = df[c].iloc[0]
        return final_df

    # --- detect periods from phase wrapping ---
    phase = df["Phase"].values
    period = np.zeros(len(phase), dtype=int)
    period[1:] = np.cumsum(np.diff(phase) < 0)
    df = df.copy()
    df["Period"] = period

    # --- remove first and last periods ---
    unique_periods = np.unique(period)
    if len(unique_periods) < 10:
        raise ValueError("Not enough periods to remove first and last and average over 8")
    valid_periods = unique_periods[1:9]
    df = df[df["Period"].isin(valid_periods)]

    # --- interpolate each period ---
    # interp_periods = []
    # for p, g in df.groupby("Period"):
    #     g = g.sort_values("Phase")
    #     ph = g["Phase"].values
    #     # periodic extension
    #     ph_ext = np.concatenate([ph - 1, ph, ph + 1])
    #     interp_dict = {
    #         "Phase": phase_grid,
    #         "Period": np.full(len(phase_grid), p)
    #     }
    #     for col in cols_interp:
    #         vals = g[col].values
    #         vals_ext = np.concatenate([vals, vals, vals])
    #         interp_dict[col] = np.interp(phase_grid, ph_ext, vals_ext)
    #     interp_periods.append(pd.DataFrame(interp_dict))
    # interp_df = pd.concat(interp_periods, ignore_index=True)
    # --- check number of periods ---
    # nperiods = interp_df["Period"].nunique()
    # print(f"Number of usable periods: {nperiods}")
    # --- compute phase averages ---
    # final_df = interp_df.copy()
    # phase_mean = interp_df.groupby("Phase").mean(numeric_only=True)
    # phase_std = interp_df.groupby("Phase").std(numeric_only=True)
    # final_df = pd.DataFrame({
    #     "Phase": phase_grid
    # })

    # for col in cols_interp:
    #     final_df[f"{col}_mean"] = phase_mean[col].values
    #     final_df[f"{col}_std"] = phase_std[col].values

    # copy metadata
    # meta_cols = ["id","CT_prime","Surge_Amplitude", "Pitch_Amplitude", "Frequency",
    #              "nx","ny","nz","Lx","Ly","Lz","dt",
    #              "filterWidth","useCorrection"]

    # for c in meta_cols:
    #     final_df[c] = df[c].iloc[0]
    final_df = df
    return final_df


uinf = 1
rho = 1
data_path = Path(au.DATA_PATH)
sim_all_folder = os.path.join(au.DATA_PATH, "F_0025_SU_Files")
all_data_fn = os.path.join(sim_all_folder, 'collected_runs_all.csv')

all_df = []
rows, fields = mplts.get_sim_varied_params(sim_all_folder)
ids,cT,surge_amplitude,surge_freq,nx,ny,nz,Lx,Ly,Lz,dt,tstop,filterWidth,useCorrection = zip(*rows) 
pitch_amplitude = np.zeros_like(surge_amplitude)

# get data from runs
row = 0
for (i, id_str) in enumerate(ids):
    try:
        run_folder = os.path.join(sim_all_folder, "Sim_" + id_str)
        sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")
        # get all of the needed values
        power = sim.read_turb_power("all", turb=1)
        uvel = sim.read_turb_uvel("all", turb = 1)
        n_dumped = np.minimum(len(power), len(uvel))
        power, uvel = power[:n_dumped], uvel[:n_dumped]
        assert len(power) > 0

        log_file = glob.glob(f'*{id_str}*.o[0-9]*', root_dir = run_folder, recursive = False)
        if not log_file: # if run in a batch, might need to extract log information
            extracted = au.extract_sim_log_from_batches(run_folder)
            if extracted:
                log_file = [extracted.name]
        assert len(log_file) == 1

        # get values from log file
        log_file_dict = pio.query_logfile(os.path.join(run_folder, log_file[0]), search_terms=["tilt", "uturb", "Time", "TIDX", "delta", "phase"], crop_equal = False)
        tidx = np.insert(log_file_dict["TIDX"], 0, 0).astype(int)
        time = log_file_dict["Time"]
        assert len(tidx) > 0

        time, tidx = time[:n_dumped], tidx[:n_dumped]
        time_mask = time > 100
        time, tidx = time[time_mask], tidx[time_mask]
        power, uvel = power[time_mask], uvel[time_mask]
        assert len(power) > 0

        tilt = log_file_dict["tilt"][:n_dumped][time_mask]
        tilt = np.deg2rad(tilt)
        uturb = log_file_dict["uturb"][:n_dumped][time_mask]
        dx = log_file_dict["delta"][:n_dumped][time_mask]
        phase = log_file_dict["phase"][:n_dumped][time_mask]

    except Exception as e:
        print(f"Error occurred in {id_str}: {e}")
        continue
    else:
        # Add needed timestep values to "all" dataframe
        print(id_str)
        # calcualte period - increment period whenever phase jumps backwards
        period = np.zeros(len(phase), dtype=int)
        period[1:] = np.cumsum(np.diff(phase) < 0)

        # add values to a dataframe
        nsamples = len(time)
        df_temp = pd.DataFrame({
            "id": int(id_str),
            "CT_prime": np.full(nsamples, float(cT[i])),
            "Surge_Amplitude": np.full(nsamples, float(surge_amplitude[i])),
            "Pitch_Amplitude": np.zeros(nsamples),
            "Frequency": np.full(nsamples, float(surge_freq[i])),
            "nx": np.full(nsamples, float(nx[i])),
            "ny": np.full(nsamples, float(ny[i])),
            "nz": np.full(nsamples, float(nz[i])),
            "Lx": np.full(nsamples, float(Lx[i])),
            "Ly": np.full(nsamples, float(Ly[i])),
            "Lz": np.full(nsamples, float(Lz[i])),
            "dt": np.full(nsamples, float(dt[i])),
            "filterWidth": np.full(nsamples, float(filterWidth[i])),
            "useCorrection": np.full(nsamples, useCorrection[i]),
            "Tilt": tilt,
            "UTurb": uturb,
            "DeltaX": dx,
            "Phase": phase,
            "Time": time,
            "TIDX": tidx,
            "Power": power,
            "UDisk": uvel,
        })
        df_phase_avg = phase_average_periods(df_temp)
        all_df.append(df_phase_avg)

print("Making CSV")
print(len(all_df))
all_df = pd.concat(all_df, ignore_index=True)
all_df.to_csv(all_data_fn)

# save the flow field for A = 0.2, f = 0.2, CT' = 1.33
sim25_run14_plt_folder = mplts.plot_instantaneous_field(sim_all_folder, 14, tidx = "all", field = "u", dpi = 300)
mplts.film_instantaneous_field(sim25_run14_plt_folder, fps = 10, video_name = "u_stability.mp4")

# # save the flow field for A = 0.2, f = 0.2, CT' = 2.66
# sim25_run15_plt_folder = mplts.plot_instantaneous_field(sim_all_folder, 15, tidx = "all", field = "u", dpi = 300)
# mplts.film_instantaneous_field(sim25_run15_plt_folder, fps = 10, video_name = "u_stability.mp4")

# # save the flow field for A = 0.6, f = 0.2, CT' = 2.66
# sim25_run17_plt_folder = mplts.plot_instantaneous_field(sim_all_folder, 17, tidx = "all", field = "u", dpi = 300)
# mplts.film_instantaneous_field(sim25_run17_plt_folder, fps = 10, video_name = "u_stability.mp4")

# # save the flow field for A = 0.6, f = 2.0, CT' = 2.66
# sim25_run25_plt_folder = mplts.plot_instantaneous_field(sim_all_folder, 25, tidx = "all", field = "u", dpi = 300)
# mplts.film_instantaneous_field(sim25_run25_plt_folder, fps = 10, video_name = "u_stability.mp4")

