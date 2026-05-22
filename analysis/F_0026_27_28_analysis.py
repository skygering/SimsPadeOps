import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts
import padeopsIO as pio
from scipy.stats import describe
import numpy as np
import pandas as pd
import glob
import padeopsIO.utils.io_utils as io

# def phase_average_periods(df, nperiods = 8):

#     phase_grid = np.linspace(0, 1, 100, endpoint=False)

#     cols_interp = ["Tilt","UTurb","DeltaX","Power","UDisk"]

#     surge_amp = df["Surge_Amplitude"].iloc[0]
#     pitch_amp = df["Pitch_Amplitude"].iloc[0]

#     # -----------------------
#     # stationary turbine
#     # -----------------------
#     if surge_amp == 0 and pitch_amp == 0:

#         final_df = pd.DataFrame({
#             "Phase": np.tile(phase_grid, nperiods),
#             "Period": np.repeat(np.arange(nperiods), 100)
#         })

#         for col in cols_interp:
#             final_df[col] = df[col].mean()

#         meta_cols = [
#             "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
#             "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
#         ]

#         for c in meta_cols:
#             final_df[c] = df[c].iloc[0]

#         return final_df


#     # -----------------------
#     # detect periods
#     # -----------------------
#     phase = df["Phase"].values

#     period = np.zeros(len(phase), dtype=int)
#     period[1:] = np.cumsum(np.diff(phase) < 0)

#     df = df.copy()
#     df["Period"] = period


#     # -----------------------
#     # keep 8 periods
#     # -----------------------
#     unique_periods = np.unique(period)
#     print(len(unique_periods))
#     if len(unique_periods) < nperiods + 2:
#         raise ValueError(f"Need {nperiods + 2} periods")

#     valid_periods = unique_periods[1:(nperiods+1)]
#     df = df[df["Period"].isin(valid_periods)]


#     # -----------------------
#     # interpolate each period
#     # -----------------------
#     interp_periods = []

#     for p, g in df.groupby("Period"):

#         g = g.sort_values("Phase")

#         ph = g["Phase"].values

#         # periodic extension
#         ph_ext = np.concatenate([ph - 1, ph, ph + 1])

#         interp_dict = {
#             "Phase": phase_grid,
#             "Period": np.full(len(phase_grid), p)
#         }

#         for col in cols_interp:

#             vals = g[col].values
#             vals_ext = np.concatenate([vals, vals, vals])
#             interp_dict[col] = np.interp(phase_grid, ph_ext, vals_ext)

#         interp_periods.append(pd.DataFrame(interp_dict))

#     final_df = pd.concat(interp_periods, ignore_index=True)


#     # -----------------------
#     # metadata
#     # -----------------------
#     meta_cols = [
#         "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
#         "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
#     ]

#     for c in meta_cols:
#         final_df[c] = df[c].iloc[0]
#     return final_df

def phase_average_periods(df, nperiods=8, n_bins=100):
    """
    Phase-average using rounding (nearest phase bin).
    Returns exactly n_bins points (default 100) per simulation.

    - No interpolation
    - Averages across all selected periods
    - Robust to irregular phase spacing
    """

    cols_to_avg = ["Tilt", "UTurb", "DeltaX", "Power", "UDisk"]

    # -----------------------
    # stationary turbine
    # -----------------------
    surge_amp = df["Surge_Amplitude"].iloc[0]
    pitch_amp = df["Pitch_Amplitude"].iloc[0]

    phase_grid = np.round(np.linspace(0, 1, n_bins, endpoint=False), 2)

    if surge_amp == 0 and pitch_amp == 0:
        final_df = pd.DataFrame({"Phase": phase_grid})

        for col in cols_to_avg:
            final_df[f"{col}_mean"] = df[col].mean()
            final_df[f"{col}_std"] = df[col].std()

        # metadata
        for c in [
            "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
            "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
        ]:
            final_df[c] = df[c].iloc[0]

        return final_df

    # -----------------------
    # detect periods
    # -----------------------
    phase = df["Phase"].values
    period = np.zeros(len(phase), dtype=int)
    period[1:] = np.cumsum(np.diff(phase) < 0)

    df = df.copy()
    df["Period"] = period

    # -----------------------
    # keep desired periods
    # -----------------------
    unique_periods = np.unique(period)
    if len(unique_periods) < nperiods + 2:
        raise ValueError(f"Need at least {nperiods + 2} periods")

    valid_periods = unique_periods[1:(nperiods+1)]
    df = df[df["Period"].isin(valid_periods)]

    # -----------------------
    # ROUND phase to bins
    # -----------------------
    dphi = 1.0 / n_bins

    phase = np.mod(df["Phase"].values, 1.0)
    phase_bin = np.round(phase / dphi) * dphi
    phase_bin = np.mod(phase_bin, 1.0)

    # avoid float precision issues
    phase_bin = np.round(phase_bin, 4)

    df["Phase_bin"] = phase_bin

    # -----------------------
    # GROUP + average across ALL periods
    # -----------------------
    grouped = df.groupby("Phase_bin", observed=True)[cols_to_avg].agg(["mean", "std"])

    # flatten columns
    grouped.columns = [
        f"{col}_{stat}" for col, stat in grouped.columns
    ]

    grouped = grouped.reset_index().rename(columns={"Phase_bin": "Phase"})

    # -----------------------
    # enforce full 100-point grid
    # -----------------------
    grouped = grouped.set_index("Phase").reindex(phase_grid).reset_index()

    # -----------------------
    # metadata
    # -----------------------
    for c in [
        "id","CT_prime","Surge_Amplitude","Pitch_Amplitude","Frequency",
        "nx","ny","nz","Lx","Ly","Lz","dt","filterWidth","useCorrection"
    ]:
        grouped[c] = df[c].iloc[0]

    return grouped


def load_simulation_timeseries(run_folder, id_str, freq, spinup_time=75):
    sim = pio.BudgetIO(run_folder, padeops=True, runid=0, normalize_origin="turbine")

    power = sim.read_turb_power("all", turb=1)
    uvel = sim.read_turb_uvel("all", turb=1)

    n_dumped = min(len(power), len(uvel))
    power, uvel = power[:n_dumped], uvel[:n_dumped]

    if len(power) == 0:
        raise RuntimeError("No turbine data found")

    # locate log file
    log_file = glob.glob(f'*{id_str}*.o[0-9]*', root_dir=run_folder)

    if not log_file:
        extracted = au.extract_sim_log_from_batches(run_folder)
        if extracted:
            log_file = [extracted.name]

    if len(log_file) != 1:
        raise RuntimeError("Could not uniquely locate log file")

    logfile_path = os.path.join(run_folder, log_file[0])

    log = pio.query_logfile(
        logfile_path,
        search_terms=["tilt", "uturb", "Time", "TIDX", "delta", "phase"],
        crop_equal=False
    )

    time = log["Time"][:n_dumped]
    tidx = np.insert(log["TIDX"], 0, 0).astype(int)[:n_dumped]

    mask = time > spinup_time
    time, tidx = time[mask], tidx[mask]

    # these might not be included
    tilt_deg = log["tilt"]
    if len(tilt_deg) == 0:
        tilt = np.zeros_like(time)
    else:
        tilt = np.deg2rad(tilt_deg[:n_dumped][mask])
    phase = log["phase"]
    if len(phase) == 0:
        if freq > 0:
            T = 1 / freq
            dt = time[1] - time[0]
            dp = dt / T
            start_phase = np.mod(time[0] / T, 1)
            phase = np.mod(start_phase + np.arange(len(time)) * dp, 1)
        else:
            phase = np.zeros_like(time)
    else:
        phase = log["phase"][:n_dumped][mask]
    data = {
        "Time": time,
        "TIDX": tidx,
        "Tilt": tilt,
        "UTurb": log["uturb"][:n_dumped][mask],
        "DeltaX": log["delta"][:n_dumped][mask],
        "Phase": phase,
        "Power": power[mask],
        "UDisk": uvel[mask],
    }
    return data

def build_sim_dataframe(data, meta):
    df = pd.DataFrame({
        "id": meta["id"],
        "CT_prime": meta["CT_prime"],
        "Surge_Amplitude": meta["Surge_Amplitude"],
        "Pitch_Amplitude": meta["Pitch_Amplitude"],
        "Frequency": meta["Frequency"],
        "nx": meta["nx"],
        "ny": meta["ny"],
        "nz": meta["nz"],
        "Lx": meta["Lx"],
        "Ly": meta["Ly"],
        "Lz": meta["Lz"],
        "dt": meta["dt"],
        "filterWidth": meta["filterWidth"],
        "useCorrection": meta["useCorrection"],
        **data
    })

    # detect periods
    phase = df["Phase"].values
    period = np.zeros(len(phase), dtype=int)
    period[1:] = np.cumsum(np.diff(phase) < 0)
    df["Period"] = period
    return df


uinf = 1
rho = 1
data_path = Path(au.DATA_PATH)

sim_all_folder = os.path.join(au.DATA_PATH, "F_0029_SU_Files")
all_data_fn = os.path.join(sim_all_folder, 'collected_runs_all.csv')
nperiods = 8

all_df = []
rows, fields = mplts.get_sim_varied_params(sim_all_folder)
# id,cT,surge_amplitude,surge_freq,nx,ny,nz,Lx,Ly,Lz,dt,tstop,n_hrs,filterWidth,useCorrection = zip(*rows) 
id,cT,surge_amplitude,surge_freq,nx,ny,nz,Lx,Ly,Lz,dt,tstop,n_hrs,filterWidth,useCorrection = zip(*rows) # for 29
# id,cT,surge_amplitude,surge_freq,nx,ny,nz,Lx,Ly,Lz,dt,tstop,n_hrs,filterWidth,useCorrection,yLoc,zLoc = zip(*rows) # for 27
pitch_amplitude = np.zeros_like(surge_amplitude)

# for i, id_str in enumerate(id):
#     try:
#         print(id_str)
#         run_folder = os.path.join(sim_all_folder, f"Sim_{id_str}")
#         freq = float(surge_freq[i])
#         data = load_simulation_timeseries(run_folder, id_str, freq, spinup_time = 75)
#         meta = {
#             "id": int(id_str),
#             "CT_prime": float(cT[i]),
#             "Surge_Amplitude": float(surge_amplitude[i]),
#             "Pitch_Amplitude": 0.0,
#             "Frequency": freq,
#             "nx": float(nx[i]),
#             "ny": float(ny[i]),
#             "nz": float(nz[i]),
#             "Lx": float(Lx[i]),
#             "Ly": float(Ly[i]),
#             "Lz": float(Lz[i]),
#             "dt": float(dt[i]),
#             "filterWidth": float(filterWidth[i]),
#             "useCorrection": useCorrection[i],
#         }
#         df_temp = build_sim_dataframe(data, meta)
#         df_phase = phase_average_periods(df_temp, nperiods=nperiods)
#         all_df.append(df_phase)
#     except Exception as e:
#         print(f"Error occurred in {id_str}: {e}")

        
# print("Making CSV")
# print(len(all_df))
# all_df = pd.concat(all_df, ignore_index=True)
# all_df.to_csv(all_data_fn)

# print("Making Period-Average CSV")
# period_ids = [0, 1, 7, 8]
# nperiod_list = [4, 6, 8, 10]

# nperiod_df = []
# for i, id_str in enumerate(id):
#     if int(id_str) not in period_ids:
#         continue

#     for nperiods in nperiod_list:
#         try:
#             print(f"id {id_str}, periods {nperiods}")
#             run_folder = os.path.join(sim_all_folder, f"Sim_{id_str}")
#             data = load_simulation_timeseries(
#                 run_folder,
#                 id_str,
#                 spinup_time=75
#             )
#             meta = {
#                 "id": int(id_str),
#                 "CT_prime": float(cT[i]),
#                 "Surge_Amplitude": float(surge_amplitude[i]),
#                 "Pitch_Amplitude": 0.0,
#                 "Frequency": float(surge_freq[i]),
#                 "nx": float(nx[i]),
#                 "ny": float(ny[i]),
#                 "nz": float(nz[i]),
#                 "Lx": float(Lx[i]),
#                 "Ly": float(Ly[i]),
#                 "Lz": float(Lz[i]),
#                 "dt": float(dt[i]),
#                 "filterWidth": float(filterWidth[i]),
#                 "useCorrection": useCorrection[i],
#             }
#             df_temp = build_sim_dataframe(data, meta)
#             df_phase = phase_average_periods(df_temp, nperiods=nperiods)
#             # add spinup column
#             df_phase["nperiods"] = nperiods
#             nperiod_df.append(df_phase)
#         except Exception as e:
#             print(f"Error occurred in {id_str} (spinup {nperiods}): {e}")

# print("Making spinup CSV")
# nperiod_df = pd.concat(nperiod_df, ignore_index=True)
# nperiod_fn = os.path.join(sim_all_folder, "surge_nperiod_avg.csv")
# nperiod_df.to_csv(nperiod_fn, index=False)
# print("Saved:", nperiod_fn)

# print("Making Spinup CSV")
# spinup_ids = [0, 1, 7, 8]
# spinup_times = [50, 75, 100]

# spinup_df = []
# for i, id_str in enumerate(id):
#     if int(id_str) not in spinup_ids:
#         continue

#     for spinup_time in spinup_times:
#         try:
#             print(f"id {id_str}, spinup {spinup_time}")
#             run_folder = os.path.join(sim_all_folder, f"Sim_{id_str}")
#             data = load_simulation_timeseries(
#                 run_folder,
#                 id_str,
#                 spinup_time=spinup_time
#             )
#             meta = {
#                 "id": int(id_str),
#                 "CT_prime": float(cT[i]),
#                 "Surge_Amplitude": float(surge_amplitude[i]),
#                 "Pitch_Amplitude": 0.0,
#                 "Frequency": float(surge_freq[i]),
#                 "nx": float(nx[i]),
#                 "ny": float(ny[i]),
#                 "nz": float(nz[i]),
#                 "Lx": float(Lx[i]),
#                 "Ly": float(Ly[i]),
#                 "Lz": float(Lz[i]),
#                 "dt": float(dt[i]),
#                 "filterWidth": float(filterWidth[i]),
#                 "useCorrection": useCorrection[i],
#             }
#             df_temp = build_sim_dataframe(data, meta)
#             df_phase = phase_average_periods(df_temp, nperiods=nperiods)
#             # add spinup column
#             df_phase["spinup_time"] = spinup_time
#             spinup_df.append(df_phase)
#         except Exception as e:
#             print(f"Error occurred in {id_str} (spinup {spinup_time}): {e}")

# print("Making spinup CSV")
# spinup_df = pd.concat(spinup_df, ignore_index=True)
# spinup_fn = os.path.join(sim_all_folder, "surge_spinup_time.csv")
# spinup_df.to_csv(spinup_fn, index=False)
# print("Saved:", spinup_fn)
xlim = [-2.5, 7.5]
ylim = [-3, 3]
zlim = 0
run3_plt_folder = mplts.plot_instantaneous_field(sim_all_folder, 12, tidx = "all", field = "u", xlim=xlim, ylim=ylim, zlim=0, vmin = 0, vmax = 2, dpi = 300)
mplts.film_instantaneous_field(run3_plt_folder, fps = 10, video_name = "u_stability.mp4")