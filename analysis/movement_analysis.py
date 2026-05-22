import analysis.lib.analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import analysis.lib.quick_metadata_plots as mplts
import padeopsIO as pio
import math
from scipy.stats import describe
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import padeopsIO.utils.io_utils as io

get_stats = False
get_all_points = True
get_centerlines = False

def get_sim_stats(Cp_vals, an_vals, cT_vals):
    # get statistics on calcualted values
    if len(Cp_vals) > 0:
        cT_stats = describe(cT_vals)
        an_stats = describe(an_vals)
        Cp_stats = describe(Cp_vals)
        mean_info = [cT_stats.mean, an_stats.mean, Cp_stats.mean]
        variance_info = [cT_stats.variance, an_stats.variance, Cp_stats.variance]
        std_info = [np.std(cT_vals), np.std(an_vals), np.std(Cp_vals)]
        skewness_info = [cT_stats.skewness, an_stats.skewness, Cp_stats.skewness]
        kurtosis_info = [cT_stats.kurtosis, an_stats.kurtosis, Cp_stats.kurtosis]
    else:
        mean_info = [np.nan, np.nan, np.nan]
        variance_info = mean_info
        std_info = mean_info
        skewness_info = mean_info
        kurtosis_info = mean_info
    return mean_info, std_info, variance_info, skewness_info, kurtosis_info


uinf = 1
rho = 1

data_path = Path(au.DATA_PATH)
# sim_all_folder = os.path.join(au.DATA_PATH, "F_0016_SU_PI_Files")
# sim_all_folder = os.path.join(au.DATA_PATH, "F_0020_SU_PI_Files")
sim_all_folder = os.path.join(au.DATA_PATH, "F_0022_SU_PI_Files")
# sim_all_folder = os.path.join(au.DATA_PATH, "F_0023_SU_Files")

stats_data_fn = os.path.join(sim_all_folder, 'collected_runs_stats.csv')
all_data_fn = os.path.join(sim_all_folder, 'collected_runs_all.csv')
surge_centerline_fn = os.path.join(sim_all_folder, 'surge_centerline.csv')

# # # go through data and collect
stats_df = pd.DataFrame(columns=['marker', 'dt', 'nx', 'ny', 'filter', 'filterFactor', 'useCorrection', 'CT_prime', "turbulence",
                           'surge_freq', 'surge_amplitude', 'pitch_amplitude',
                           'mean_CT_ground','mean_an_ground','mean_Cp_ground',
                           'std_CT_ground', 'std_an_ground', 'std_Cp_ground',
                           'skewness_CT_ground', 'skewness_an_ground', 'skewness_Cp_ground',
                           'kurtosis_CT_ground', 'kurtosis_an_ground', 'kurtosis_Cp_ground',
                           'mean_CT_turb','mean_an_turb','mean_Cp_turb',
                           'std_CT_turb', 'std_an_turb', 'std_Cp_turb',
                           'skewness_CT_turb', 'skewness_an_turb', 'skewness_Cp_turb',
                           'kurtosis_CT_turb', 'kurtosis_an_turb', 'kurtosis_Cp_turb'])

all_df = []
surge_centerline_df = []

rows, fields = mplts.get_sim_varied_params(sim_all_folder)

# ids,cT,surge_freq,surge_amplitude,pitch_amplitude,dt,filterWidth = zip(*rows)
# ny, nz = [256] * len(ids), [256] * len(ids)

ids,cT,surge_freq,surge_amplitude,pitch_amplitude,dt,n_hrs,filterWidth,ny,nz = zip(*rows) # FOR 22

nx = [256] * len(ids)
useCorrection = [True] * len(ids)
turbulence = [False] * len(ids)

# get data from runs
row = 0
for (i, id_str) in enumerate(ids):
    run_folder = os.path.join(sim_all_folder, "Sim_" + id_str)
    surge_not_pitch = float(surge_amplitude[i]) != 0 and float(pitch_amplitude[i]) == 0
    pitch_not_surge = (float(surge_amplitude[i]) == 0) and (float(pitch_amplitude[i]) != 0)
    stationary = float(surge_amplitude[i]) == 0 and  float(pitch_amplitude[i]) == 0
    if surge_not_pitch:
        amp = float(surge_amplitude[i])
        movement = "Surge"
        pitch_amp = 0
    elif stationary:
        amp = 0
        movement = "Stationary"
        pitch_amp = 0
    elif pitch_not_surge:
        amp = float(pitch_amplitude[i])
        movement = "Pitch"
        pitch_amp = float(pitch_amplitude[i])
    else: # both pitch and surge
        amp = float(surge_amplitude[i])
        movement = "Surge and Pitch"
        pitch_amp = float(pitch_amplitude[i])
    
    try:
        sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")
        # get all of the needed values
        power = sim.read_turb_power("all", turb=1)#[time_mask]
        uvel = sim.read_turb_uvel("all", turb = 1)#[time_mask]
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
        log_file_dict = pio.query_logfile(os.path.join(run_folder, log_file[0]), search_terms=["tilt", "uturb", "Time", "TIDX", "delta"], crop_equal = False)
        tidx = np.insert(log_file_dict["TIDX"], 0, 0).astype(int)
        time = log_file_dict["Time"]
        assert len(tidx) > 0

        time, tidx = time[:n_dumped], tidx[:n_dumped]
        time_mask = time > 100
        time, tidx = time[time_mask], tidx[time_mask]
        power, uvel = power[time_mask], uvel[time_mask]
        assert len(power) > 0

        if surge_not_pitch or stationary:
            tilt = np.zeros_like(uvel)
        else:
            tilt = log_file_dict["tilt"][:n_dumped][time_mask]
            tilt = np.deg2rad(tilt)
        uturb, dx = log_file_dict["uturb"][:n_dumped][time_mask], log_file_dict["delta"][:n_dumped][time_mask]

        udisk = uvel + uturb
        uwind = uinf - uturb

    except Exception as e:
        print(f"Error occurred in {id_str}: {e}")
        continue
    else:
        # Add needed timestep values to "all" dataframe
        print(id_str)
        cT_prime_val = float(cT[i])
        dt_val = float(dt[i])
        h = ((25 / float(nx[i]))**2 + 2 * (10 / float(ny[i]))**2)**(1/2)
        filter_width = float(filterWidth[i])

        if get_stats:
            # # ground perspective
            Cp_vals_ground = power / (0.5 * rho * math.pi * 0.5**2 * (uinf * np.cos(tilt))**3)
            an_vals_ground = 1 - (udisk / (uinf * np.cos(tilt)))
            cT_vals_ground = [cT_prime_val * (1 - an)**2 for an in an_vals_ground]
            # turbine perspective
            Cp_vals_turb = power / (0.5 * rho * math.pi * 0.5**2 * (uwind * np.cos(tilt))**3)
            an_vals_turb = 1 - (uvel / (uwind  * np.cos(tilt)))
            cT_vals_turb = [cT_prime_val * (1 - an)**2 for an in an_vals_turb]

            if float(surge_freq[i]) == 0:
                marker = "o"
            elif float(surge_amplitude[i]) != 0:
                marker = "s"
            else:
                marker = "^"
            filter_factor = round(filter_width / h, 3)
            turbulence = False

            mean_info_ground, std_info_ground, variance_info_ground, skewness_info_ground, kurtosis_info_ground = get_sim_stats(Cp_vals_ground, an_vals_ground, cT_vals_ground)
            mean_info_turb, std_info_turb, variance_info_turb, skewness_info_turb, kurtosis_info_turb = get_sim_stats(Cp_vals_turb, an_vals_turb, cT_vals_turb)

            simulation_info = [marker, dt_val, float(nx[i]), float(ny[i]), filter_width, filter_factor, useCorrection[i], cT_prime_val, turbulence, surge_freq[i], surge_amplitude[i], pitch_amplitude[i]]
            ground_frame_info = [*mean_info_ground, *std_info_ground, *skewness_info_ground, *kurtosis_info_ground]
            turb_frame_info =  [*mean_info_turb, *std_info_turb, *skewness_info_turb, *kurtosis_info_turb]
            stats_df.loc[row] = np.concatenate((simulation_info, ground_frame_info, turb_frame_info))
            row += 1

        # add values to a dataframe
        if get_all_points:
            nsamples = len(time)
            df_temp = pd.DataFrame({
                "id": int(id_str),
                "Movement": np.full(nsamples, movement),
                "Frequency": np.full(nsamples, float(surge_freq[i])),
                "Amplitude": np.full(nsamples, amp),
                "Thrust Coefficient": np.full(nsamples, cT_prime_val),
                "Tilt": tilt,
                "UTurb": uturb,
                "DeltaX": dx,
                "Time": time,
                "TIDX": tidx,
                "Power": power,
                "UDisk": uvel,
                "PitchAmp": pitch_amp,
                "dt": dt_val,
                "filterWidth": filter_width,
                "nx": float(nx[i]),
                "ny": float(ny[i]),
                "nz": float(nz[i]),
            })
            print(float(nz[i]))
            all_df.append(df_temp)

        # add surge centerline to dataframe
        if get_centerlines and surge_not_pitch:
            field_tidx_vals = sim.unique_tidx()
            field_time_vals = sim.unique_times()
            for (j, field_tidx) in enumerate(field_tidx_vals):
                field_time = field_time_vals[j]
                if field_time > 200:
                    ds = sim.slice(field_terms=["u", "x"], xlim = [-4, 4], ylim = 0, zlim = 0, tidx = field_tidx)
                    nsamples = len(ds.x)
                    field_uturb = uturb[np.where(tidx == field_tidx)[0][0]]
                    centerline_temp = pd.DataFrame({
                        "Frequency": np.full(nsamples, float(surge_freq[i])),
                        "Amplitude": np.full(nsamples, amp),
                        "Thrust Coefficient": np.full(nsamples, cT_prime_val),
                        "Time": np.full(nsamples, field_time),
                        "TIDX": np.full(nsamples, field_tidx),
                        "xVals": ds.x,
                        "uVals": ds.u,
                        "UTurb": np.full(nsamples, field_uturb),
                    })
                    surge_centerline_df.append(centerline_temp)

# saving the dataframes
if get_stats:
    stats_df.to_csv(stats_data_fn)

if get_all_points:
    print("Making CSV")
    print(len(all_df))
    all_df = pd.concat(all_df, ignore_index=True)
    all_df.to_csv(all_data_fn)

if get_centerlines:
    surge_centerline_df = pd.concat(surge_centerline_df, ignore_index = True)
    surge_centerline_df.to_csv(surge_centerline_fn)