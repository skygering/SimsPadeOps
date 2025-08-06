import analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import quick_metadata_plots as mplts
import padeopsIO as pio
import math
from scipy.stats import describe
from numpy import std
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import glob

def get_stats(Cp_vals, an_vals, cT_vals):
    # get statistics on calcualted values
    if len(Cp_vals) > 0:
        cT_stats = describe(cT_vals)
        an_stats = describe(an_vals)
        Cp_stats = describe(Cp_vals)
        mean_info = [cT_stats.mean, an_stats.mean, Cp_stats.mean]
        variance_info = [cT_stats.variance, an_stats.variance, Cp_stats.variance]
        std_info = [std(cT_vals), std(an_vals), std(Cp_vals)]
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
sim_16_all_folder = os.path.join(au.DATA_PATH, "F_0016_SU_PI_Files")
data_fn = os.path.join(sim_16_all_folder, 'collected_runs.csv')

# # go through data and collect
df = pd.DataFrame(columns=['marker', 'dt', 'nx', 'ny', 'filter', 'filterFactor', 'useCorrection', 'CT_prime', "turbulence",
                           'surge_freq', 'surge_amplitude', 'pitch_amplitude',
                           'mean_CT_ground','mean_an_ground','mean_Cp_ground',
                           'std_CT_ground', 'std_an_ground', 'std_Cp_ground',
                           'skewness_CT_ground', 'skewness_an_ground', 'skewness_Cp_ground',
                           'kurtosis_CT_ground', 'kurtosis_an_ground', 'kurtosis_Cp_ground',
                           'mean_CT_turb','mean_an_turb','mean_Cp_turb',
                           'std_CT_turb', 'std_an_turb', 'std_Cp_turb',
                           'skewness_CT_turb', 'skewness_an_turb', 'skewness_Cp_turb',
                           'kurtosis_CT_turb', 'kurtosis_an_turb', 'kurtosis_Cp_turb'])

rows, fields = mplts.get_sim_varied_params(sim_16_all_folder)
ids,cT,surge_freq,surge_amplitude,pitch_amplitude,dt,filterWidth = zip(*rows)
nx, ny, nz = [256] * len(ids), [256] * len(ids), [256] * len(ids)
useCorrection = [True] * len(ids)
turbulence = [False] * len(ids)

# create plots
unique_ctp = np.unique(cT)
unique_f = np.unique(surge_freq)
unique_surge_A = np.unique(surge_amplitude)
unique_pitch_A = np.unique(pitch_amplitude)
# 

# get data from runs
row = 0
for (i, id_str) in enumerate(ids):
    run_folder = os.path.join(sim_16_all_folder, "Sim_" + id_str)
    sim = pio.BudgetIO(run_folder, padeops = True, runid = 0)
    dt_i = dt[i]
    trans_tau = int(math.ceil(100 / float(dt_i)) + 1)
    try:
    # get all of the needed values
        power = sim.read_turb_power("all", turb=1)[trans_tau:]
        uvel = sim.read_turb_uvel("all", turb = 1)[trans_tau:]
        assert len(power) > 0
        log_file = glob.glob(f'*_{id_str}.o*', root_dir = run_folder, recursive = False)
        if len(log_file) > 0:
            log_file_dict = pio.query_logfile(os.path.join(run_folder, log_file[0]), search_terms=["tilt", "uturb"], crop_equal = False)
            tilt, uturb = log_file_dict["tilt"][trans_tau:], log_file_dict["uturb"][trans_tau:]
            if len(tilt) == 0:
                tilt = np.zeros_like(uturb)
        else:
            tilt = np.zeros_like(uvel)
            uturb = np.zeros_like(uvel)
        udisk = uvel + uturb
        uwind = uinf - uturb
    except: 
        continue
    else:
        print(id_str)
        h = ((25 / float(nx[i]))**2 + 2 * (10 / float(ny[i]))**2)**(1/2)
        filter_width = float(filterWidth[i])
        dt_val = float(dt[i])
        cT_prime_val = float(cT[i])
        # ground perspective
        Cp_vals_ground = power / (0.5 * rho * math.pi * 0.5**2 * (uinf * np.cos(tilt))**3)
        an_vals_ground = 1 - (udisk / (uinf * np.cos(tilt)))
        cT_vals_ground = [cT_prime_val * (1 - an)**2 for an in an_vals_ground]
        # turbine perspective
        Cp_vals_turb = power / (0.5 * rho * math.pi * 0.5**2 * (uwind * np.cos(tilt))**3)
        an_vals_turb = 1 - (uvel / (uwind  * np.cos(tilt)))
        cT_vals_turb = [cT_prime_val * (1 - an)**2 for an in an_vals_turb]

        # save sim info
        if float(surge_freq[i]) == 0:
            marker = "o"
        elif float(surge_amplitude[i]) != 0:
            marker = "s"
        else:
            marker = "^"
        filter_factor = round(filter_width / h, 3)
        turbulence = False

        mean_info_ground, std_info_ground, variance_info_ground, skewness_info_ground, kurtosis_info_ground = get_stats(Cp_vals_ground, an_vals_ground, cT_vals_ground)
        mean_info_turb, std_info_turb, variance_info_turb, skewness_info_turb, kurtosis_info_turb = get_stats(Cp_vals_turb, an_vals_turb, cT_vals_turb)

        simulation_info = [marker, dt_val, float(nx[i]), float(ny[i]), filter_width, filter_factor, useCorrection[i], cT_prime_val, turbulence, surge_freq[i], surge_amplitude[i], pitch_amplitude[i]]
        ground_frame_info = [*mean_info_ground, *std_info_ground, *skewness_info_ground, *kurtosis_info_ground]
        turb_frame_info =  [*mean_info_turb, *std_info_turb, *skewness_info_turb, *kurtosis_info_turb]
        df.loc[row] = np.concatenate((simulation_info, ground_frame_info, turb_frame_info))
        row += 1
# saving the dataframe
df.to_csv(data_fn)


# TODO - add in plots that must be made while 