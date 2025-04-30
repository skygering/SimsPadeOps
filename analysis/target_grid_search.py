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

data_path = Path(au.DATA_PATH)
# all simulation folders used
sim_4_X_folder = os.path.join(au.DATA_PATH, "F_0004_X_Files")
sim_4_SU_PI_folder = os.path.join(au.DATA_PATH, "F_0004_SU_PI_Files")
sim_5_all_folder = os.path.join(au.DATA_PATH, "F_0005_X_SU_PI_Files")
sim_6_all_folder = os.path.join(au.DATA_PATH, "F_0006_X_SU_PI_Files")
sim_8_all_folder = os.path.join(au.DATA_PATH, "F_0008_X_SU_PI_Files")
sim_9_all_folder = os.path.join(au.DATA_PATH, "F_0009_X_SU_PI_Files")
sim_10_all_folder = os.path.join(au.DATA_PATH, "F_0010_X_SU_PI_Files")
sim_10H_all_folder = os.path.join(au.DATA_PATH, "F_0010_SU_PI_H_Files")
sim_11_all_folder = os.path.join(au.DATA_PATH, "F_0011_SU_PI_Files")
sim_12_all_folder = os.path.join(au.DATA_PATH, "F_0012_SU_PI_Files")
data_fn = os.path.join(sim_6_all_folder, 'collected_runs.csv')

# # go through data and collect
df = pd.DataFrame(columns=['marker', 'nx', 'ny', 'filter', 'filterFactor', 'useCorrection', 'CT_prime', "turbulence",
                           'surge_freq', 'surge_amplitude', 'pitch_amplitude',
                           'mean_CT','mean_an','mean_Cp',
                           'variance_CT', 'variance_an', 'variance_Cp',
                           'std_CT', 'std_an', 'std_Cp',
                           'skewness_CT', 'skewness_an', 'skewness_Cp',
                           'kurtosis_CT', 'kurtosis_an', 'kurtosis_Cp'])
folders = (sim_4_X_folder, sim_4_SU_PI_folder, sim_5_all_folder, sim_6_all_folder, sim_8_all_folder, sim_9_all_folder, sim_10_all_folder, sim_10H_all_folder, sim_11_all_folder, sim_12_all_folder)
row = 0
for (k, folder) in enumerate(folders):
    rows, fields = mplts.get_sim_varied_params(folder)
    if k == 0:
        print("A")
        ids, cT, tstop, nx, ny, nz, dt, filterWidth, useCorrection = zip(*rows)
        surge_freq = [0] * len(ids)
        surge_amplitude = surge_freq
        pitch_amplitude = surge_amplitude
    elif k == 1:
        print("B")
        ids, cT, nx, ny, nz, dt, filterWidth, useCorrection, surge_amplitude, pitch_amplitude = zip(*rows)
        surge_freq = [1] * len(ids)
    elif k == 2:
        print("C")
        ids, dt, cT, surge_freq, surge_amplitude, pitch_amplitude, nx, ny, nz, filterWidth = zip(*rows)
        useCorrection = [False] * len(ids)
    elif k == 3:
        print("D")
        ids, dt, cT, surge_freq, surge_amplitude, pitch_amplitude, nx, ny, nz, filterWidth = zip(*rows)
        useCorrection = [False] * len(ids)
    elif k == 4:
        print("E")
        ids, dt, nx, ny, nz, filterWidth, cT, surge_freq, surge_amplitude, pitch_amplitude = zip(*rows)
        useCorrection = [True] * len(ids)
    elif k == 5:
        print("F")
        ids, dt, cT, nx, ny, nz, filterWidth, useCorrection, hit_inputdir, TI_fact, surge_freq, surge_amplitude, pitch_amplitude = zip(*rows)
    elif k == 6:
        ids, dt, cT, surge_freq, surge_amplitude, pitch_amplitude, filterWidth, useCorrection = zip(*rows)
        nx, ny, nz = [256] * len(ids), [128] * len(ids), [128] * len(ids)
    elif k == 7:
        ids, dt, surge_freq, surge_amplitude, pitch_amplitude, filterWidth = zip(*rows)
        nx, ny, nz = [320] * len(ids), [128] * len(ids), [128] * len(ids)
        useCorrection = [False] * len(ids)
        cT = [1.0] * len(ids)
    elif k == 8:
        ids,cT,dt,surge_freq,surge_amplitude,pitch_amplitude,filterWidth = zip(*rows)
        nx, ny, nz = [256] * len(ids), [128] * len(ids), [128] * len(ids)
        useCorrection = [True] * len(ids)
    elif k == 9:
        ids,surge_freq,surge_amplitude,pitch_amplitude,dt,filterWidth = zip(*rows)
        nx, ny, nz = [256] * len(ids), [128] * len(ids), [128] * len(ids)
        useCorrection = [True] * len(ids)
        cT = [1.0] * len(ids)

    # get data from runs
    for (i, id_str) in enumerate(ids):
        run_folder = os.path.join(folder, "Sim_" + id_str)
        sim = pio.BudgetIO(run_folder, padeops = True, runid = 0)
        dt_i = dt[i]
        trans_tau = int(math.ceil(50 / float(dt_i)) + 1)
        try:
            # get all of the needed values
            power = sim.read_turb_power("all", turb=1)[trans_tau:]
            uvel = sim.read_turb_uvel("all", turb = 1)[trans_tau:]
        except: 
            continue
        else:
            print("HERE")
            h = ((25 / float(nx[i]))**2 + 2 * (10 / float(ny[i]))**2)**(1/2)
            filter_width = float(filterWidth[i])
            # if useCorrection[i] and filter_width > 0.1:
            #     continue
            Cp_vals = [au.power_to_Cp(p) for p in power]
            an_vals = [au.vel_to_a(u) for u in uvel]
            cT_prime_val = float(cT[i])
            cT_vals = [cT_prime_val * (1 - an)**2 for an in an_vals]
            # save sim info
            if float(surge_freq[i]) == 0:
                marker = "o"
            elif float(surge_amplitude[i]) != 0:
                marker = "s"
            else:
                marker = "^"
            filter_factor = round(filter_width / h, 3)

            turbulence = True if (k == 5 or k == 7) else False

            Cp_stats = describe(Cp_vals)
            an_stats = describe(an_vals)
            cT_stats = describe(cT_vals)
            mean_info = [cT_stats.mean, an_stats.mean, Cp_stats.mean]
            variance_info = [cT_stats.variance, an_stats.variance, Cp_stats.variance]
            std_info = [std(cT_vals), std(an_vals), std(Cp_vals)]
            skewness_info = [cT_stats.skewness, an_stats.skewness, Cp_stats.skewness]
            kurtosis_info = [cT_stats.kurtosis, an_stats.kurtosis, Cp_stats.kurtosis]
            # ['marker', 'nx', 'ny', 'filter', 'filterFactor', 'useCorrection', 'CT_prime', "turbulence",'mean_CT','mean_an','mean_Cp','variance_CT', 'variance_an', 'variance_Cp', 'std_CT', 'std_an', 'std_Cp','skewness_CT', 'skewness_an', 'skewness_Cp','kurtosis_CT', 'kurtosis_an', 'kurtosis_Cp']
            df.loc[row] = [marker, float(nx[i]), float(ny[i]), filter_width, filter_factor, useCorrection[i], cT_prime_val, turbulence, surge_freq[i], surge_amplitude[i], pitch_amplitude[i]] + mean_info + variance_info + std_info + skewness_info + kurtosis_info
            row += 1
# saving the dataframe
df.to_csv(data_fn)


# def four_plot(df, title, an_key, CT_key, Cp_key, save_fn, add_classical = False, extra_plots = False, ax_label_type = ''):
#     if not extra_plots:
#         fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(14, 8))
#         SMALL_SIZE = 10
#         MEDIUM_SIZE = 14
#         BIGGER_SIZE = 18
#     else: 
#         fig, axes = plt.subplots(3, 2, figsize=(20, 20))
#         ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = axes
#         ax4.set(xlabel='$a_n$ Mean', ylabel = "$C_p$ " + ax_label_type)
#         ax5.set(xlabel='$C_p$ Mean', ylabel = "$a_n$ " + ax_label_type)
#         SMALL_SIZE = 20
#         MEDIUM_SIZE = 20
#         BIGGER_SIZE = 24
#         for ax_row in axes:
#             for ax in ax_row:
#                 ax.tick_params(axis='both', which='major', labelsize=16)
#                 ax.xaxis.label.set_size(16)
#                 ax.yaxis.label.set_size(16)


#     plt.rc('font', size=SMALL_SIZE)
#     plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#     fig.suptitle(title)
#     ax0.set(ylabel='$C_T$ ' + ax_label_type)
#     ax2.set(ylabel='$C_p$ ' + ax_label_type, xlabel='$a_n $' + ax_label_type)
#     ax3.set(xlabel='$C_T\'$ ' + ax_label_type)

#     unique_factors = df['filterFactor'].unique()
#     unique_factors.sort()
#     unique_nx = df['nx'].unique()
#     unique_nx.sort()

#     cmap = plt.get_cmap('jet')
#     num_colors = len(unique_factors)
#     colors = [cmap(value) for value in np.linspace(0, 1, num_colors)]
#     colors.reverse()

#     sizes = [50, 100, 200, 400]

#     for index, row in df.iterrows():
#         i = np.where(unique_nx == row["nx"])[0][0]
#         marker_size = sizes[i]
#         j = np.where(unique_factors == row["filterFactor"])[0][0]
#         marker_color = colors[j]
#         if row["useCorrection"]:
#             edgecolor = 'r'
#             edgewidth = 2
#             alpha = 0.5
#         else:
#             edgecolor = 'face'
#             edgewidth = 0
#             alpha = 0.5
#         # Columns in row: ['marker', 'nx', 'ny', 'filter', 'filterFactor','CT_prime','mean_CT','mean_an','mean_Cp']
#         # plot CT vs an - size based off of resolution, color based off of filter, marker based off of turbine movement
#         ax0.scatter(row[an_key], row[CT_key], marker = row["marker"], s = marker_size, color = marker_color, edgecolors = edgecolor, linewidth = edgewidth, alpha=alpha)
#         # plot CT vs CT' - size based off of resolution, color based off of filter, marker based off of turbine movement
#         ax1.scatter(row["CT_prime"], row[CT_key], marker = row["marker"], s = marker_size, color = marker_color, edgecolors = edgecolor, linewidth = edgewidth, alpha=alpha)
#         # plot Cp vs an - size based off of resolution, color based off of filter, marker based off of turbine movement
#         ax2.scatter(row[an_key], row[Cp_key], marker = row["marker"], s = marker_size, color = marker_color, edgecolors = edgecolor, linewidth = edgewidth, alpha=alpha)
#         # plot Cp vs CT' - size based off of resolution, color based off of filter, marker based off of turbine movement
#         ax3.scatter(row["CT_prime"], row[Cp_key], marker = row["marker"], s = marker_size, color = marker_color, edgecolors = edgecolor, linewidth = edgewidth, alpha=alpha)

#         if add_classical:
#             # classical momentum values for statinary turbine
#             ctp_vals = np.linspace(0, 12, 50)
#             classical_an = [au.analytical_a(ctp) for ctp in ctp_vals]
#             classical_cp = [au.a_to_Cp(a) for a in classical_an]
#             classical_ct = [ctp * (1 - classical_an[i])**2 for (i, ctp) in enumerate(ctp_vals)]
#             ax0.plot(classical_an, classical_ct, c = 'k')
#             ax1.plot(ctp_vals, classical_ct, c = 'k')
#             ax2.plot(classical_an, classical_cp, c = 'k')
#             ax3.plot(ctp_vals, classical_cp, c = 'k')

#         if extra_plots:
#             ax4.scatter(row["mean_an"], row[Cp_key], marker = row["marker"], s = marker_size, color = marker_color, edgecolors = edgecolor, alpha=0.5)
#             ax5.scatter(row["mean_Cp"], row[an_key],  marker = row["marker"], s = marker_size, color = marker_color, edgecolors = edgecolor, alpha=0.5)

#         # shape legend
#         shape_elements = [Line2D([0], [0], marker='.', color='w', label='Stationary', markerfacecolor='tab:gray', markersize=12),
#                           Line2D([0], [0], marker='s', color='w', label='Surging', markerfacecolor='tab:gray', markersize=8),
#                           Line2D([0], [0], marker='^', color='w', label='Pitching', markerfacecolor='tab:gray', markersize=8)]
#         ax0.legend(handles=shape_elements, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", title = "Turbine Movement", ncols = 3)
#         # color legend
#         color_elements = [Line2D([0], [0], marker='o', color='w', label=f'{unique_factors[i]}', markerfacecolor=c, markersize=8) for (i, c) in enumerate(colors)]
#         ax1.legend(handles=color_elements, bbox_to_anchor=(1.04, 1), loc="upper left", title = "Filter Factors")
#         # size legend
#         size_elements = [Line2D([0], [0], marker='o', color='w', label=f'{unique_nx[i]}', markerfacecolor='tab:gray', markersize=(s)**(1/2)) for (i, s) in enumerate(sizes)]
#         ax3.legend(handles=size_elements, bbox_to_anchor=(1.04, 1), loc="upper left", title = "Nx")
        


#     # Create the figure
#     plt.tight_layout()
#     # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", handles=legend_elements)
#     plt.savefig(os.path.join(sim_6_all_folder, save_fn))
#     plt.close()

# df = pd.read_csv(data_fn)
# # only includes ONE final run with correction on to see the changes
# df = df[(df['useCorrection'] == False) | ((df['useCorrection'] == True) & (df['filterFactor'] == 1.5) & (df['nx'] == 256))]
# df = df[df['filterFactor'] != 0.3]
# stationary_df = df[df['marker'] == "o"]
# surging_df = df[df['marker'] == "s"]
# pitching_df = df[df['marker'] == "^"]



# four_plot(df, "Means for Simulations ", "mean_an", "mean_CT", "mean_Cp", 'mean_target_grid_search.png', add_classical = True, ax_label_type = "Mean")
# four_plot(df, "Skewness for Simulations", "skewness_an", "skewness_CT", "skewness_Cp", 'skewness_target_grid_search.png', extra_plots = True, ax_label_type = "Skew")
# four_plot(df, "Standard Deviation for Simulations", "std_an", "std_CT", "std_Cp", 'std_target_grid_search.png', extra_plots = True, ax_label_type = "STD")
# four_plot(df, "Kurtosis for Simulations", "kurtosis_an", "kurtosis_CT", "kurtosis_Cp", 'kurtosis_target_grid_search.png', extra_plots = True, ax_label_type = "Kurtosis")

# four_plot(stationary_df, "Stationary Turbine - Means for Simulations", "mean_an", "mean_CT", "mean_Cp", 'stationary_mean_target_grid_search.png', add_classical = True, ax_label_type = "Mean")
# # four_plot(stationary_df, "Stationary Turbine - Skewness for Simulations", "skewness_an", "skewness_CT", "skewness_Cp", 'stationary_skewness_target_grid_search.png', extra_plots = True, ax_label_type = "Skew")
# four_plot(stationary_df, "Stationary Turbine - Standard Deviation for Simulations", "std_an", "std_CT", "std_Cp", 'stationary_std_target_grid_search.png', extra_plots = True, ax_label_type = "STD")
# # four_plot(stationary_df, "Stationary Turbine - Kurtosis for Simulations", "kurtosis_an", "kurtosis_CT", "kurtosis_Cp", 'stationary_kurtosis_target_grid_search.png', extra_plots = True, ax_label_type = "Kurtosis")

# four_plot(surging_df, "Surging Turbine - Means for Simulations", "mean_an", "mean_CT", "mean_Cp", 'surging_mean_target_grid_search.png', add_classical = True, ax_label_type = "Mean")
# four_plot(surging_df, "Surging Turbine - Skewness for Simulations", "skewness_an", "skewness_CT", "skewness_Cp", 'surging_skewness_target_grid_search.png', extra_plots = True, ax_label_type = "Skew")
# four_plot(surging_df, "Surging Turbine - Standard Deviation for Simulations", "std_an", "std_CT", "std_Cp", 'surging_std_target_grid_search.png', extra_plots = True, ax_label_type = "STD")
# four_plot(surging_df, "Surging Turbine - Kurtosis for Simulations", "kurtosis_an", "kurtosis_CT", "kurtosis_Cp", 'surging_kurtosis_target_grid_search.png', extra_plots = True, ax_label_type = "Kurtosis")

# four_plot(pitching_df, "Pitching Turbine - Means for Simulations", "mean_an", "mean_CT", "mean_Cp", 'pitching_mean_target_grid_search.png', add_classical = True, ax_label_type = "Mean")
# four_plot(pitching_df, "Pitching Turbine - Skewness for Simulations", "skewness_an", "skewness_CT", "skewness_Cp", 'pitching_skewness_target_grid_search.png', extra_plots = True, ax_label_type = "Skew")
# four_plot(pitching_df, "Pitching Turbine - Standard Deviation for Simulations", "std_an", "std_CT", "std_Cp", 'pitching_std_target_grid_search.png', extra_plots = True, ax_label_type = "STD")
# four_plot(pitching_df, "Pitching Turbine - Kurtosis for Simulations", "kurtosis_an", "kurtosis_CT", "kurtosis_Cp", 'pitching_kurtosis_target_grid_search.png', extra_plots = True, ax_label_type = "Kurtosis")