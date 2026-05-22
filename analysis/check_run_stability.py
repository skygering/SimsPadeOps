# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: simspadeops-YbNxeWqo-py3.11
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import analysis.lib.analysis_utils as au
import matplotlib.pyplot as plt
import os
from pathlib import Path
import analysis.lib.quick_metadata_plots as mplts
from scipy.stats import describe
import padeopsIO as pio
import math

# %%
dpi = 300

# %%
data_path = Path(au.DATA_PATH)

# %%
sim_11_all_folder = os.path.join(au.DATA_PATH, "F_0011_SU_PI_Files")

# %%
sim11_run8_plt_folder = mplts.plot_instantaneous_field(sim_11_all_folder, 8, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim11_run8_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim11_run8_plt_folder = mplts.plot_instantaneous_field(sim_11_all_folder, 8, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim11_run8_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim11_run9_plt_folder = mplts.plot_instantaneous_field(sim_11_all_folder, 9, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim11_run9_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim11_run9_plt_folder = mplts.plot_instantaneous_field(sim_11_all_folder, 9, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim11_run9_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim11_run10_plt_folder = mplts.plot_instantaneous_field(sim_11_all_folder, 10, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim11_run10_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim11_run10_plt_folder = mplts.plot_instantaneous_field(sim_11_all_folder, 10, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim11_run10_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim_12_all_folder = os.path.join(au.DATA_PATH, "F_0012_SU_PI_Files")

# %%
sim12_run16_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 16, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run16_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run17_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 17, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run17_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run18_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 18, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run18_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run19_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 19, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run19_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run20_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 20, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run20_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run21_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 21, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run21_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run41_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 41, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run41_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run42_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 42, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run42_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run42_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 42, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim12_run42_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim12_run43_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 43, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run43_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run44_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 44, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run44_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run45_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 45, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run45_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run45_plt_folder = mplts.plot_instantaneous_field(sim_12_all_folder, 45, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim12_run45_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim_12_cont_folder = os.path.join(au.DATA_PATH, "F_0012_SU_PI_Cont_Files")

# %%
sim12_run15_plt_folder = mplts.plot_instantaneous_field(sim_12_cont_folder, 15, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run15_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run15_plt_folder = mplts.plot_instantaneous_field(sim_12_cont_folder, 15, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim12_run15_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim12_run31_plt_folder = mplts.plot_instantaneous_field(sim_12_cont_folder, 31, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run31_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim12_run31_plt_folder = mplts.plot_instantaneous_field(sim_12_cont_folder, 31, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim12_run31_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim_12_stab_folder = os.path.join(au.DATA_PATH, "F_0012_SU_PI_Stability_Files")

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_12_stab_folder, [17], labels = ["high pitch"], zoom = [0, 250])

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_12_stab_folder, [18], labels = ["high pitch"], zoom = [150, 250])

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_12_stab_folder, [19], labels = ["high surge"], zoom = [100, 250])

# %%
from scipy.fft import fft, fftfreq
import padeopsIO as pio
import math

# %%
run_folder = au.get_run_folder(sim_12_stab_folder, 19)
sim = pio.BudgetIO(run_folder, padeops = True, runid = 0)
dt = sim.input_nml["input"]["dt"]
trans_tau = int(math.ceil(100 / dt) + 1)
power = sim.read_turb_uvel("all", turb=1)[trans_tau:]
mean_power = np.mean(power)
non_mean_power = power - mean_power

yf = fft(non_mean_power)
N = len(yf)
xf = fftfreq(N, dt)[:N//2]
fig, ax = plt.subplots(dpi = 100)
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax.grid()

# %%
run_folder = au.get_run_folder(sim_12_stab_folder, 14)
sim = pio.BudgetIO(run_folder, padeops = True, runid = 0)
dt = sim.input_nml["input"]["dt"]
trans_tau = int(math.ceil(100 / dt) + 1)
power = sim.read_turb_uvel("all", turb=1)[trans_tau:]
mean_power = np.mean(power)
non_mean_power = power - mean_power

yf = fft(non_mean_power)
N = len(yf)
xf = fftfreq(N, dt)[:N//2]
fig, ax = plt.subplots(dpi = 100)
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax.grid()

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_12_stab_folder, [17], labels = ["lower surge"], zoom = [0, 250])

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_12_stab_folder, [28], labels = ["lower surge"], zoom = [0, 250])

# %%
sim12_run39_plt_folder = mplts.plot_instantaneous_field(sim_12_stab_folder, 39, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim12_run39_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim12_run39_plt_folder = mplts.plot_instantaneous_field(sim_12_stab_folder, 39, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim12_run39_plt_folder, fps = 10, video_name = "u_stability.mp4")

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_12_stab_folder, [36], labels = ["lower surge"], zoom = [0, 250])

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_12_stab_folder, [37], labels = ["lower surge"], zoom = [0, 200])

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_12_stab_folder, [38], labels = ["lower surge"], zoom = [0, 250])

# %%
sim12_run7_plt_folder = mplts.plot_instantaneous_field(sim_12_stab_folder, 7, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim12_run7_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim_08_folder = os.path.join(au.DATA_PATH, "F_0008_X_SU_PI_Files")

# %%
sim8_run4_plt_folder = mplts.plot_instantaneous_field(sim_08_folder, 4, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim8_run4_plt_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
import padeopsIO as pio

# %%
unmoving_folder = os.path.join(au.DATA_PATH, "B_0001_Files")
big_yaw_folder = os.path.join(unmoving_folder, "Sim_0044")
big_yaw_folder

# %%
bio = pio.budgetIO.BudgetIO(big_yaw_folder, padeops=True, runid = 0)

# %%
bio.write_data(big_yaw_folder, ["ubar", "vbar", "wbar"], "yaw_30_velocities")

# %%
sim_15_folder = os.path.join(au.DATA_PATH, "F_0015_X_SU_PI_Files")

# %%
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, dpi = 100)
fig.suptitle("$C_P$ Over Time", fontsize = 16)
ax0.set_ylim(0.5, 0.65)
ax0.set_title("Stationary Turbine")
ax0.set_ylabel("$C_P$")
ax1.set_ylim(0.5, 0.65)
ax1.set_title("Pitching Turbine ($f = 0.25$, $A = 5^\circ$)")
ax1.set_ylabel("$C_P$")
ax2.set_ylim(-0.1, 2.5)
ax2.set_title("Surging Turbine ($f = 0.25$, $A = 0.25$)")
ax2.set_ylabel("$C_P$")
ax2.set_xlabel("Timesteps")
zoom = [0, 250]
mplts.plot_requested_turb_power(ax0, sim_15_folder, [0], labels = ["lower surge"], zoom = zoom)
mplts.plot_requested_turb_power(ax1, sim_15_folder, [2], labels = ["lower surge"], zoom = zoom)
mplts.plot_requested_turb_power(ax2, sim_15_folder, [1], labels = ["lower surge"], zoom = zoom)
ax0.vlines(100, 0, 3, color = "grey", linestyles= "dashed")
ax1.vlines(100, 0, 3, color = "grey", linestyles= "dashed")
ax2.vlines(100, 0, 3, color = "grey", linestyles= "dashed")
fig.subplots_adjust(hspace=0.5)

# %%
sim_15_folder_run1 = mplts.plot_instantaneous_field(sim_15_folder, 1, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim_15_folder_run1, fps = 10, video_name = "u_stability.mp4")

# %%
run_folder = au.get_run_folder(sim_15_folder, 0)
sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")
ds = sim.slice(budget_terms=['ubar'], xlim = [-5, 20], ylim = [-5, 5], zlim=0)
ds

# %%
ds['ubar'].imshow()

# %%
sim_16_folder = os.path.join(au.DATA_PATH, "F_0016_SU_PI_Files")

# %%
sim_16_124_folder = mplts.plot_instantaneous_field(sim_16_folder, 124, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim_16_124_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
sim_16_124_folder = mplts.plot_instantaneous_field(sim_16_folder, 124, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim_16_124_folder, fps = 10, video_name = "u_stability.mp4")

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_16_folder, [123, 73], labels = ["2.0", "1.66"], zoom = [240, 250])

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_16_folder, [123, 73], labels = ["2.0", "1.66"], zoom = [240, 250])

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_16_folder, [123, 73], labels = ["2.0", "1.66"], zoom = [240, 250])

# %%
sim_16_118_folder = mplts.plot_instantaneous_field(sim_16_folder, 118, tidx = "all", field = "u", dpi = dpi)
mplts.film_instantaneous_field(sim_16_118_folder, fps = 10, video_name = "u_stability.mp4")

# %%
sim_16_18_folder = mplts.plot_instantaneous_field(sim_16_folder, 18, tidx = "all", field = "u", xlim = 0, dpi = dpi, set_plot_lims = True)
mplts.film_instantaneous_field(sim_16_18_folder, fps = 10, video_name = "u_rotor_stability.mp4")

# %%
run_folder_high_res = au.get_run_folder(sim_16_folder, 123)
run_folder_low_res = run_folder_high_res + "_low_res"
sim_low_res = pio.BudgetIO(run_folder_low_res, padeops = True, runid = 0)
dt_low_res = sim_low_res.input_nml["input"]["dt"]
trans_tau_low_res = int(math.ceil(100 / dt_low_res) + 1)
power_low_res = sim_low_res.read_turb_uvel("all", turb=1)[trans_tau_low_res:]
Cp_vals_low_res = [au.power_to_Cp(p) for p in power_low_res]
Cp_stats_low_res = describe(Cp_vals_low_res)
Cp_stats_low_res

# %%
sim_high_res = pio.BudgetIO(run_folder_high_res, padeops = True, runid = 0)
dt_high_res = sim_high_res.input_nml["input"]["dt"]
trans_tau_high_res = int(math.ceil(100 / dt_high_res) + 1)
power_high_res = sim_high_res.read_turb_uvel("all", turb=1)[trans_tau_high_res:]
Cp_vals_high_res = [au.power_to_Cp(p) for p in power_high_res]
Cp_stats_high_res = describe(Cp_vals_high_res)
Cp_stats_high_res

# %%
run_folder_high_res = au.get_run_folder(sim_16_folder, 73)
run_folder_low_res = run_folder_high_res + "_low_res"
sim_low_res = pio.BudgetIO(run_folder_low_res, padeops = True, runid = 0)
dt_low_res = sim_low_res.input_nml["input"]["dt"]
trans_tau_low_res = int(math.ceil(100 / dt_low_res) + 1)
power_low_res = sim_low_res.read_turb_uvel("all", turb=1)[trans_tau_low_res:]
Cp_vals_low_res = [au.power_to_Cp(p) for p in power_low_res]
Cp_stats_low_res = describe(Cp_vals_low_res)
Cp_stats_low_res

# %%
sim_high_res = pio.BudgetIO(run_folder_high_res, padeops = True, runid = 0)
dt_high_res = sim_high_res.input_nml["input"]["dt"]
trans_tau_high_res = int(math.ceil(100 / dt_high_res) + 1)
power_high_res = sim_high_res.read_turb_uvel("all", turb=1)[trans_tau_high_res:]
Cp_vals_high_res = [au.power_to_Cp(p) for p in power_high_res]
Cp_stats_high_res = describe(Cp_vals_high_res)
Cp_stats_high_res

# %%
fig, ax = plt.subplots(dpi = 100)
mplts.plot_requested_turb_power(ax, sim_16_folder, [123, 73], labels = ["2.0", "1.66"], zoom = [240, 250])

# %%
import seaborn as sns
blues = sns.color_palette("Blues_r", as_cmap=True)

# %%
sim_16_12_folder = mplts.plot_instantaneous_field(sim_16_folder, 12, xlim = [-2, 15], ylim = [-2, 2], tidx = 5000, field = "u", dpi = 300, colormap = blues)

# %%
sim_9H_all_folder = os.path.join(au.DATA_PATH, "F_0009_X_SU_PI_Files")

# %%
sim_16_18_folder = mplts.plot_instantaneous_field(sim_9H_all_folder, 0, tidx = "all", field = "u", xlim = 4, ylim = [-5, 5], zlim = [-5, 5], dpi = dpi, set_plot_lims = True)
