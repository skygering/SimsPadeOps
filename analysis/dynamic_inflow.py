# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: simspadeops-Wg-7Zt3Y-py3.11
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
from UnifiedMomentumModel.Momentum import UnifiedMomentum, ThrustBasedUnified
import seaborn as sns
from matplotlib.lines import Line2D


# %%
# import lib.quick_metadata_plots as mplts
# import lib.analysis_utils as au

# %%
def num(x):
    return pd.to_numeric(x, errors="coerce")  # bad parses -> NaN


# %%
# data_path = Path(au.DATA_PATH)

# --- F_0029 ---
# sim_reg_f_folder = os.path.join(au.DATA_PATH, "F_0029_SU_Files")
# rows_29, fields_29 = mplts.get_sim_varied_params(sim_reg_f_folder)
# (ids_29, cT_29, surge_amp_29, surge_freq_29, nx_29, ny_29, nz_29,
#     Lx_29, Ly_29, Lz_29, dt_29, tstop_29, n_hrs_29, filterWidth_29, useCorrection_29
# ) = zip(*rows_29)

# %%
# df29 = pd.DataFrame({
#     "id": ids_29,
#     "cT": num(cT_29),
#     "surge_amplitude": num(surge_amp_29),
#     "surge_freq": num(surge_freq_29),
#     "nx": num(nx_29), "ny": num(ny_29), "nz": num(nz_29),
#     "Lx": num(Lx_29), "Ly": num(Ly_29), "Lz": num(Lz_29),
#     "dt": num(dt_29),
#     "filterWidth": num(filterWidth_29),
#     # leave non-numeric as-is:
#     "useCorrection": useCorrection_29,
# })

# df29 = df29[df29["dt"] == 0.005]
# # keep only Av = 0.3
# df29 = df29[np.isclose(df29["surge_amplitude"], 0.1)]
# # keep only CT = CT_val
# CT_val = 1.33
# df29 = df29[np.isclose(df29["cT"], CT_val)]

# %%
# def load_runs_to_timeseries(df_runs, sim_folder, spinup_time):
#     dfs = []
#     for r in df_runs.itertuples(index=False):
#         try:
#             id_str = r.id
#             run_folder = os.path.join(sim_folder, f"Sim_{id_str}")

#             data = au.load_simulation_timeseries(
#                 run_folder,
#                 id_str,
#                 r.surge_freq,
#                 spinup_time=spinup_time
#             )

#             meta = {
#                 "id": int(r.id),
#                 "CT_prime": float(r.cT),  # or r.CT_prime if that is your column name
#                 "Surge_Amplitude": float(r.surge_amplitude),
#                 "Pitch_Amplitude": 0.0,
#                 "Frequency": float(r.surge_freq),
#                 "nx": float(r.nx),
#                 "ny": float(r.ny),
#                 "nz": float(r.nz),
#                 "Lx": float(r.Lx),
#                 "Ly": float(r.Ly),
#                 "Lz": float(r.Lz),
#                 "dt": float(r.dt),
#                 "filterWidth": float(r.filterWidth),
#                 "useCorrection": r.useCorrection,
#             }

#             df_temp = au.build_sim_dataframe(data, meta)
#             dfs.append(df_temp)
#         except Exception as e:
#             print(f"Error occurred in {id_str}: {e}")
#     return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# %%
# ts29 = load_runs_to_timeseries(df29, sim_reg_f_folder, 75)
# ts29 = ts29.rename(columns={"UDisk": "LES_UDisk", "Power": "LES_Power", "DeltaX": "XTurb"})
# ts29.keys()

# %%
UInf = 1.0
D = 1.0
rho = 1.0
A = math.pi * (D/2)**2
Pnorm = 0.5 * rho * A * UInf**3

# ts29["URel"] = UInf - ts29["UTurb"]

# ts29["LES_an"] =  1 - (ts29["LES_UDisk"] / ts29["URel"])
# ts29["LES_CT"] =  ts29["CT_prime"] * ts29["LES_UDisk"]**2 / (UInf)**2
# ts29["LES_CP"] = ts29["LES_Power"] / Pnorm

# model = UnifiedMomentum()
# CT_prime = CT_val
# umm_sol = model(CT_prime, yaw=0.0, tilt=0.0)

# def get_umm_udisk(row):
#     return (1 - umm_sol.an) * row.URel
# def get_umm_ct(row):
#     return umm_sol.Ct * row.URel**2
# def get_umm_cp(row):
#     return umm_sol.Cp * row.URel**3

# ts29["UMM_UDisk"] = ts29.apply(get_umm_udisk, axis=1)
# ts29["UMM_an"] = 1 - (ts29["UMM_UDisk"] / ts29["URel"])
# ts29["UMM_CT"] = ts29.apply(get_umm_ct, axis=1)
# ts29["UMM_CP"] = ts29.apply(get_umm_cp, axis=1)

# %%
# needed_cols = [
#     'id', 'CT_prime', 'Surge_Amplitude', 'Frequency', 'Time', 'TIDX', 'XTurb', 'UTurb', 'Phase', 'Period', 'URel', 
#     'LES_UDisk', 'LES_an', 'LES_CT', 'LES_CP', 'UMM_UDisk', 'UMM_an', 'UMM_CT', 'UMM_CP',
# ]

# %%
# test_vals = ts29[ts29["Frequency"] == 0.5]
# test_vals = test_vals[needed_cols]
# test_vals

# %%
test_vals = pd.read_csv("/Users/sky/Downloads/dynamic_inflow_test_vals.csv")

# %% [markdown]
# ## Ferreira

# %%
L_act = 0.5 * D
L_str = 2.5 * D


# %%
def get_Ustr(u_str, u_act):
    return UInf - (u_str + u_act) / 2

def get_u_qs(CT, u_str, u_act):
    Ustr = get_Ustr(u_str, u_act)
    u_qs = CT * UInf**2 / (4 * Ustr)                    # Eq. (10)

    Induction_Glauert = 1 - np.sqrt(1.816) / 2
    if Ustr < UInf * (1 - Induction_Glauert) and u_qs > 0:   # appendix criterion
        u_qs = (-1.88254912
                - 1.54029219 * u_qs**0.5
                + 4.08622347 * u_qs**0.25)              # Eq. (12)
    return u_qs

def get_ferreira_taus(u_str, u_act):
    tau_str = L_str / (UInf - 0.5 * u_str)
    tau_act2 = L_act / (UInf - 0.5 * u_act)
    return tau_str, tau_act2


# %%
def get_ferreira(a0, v_act, ct, time, dt):
    # a0 is a dimensionless induction factor; states are induction velocities
    u_act = a0 * UInf
    u_str = a0 * UInf
    an_dyn = np.zeros_like(time, dtype=float)

    for i in range(len(time)):
        v = v_act[i]
        tau_str, tau_act2 = get_ferreira_taus(u_str, u_act)
        u_qs = get_u_qs(ct[i], u_str, u_act)

        # Eq. (20): exp(-dt/tau_act2) * exp(dt * v / L_act), no tau_v needed
        decay_act = np.exp(-dt / tau_act2) * np.exp(dt * v / L_act)
        u_act_new = u_act * decay_act + u_qs * (1 - np.exp(-dt / tau_act2))
        u_str_new = u_str * np.exp(-dt / tau_str) + u_qs * (1 - np.exp(-dt / tau_str))

        an_dyn[i] = u_act_new / UInf
        u_act, u_str = u_act_new, u_str_new

    return an_dyn


# %%
time = np.array(test_vals["Time"])
an_umm = np.array(test_vals["UMM_an"])
an_les = np.array(test_vals["LES_an"])
ct = np.array(test_vals["LES_CT"])
# umm_ct = np.array(test_vals["UMM_CT"])
x_act = np.array(test_vals["XTurb"])
v_act = np.array(test_vals["UTurb"])
dt = time[1] - time[0]

# %%
an_ferreira = get_ferreira(an_umm[0], v_act, ct, time, dt)

# %%
fig, ax1 = plt.subplots()

# Plot on left axis
ax1.plot(time, x_act, label="x", color='tab:red')
# ax1.plot(time, v_act, label="x", color='tab:orange')
ax1.plot(time, an_les, label = "LES an", color="tab:orange")
ax1.plot(time, an_umm, label = "UMM an", color="tab:orange", linestyle = "--")
ax1.plot(time, an_ferreira, label = "Ferreira", color="tab:green")
ax1.set_ylim(-0.1, 0.7)
# ax1.set_xlim(85, 86)
ax1.set_ylabel('x, v, an')  # Label for left axis
ax1.set_xlabel('time')

# Create second y-axis and plot ct
ax2 = ax1.twinx()
ax2.plot(time, ct, label="ct", color='tab:blue')
ax2.set_ylim(-0.1, 1.5)
ax2.set_ylabel('ct')  # Label for right axis

ax1.grid(True, alpha=0.3)  # Add grid to left axis

# # Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.show()

# %% [markdown]
# ## Oye

# %%
b = 0.6
R = D / 2


# %%
def get_oye_taus(an, V0, rj=0):
    a_star = min(an, 0.5) # cap on constants
    tau1 = 1.1 / (1 - 1.3 * a_star) * R / V0
    tau2 = (0.39 - 0.26 * (rj / R)**2) * tau1
    return tau1, tau2



# %%
b = 0.6
R = D / 2

def get_oye_taus(an, V0, rj=0):
    a_star = min(an, 0.5) # cap on constants
    tau1 = 1.1 / (1 - 1.3 * a_star) * R / V0
    tau2 = (0.39 - 0.26 * (rj / R)**2) * tau1
    return tau1, tau2


def get_oye(an_qs_arr, time, dt, an_for_tau = None):
    an_qs_arr = np.asarray(an_qs_arr, dtype=float)
    v_qs_arr = an_qs_arr * UInf
    time = np.asarray(time, dtype=float)

    v_int_arr = np.zeros_like(v_qs_arr)
    v_dyn_arr = np.zeros_like(v_qs_arr)

    # Initialize states at the initial quasi-steady induction
    v_int_arr[0] = v_qs_arr[0]
    v_dyn_arr[0] = v_qs_arr[0]

    # Derivative of quasi-steady induction
    dv_qs_dt = np.gradient(v_qs_arr, time)

    tau1_min = np.inf
    tau2_min = np.inf

    if an_for_tau is None:
        an_for_tau = an_qs_arr
    else:
        an_for_tau = np.asarray(an_for_tau, dtype=float)

    for i in range(1, len(time)):
        v_qs = v_qs_arr[i]
        tau1, tau2 = get_oye_taus(an_for_tau[i], UInf)

        tau1_min = min(tau1_min, tau1)
        tau2_min = min(tau2_min, tau2)

        # Intermediate induction
        v_int_arr[i] = v_int_arr[i-1] + dt * (
            (v_qs - v_int_arr[i-1]) / tau1
            + b * dv_qs_dt[i]
        )

        # Dynamic induction
        v_dyn_arr[i] = v_dyn_arr[i-1] + dt * (
            (v_int_arr[i] - v_dyn_arr[i-1]) / tau2
        )

    print(f"Minimum tau1: {tau1_min:.6g}")
    print(f"Minimum tau2: {tau2_min:.6g}")

    return v_dyn_arr


# %%
an_oye = get_oye(an_umm, time, dt)

# %%
fig, ax1 = plt.subplots()

# Plot on left axis
ax1.plot(time, x_act, label="x", color='tab:red')
# ax1.plot(time, v_act, label="x", color='tab:orange')
ax1.plot(time, an_les, label = "LES an", color="tab:orange")
ax1.plot(time, an_umm, label = "UMM an", color="tab:orange", linestyle = "--")
ax1.plot(time, an_oye, label = "Oye", color="tab:green")
ax1.set_ylim(-0.1, 0.7)
# ax1.set_xlim(85, 86)
ax1.set_ylabel('x, v, an')  # Label for left axis
ax1.set_xlabel('time')

# Create second y-axis and plot ct
ax2 = ax1.twinx()
ax2.plot(time, ct, label="ct", color='tab:blue')
ax2.set_ylim(-0.1, 1.5)
ax2.set_ylabel('ct')  # Label for right axis

ax1.grid(True, alpha=0.3)  # Add grid to left axis

# # Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.show()


# %% [markdown]
# ## Test on Ferreria Paper Examples

# %%
def induction_from_CT_UMM(CT):  # in ground frame
    model = ThrustBasedUnified()
    umm_sol = model(CT, yaw=0.0, tilt=0.0)
    return umm_sol.an


# %%
# def induction_from_CT_emperical(CT):   # in ground frame
#     CT = np.asarray(CT, dtype=float)
#     return  0.2460 * CT + 0.0586 * CT**2 + 0.0883 * CT**3

def induction_from_CT_emperical(CT):
    CT = np.clip(np.asarray(CT, dtype=float), 0.0, None)

    a_low = 0.5 * (1 - np.sqrt(np.maximum(1 - CT, 0.0)))

    a_high = 0.143 + np.sqrt(
        np.maximum(0.0203 - 0.6427 * (0.889 - CT), 0.0)
    )

    return np.where(CT <= 0.96, a_low, a_high)


# %%
CT_arr = np.linspace(0, 1.7, num = 200)
an_umm_arr = induction_from_CT_UMM(CT_arr)
an_emperical_arr = induction_from_CT_emperical(CT_arr)

plt.plot(CT_arr, an_umm_arr, label = "UMM")
plt.plot(CT_arr, an_emperical_arr, label = "emperical")
plt.axvline(x = 0.4, linestyle = "--", c = "k")
plt.legend()
plt.xlabel("CT")
plt.ylabel("an")


# %%
def ferreira_paper_inputs(A, k, CT0, dCT, qs_method = "UMM", FOR = "absolute"):
    T = 1 / (k / (2 * np.pi))
    nperiods = 5
    time = np.arange(-0.25 * T, nperiods * T, 0.01)
    dt = time[1] - time[0]
    x_act = A * np.sin(k * time / D)
    v_act = A * k * np.cos(k * time / D)
    ct = CT0 - dCT * np.cos(k * time / D)

    if qs_method == "UMM":
        an_qs = induction_from_CT_UMM(ct)
    elif qs_method == "emperical":
        an_qs = induction_from_CT_emperical(ct)

    if FOR == "rotor":
        udisk = (1 - an_qs) * (UInf - v_act) # turbine frame
        an_qs = 1 - udisk / (UInf) # ground frame

    return an_qs, dt, T, time, x_act, v_act, ct


# %%
def make_ferreira_plot(A, k, CT0, dCT, y_lim1, y_lim2):
    an_qs_umm_test, dt_test, T_test, time_test, x_act_test, v_act_test, ct_test = ferreira_paper_inputs(A, k, CT0, dCT, qs_method = "UMM")
    an_ferreira_umm_test = get_ferreira(an_qs_umm_test[0], v_act_test, ct_test, time_test, dt_test)
    an_oye_test_umm = get_oye(an_qs_umm_test, time_test, dt_test)

    an_qs_emp_test, dt_test, T_test, time_test, x_act_test, v_act_test, ct_test = ferreira_paper_inputs(A, k, CT0, dCT, qs_method = "emperical")
    an_ferreira_emp_test = get_ferreira(an_qs_emp_test[0], v_act_test, ct_test, time_test, dt_test)
    an_oye_test_emp = get_oye(an_qs_emp_test, time_test, dt_test, an_for_tau = an_qs_umm_test)

    fig, ax1 = plt.subplots()

    # Plot on left axis
    ax1.plot(time_test, x_act_test, label="x", color='tab:red', linestyle = "dotted", lw = 2)
    # ax1.plot(time_test, v_act_test, label="v", color='tab:red')
    ax1.plot(time_test, an_qs_umm_test, label = "QS UMM", color = "k", lw = 2)
    ax1.plot(time_test, an_ferreira_umm_test, label = "Ferreira UMM", color="tab:green", lw = 2)
    ax1.plot(time_test, an_oye_test_umm, label = "Oye UMM", color="tab:purple", lw = 2)

    ax1.plot(time_test, an_qs_emp_test, label = "QS emperical", color = "k", linestyle = "--", lw = 2)
    ax1.plot(time_test, an_ferreira_emp_test, label = "Ferreira emperical", color="tab:green", linestyle = "--", lw = 2)
    ax1.plot(time_test, an_oye_test_emp, label = "Oye emperical", color="tab:purple", linestyle = "--", lw = 2)
    ax1.set_ylim(y_lim1[0], y_lim1[1])
    ax1.set_xlim(4*T_test, 5*T_test)
    ax1.set_ylabel('x, v, an')  # Label for left axis
    ax1.set_xlabel('time')

    # Create second y-axis and plot ct
    ax2 = ax1.twinx()
    ax2.plot(time_test, ct_test, label="ct", color='tab:blue', linestyle = "dotted", lw = 2)
    ax2.set_ylim(y_lim2[0], y_lim2[1])
    ax2.set_ylabel('ct')  # Label for right axis

    ax1.minorticks_on()
    ax1.grid(True, which="major", alpha=0.5)
    ax1.grid(True, which="minor", alpha=0.2)

    # # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
            loc='upper left', bbox_to_anchor=(1.15, 1))

    plt.show()


# %%
A, k, CT0, dCT = 0.0, 1, 0.5, 0.5
y_lim1 = (-0.1, 0.7)
y_lim2 = (-0.1, 1.10)
make_ferreira_plot(A, k, CT0, dCT, y_lim1, y_lim2)

# %%
A, k, CT0, dCT = 0.1, 5, 0.5, 0.5
y_lim1 = (-0.1, 0.7)
y_lim2 = (-0.1, 1.10)
make_ferreira_plot(A, k, CT0, dCT, y_lim1, y_lim2)

# %%
A, k, CT0, dCT = 0.0, 5, 0.8, 0.5
y_lim1 = (-0.1, 0.7)
y_lim2 = (0.2, 1.4)
make_ferreira_plot(A, k, CT0, dCT, y_lim1, y_lim2)

# %%
A, k, CT0, dCT = 0.1, 5, 0.8, 0.0
y_lim1 = (-0.1, 0.7)
y_lim2 = (0.6, 1.0)
make_ferreira_plot(A, k, CT0, dCT, y_lim1, y_lim2)

# %%
A, k, CT0, dCT = 0.1, 5, 0.8, 0.5
y_lim1 = (-0.1, 1.0)
y_lim2 = (0.2, 1.4)
make_ferreira_plot(A, k, CT0, dCT, y_lim1, y_lim2)

# %%
A, k, CT0, dCT = 0.1, 15, 0.8, 1.5
y_lim1 = (-0.1, 0.7)
y_lim2 = (-0.8, 2.4)
make_ferreira_plot(A, k, CT0, dCT, y_lim1, y_lim2)
# CT goes above 2 -> QS UMM doesn't work!

# %%
