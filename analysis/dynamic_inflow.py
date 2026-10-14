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
from UnifiedMomentumModel.Momentum import UnifiedMomentum, ThrustBasedUnified
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

# --- F_0029 ---
sim_reg_f_folder = os.path.join(au.DATA_PATH, "F_0029_SU_Files")
rows_29, fields_29 = mplts.get_sim_varied_params(sim_reg_f_folder)
(ids_29, cT_29, surge_amp_29, surge_freq_29, nx_29, ny_29, nz_29,
    Lx_29, Ly_29, Lz_29, dt_29, tstop_29, n_hrs_29, filterWidth_29, useCorrection_29
) = zip(*rows_29)

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
    # leave non-numeric as-is:
    "useCorrection": useCorrection_29,
})

df29 = df29[df29["dt"] == 0.005]
# keep only Av = 0.3
df29 = df29[np.isclose(df29["surge_amplitude"], 0.1)]
# keep only CT = CT_val
CT_val = 1.33
df29 = df29[np.isclose(df29["cT"], CT_val)]


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
ts29 = load_runs_to_timeseries(df29, sim_reg_f_folder, 75)
ts29 = ts29.rename(columns={"UDisk": "LES_UDisk", "Power": "LES_Power", "DeltaX": "XTurb"})
ts29.keys()

# %%
UInf = 1.0
D = 1.0
rho = 1.0
A = math.pi * (D/2)**2
Pnorm = 0.5 * rho * A * UInf**3

ts29["URel"] = UInf - ts29["UTurb"]

ts29["LES_an"] =  1 - (ts29["LES_UDisk"] / ts29["URel"])
ts29["LES_CT"] =  ts29["CT_prime"] * ts29["LES_UDisk"]**2 / (UInf)**2
ts29["LES_CP"] = ts29["LES_Power"] / Pnorm

model = UnifiedMomentum()
CT_prime = CT_val
umm_sol = model(CT_prime, yaw=0.0, tilt=0.0)

def get_umm_udisk(row):
    return (1 - umm_sol.an) * row.URel
def get_umm_ct(row):
    return umm_sol.Ct * row.URel**2
def get_umm_cp(row):
    return umm_sol.Cp * row.URel**3

ts29["UMM_UDisk"] = ts29.apply(get_umm_udisk, axis=1)
ts29["UMM_an"] = 1 - (ts29["UMM_UDisk"] / ts29["URel"])
ts29["UMM_CT"] = ts29.apply(get_umm_ct, axis=1)
ts29["UMM_CP"] = ts29.apply(get_umm_cp, axis=1)

# %%
needed_cols = [
    'id', 'CT_prime', 'Surge_Amplitude', 'Frequency', 'Time', 'TIDX', 'XTurb', 'UTurb', 'Phase', 'Period', 'URel', 
    'LES_UDisk', 'LES_an', 'LES_CT', 'LES_CP', 'UMM_UDisk', 'UMM_an', 'UMM_CT', 'UMM_CP',
]

# %%
test_vals = ts29[ts29["Frequency"] == 0.5]
test_vals = test_vals[needed_cols]
test_vals

# %% [markdown]
# ## Ferreira

# %%
L_act = 0.5 * D
L_str = 2.5 * D


# %%
def get_Ustr(u_str, u_act):
    return UInf - (u_str + u_act) / 2

def get_u_qs(CT, u_str, u_act):
    return CT * UInf**2 / (4 * get_Ustr(u_str, u_act)) # needs to be re-derived for CT'

def get_ferreira_taus(u_str, u_act, v_act):
    tau_str = L_str / (UInf - 0.5 * u_str)
    tau_act1 = L_act / (UInf - 0.5 * u_act - v_act)
    tau_act2 = L_act / (UInf - 0.5 * u_act)
    return tau_str, tau_act1, tau_act2


# %%
def get_ferreira(u0, v_act, ct, time, dt):
    u_act, u_str = u0, u0
    an_dyn = np.zeros_like(time)

    for i, t in enumerate(time):
        v = v_act[i] 
        tau_str, tau_act1, tau_act2 = get_ferreira_taus(u_str, u_act, v)
        u_qs = get_u_qs(ct[i], u_str, u_act)
        u_act_new = u_act*np.exp(-dt / tau_act1) + u_qs * (1 - np.exp(-dt / tau_act2))
        u_str_new = u_str*np.exp(-dt / tau_str) + u_qs * (1 - np.exp(-dt / tau_str))
        an_dyn[i] = u_act_new / (UInf)
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
def get_oye_taus(an, V0, rj = 0):
    tau1 = 1.1 / (1 - (1.3 * an)) * R / V0
    tau2 = (0.39 - (0.26 * (rj / R)**2)) * tau1
    return tau1, tau2


# %%
def get_oye(v_qs_arr, v_act_arr, time, dt):
    v_int_arr = np.zeros_like(v_qs_arr)
    v_dyn_arr = np.zeros_like(v_qs_arr)

    # initialize states (start at quasi-steady)
    v_int_arr[0] = v_qs_arr[0]
    v_dyn_arr[0] = v_qs_arr[0]

    # helper derivative of v_qs
    dv_qs_dt = np.zeros_like(v_qs_arr)
    dv_qs_dt[1:] = (v_qs_arr[1:] - v_qs_arr[:-1]) / dt

    for i, t in enumerate(time):
        if i == 0: continue
        v_qs = v_qs_arr[i]
        V0 = UInf - v_act_arr[i]
        tau1, tau2 = get_oye_taus(v_qs, V0)

        # Explicit Euler:
        # dv_int/dt = (v_qs + b*tau1*dv_qs_dt - v_int)/tau1
        v_int_arr[i] = v_int_arr[i-1] + dt * (
            (v_qs + b * tau1 * dv_qs_dt[i] - v_int_arr[i-1]) / tau1
        )

        # dv_dyn/dt = (v_int - v_dyn)/tau2
        v_dyn_arr[i] = v_dyn_arr[i-1] + dt * (
            (v_int_arr[i] - v_dyn_arr[i-1]) / tau2
        )
        
    return v_dyn_arr


# %%
an_oye = get_oye(an_umm, v_act, time, dt)

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
def ferreira_paper_inputs(A, k, CT0, dCT):
    T = 1 / (k / (2 * np.pi))
    nperiods = 5
    time = np.linspace(-0.5 * T, nperiods * T, nperiods * 50)
    dt = time[1] - time[0]
    x_act = A * np.sin(k * time / D)
    v_act = A * k * np.cos(k * time / D)
    ct = CT0 - dCT * np.cos(k * time / D)

    model = ThrustBasedUnified()
    umm_sol = model(ct, yaw=0.0, tilt=0.0)
    umm_an = np.array(umm_sol.an)
    udisk = (1 - umm_an) * (UInf - v_act)
    an_qs = 1 - udisk / (UInf) #- v_act)

    return an_qs, dt, T, time, x_act, v_act, ct


# %%
A, k, CT0, dCT = 0.0, 5, 0.5, 0.5
an_qs_test, dt_test, T_test, time_test, x_act_test, v_act_test, ct_test = ferreira_paper_inputs(A, k, CT0, dCT)

# %%
an_ferreira_test = get_ferreira(an_qs_test[0], v_act_test, ct_test, time_test, dt_test)

# %%
an_oye_test = get_oye(an_qs_test, v_act_test, time_test, dt_test)

# %%
fig, ax1 = plt.subplots()

# Plot on left axis
ax1.plot(time_test, x_act_test, label="x", color='tab:red')
ax1.plot(time_test, an_qs_test, label = "QS", color = "k", linestyle = "--")
ax1.plot(time_test, an_ferreira_test, label = "Ferreira", color="tab:green")
ax1.plot(time_test, an_oye_test, label = "Oye", color="tab:purple")
ax1.set_ylim(-0.1, 0.9)
ax1.set_xlim(4*T_test, 5*T_test)
ax1.set_ylabel('x, v, an')  # Label for left axis
ax1.set_xlabel('time')

# Create second y-axis and plot ct
ax2 = ax1.twinx()
ax2.plot(time_test, ct_test, label="ct", color='tab:blue')
ax2.set_ylim(0.2, 1.4)
ax2.set_ylabel('ct')  # Label for right axis

ax1.grid(True, alpha=0.3)  # Add grid to left axis

# # Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.show()

# %%
