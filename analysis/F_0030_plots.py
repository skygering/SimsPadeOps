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
import os
import analysis_utils as au
import quick_metadata_plots as qmplt
import padeopsIO as pio
import matplotlib.pyplot as plt
import numpy as np
import streamtube
import pandas as pd
import seaborn as sns
import glob
from pathlib import Path
from UnifiedMomentumModel.Momentum import UnifiedMomentum, ThrustBasedUnified
import math
import streamtube

# %%
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams['figure.dpi'] = 100

# %%
Ct_prime = 2
f = 0.5
Av = 0.3
D = 1
rho = 1

# %%
sim_folder =  os.path.join(au.DATA_PATH, "F_0030_SU_Files/")
run_folder = au.get_run_folder(sim_folder, 0)
sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine", verbose=False)
log_file = glob.glob(f'*{0000}*.o[0-9]*', root_dir=run_folder)
logfile_path = os.path.join(run_folder, log_file[0])

# %%
power = sim.read_turb_power("all", turb=1)
uvel = sim.read_turb_uvel("all", turb=1)
log = pio.query_logfile(
    logfile_path,
    search_terms=["tilt", "uturb", "Time", "TIDX", "delta", "normalized turbine phase"],
    crop_equal=False
)
time = log["Time"]
mask = time > 100
time = time[mask]
uturb = log["uturb"][mask]
xturb = log["delta"][mask]
phase = log["normalized turbine phase"][mask]
power = power[mask]
uvel = uvel[mask]

df = pd.DataFrame({
    "phase": phase,
    "power": power,
    "uvel": uvel,
    "uturb": uturb,
    "xturb": xturb,
})
df["phase"] = df["phase"].round(4)
df["period"] = np.zeros(len(phase), dtype=int)
df.loc[1:, "period"] = np.cumsum(np.diff(phase) < 0)
max_period = np.max(df["period"])
df = df[(df["period"] > 0) & (df["period"] < max_period)]

# %%
df_stats = df.groupby("phase", as_index=False).agg({
    "uturb": ["mean", "std"],
    "xturb": ["mean", "std"],
    "uvel": ["mean", "std"],
    "power": ["mean", "std"],
})

# flatten column names
df_stats.columns = ["phase",
                    "uturb_mean", "uturb_std",
                    "xturb_mean", "xturb_std",
                    "uvel_mean", "uvel_std",
                    "power_mean", "power_std"]
df_stats["urel"] = 1 - df_stats["uturb_mean"]
# sort
df_stats = df_stats.sort_values("phase")

# %%
# mean lines
plt.plot(df_stats["phase"], df_stats["uturb_mean"], label="$u_t$")
plt.plot(df_stats["phase"], df_stats["xturb_mean"], label="$x_t$")
plt.plot(df_stats["phase"], df_stats["uvel_mean"], label="$u_d$")

# combined quantity
u_sum_mean = df_stats["uvel_mean"] + df_stats["uturb_mean"]
u_sum_std  = np.sqrt(df_stats["uvel_std"]**2 + df_stats["uturb_std"]**2)

plt.plot(df_stats["phase"], u_sum_mean, label="$u_d + u_t$")

u_rel = 1 - df_stats["uturb_mean"]
plt.plot(df_stats["phase"], u_rel, label="$u_{\infty_\\text{rel}}$")

plt.legend()

# %%
ut = df_stats["uturb_mean"]
ud = df_stats["uvel_mean"]
urel = df_stats["urel"]
ut = ut - ut.mean()
ud = ud - ud.mean()
urel = urel - urel.mean()

ut_fft = np.fft.fft(ut)
ud_fft = np.fft.fft(ud)
urel_fft = np.fft.fft(urel)

# first harmonic (k=1)
phase_ut = np.angle(ut_fft[1])
phase_ud = np.angle(ud_fft[1])
phase_urel = np.angle(urel_fft[1])

phase_offset = phase_ud - phase_urel
phase_offset = (phase_offset + np.pi) % (2*np.pi) - np.pi
phase_deg = np.degrees(phase_offset)
phase_deg

# %%
ud_max_idx = np.argmax(ud)
ud_max_phase = df_stats["phase"][ud_max_idx]
urel_max_idx = np.argmax(urel)
ured_max_phase = df_stats["phase"][urel_max_idx]
phase_offset = ud_max_phase - ured_max_phase
phase_offset * 360

# %%
df_stats["an"] = 1 - (df_stats["uvel_mean"] / (df_stats["urel"]))
df_stats["Ct"] = Ct_prime * df_stats["uvel_mean"]**2
df_stats["Cp"] = df_stats["power_mean"] / (0.5 * rho * math.pi * (D/2)**2)

# %%
model = UnifiedMomentum()
stationary_vals = model(Ct_prime, yaw = 0.0, tilt = 0.0)
df_stats["an_umm"] = stationary_vals.an
df_stats["an_diff"] = df_stats["an"] - df_stats["an_umm"]
df_stats["Ct_umm"] = stationary_vals.Ct * df_stats["urel"]**2
df_stats["Ct_diff"] = df_stats["Ct"] - df_stats["Ct_umm"]
df_stats["Cp_umm"] = stationary_vals.Cp * df_stats["urel"]**3
df_stats["Cp_diff"] = df_stats["Cp"] - df_stats["Cp_umm"]

# %%
plt.plot(df_stats["phase"], df_stats["Ct"], label="LES $C_T$")
plt.plot(df_stats["phase"], df_stats["Ct_umm"], label="UMM $C_T$")
plt.plot(df_stats["phase"], df_stats["Ct_diff"], label="Differece LES - UMM $C_T$")
plt.legend()

# %%
plt.plot(df_stats["phase"], df_stats["an"], label="LES $a_n$")
plt.plot(df_stats["phase"], df_stats["an_umm"], label="UMM $a_n$")
plt.plot(df_stats["phase"], df_stats["an_diff"], label="Differece LES - UMM $a_n$")
plt.legend()

# %%
# dphi = df_stats["phase"].diff().mean()
# dt = dphi / f

# def dynamic_inflow_step(
#     Ct, Uinf, Uref, R, dt,
#     u_act, u_str,
#     uturb,
#     glauert=True,
#     dynamic = False
# ):
#     """
#     One time-step update of Ferreira dynamic inflow model
#     adapted for your UMM workflow.
#     """

#     # --- length scales ---
#     len_act = 1.0 * R
#     len_str = 5.0 * R

#     # --- update reference velocity ---
#     if dynamic:
#         expf = np.exp(-dt * Uref / len_str)
#         Uref = Uref * expf + (Uinf - uturb) * (1 - expf)
#     else:
#         Uref = Uinf

#     # --- streamtube velocity ---
#     Ustr = Uref - 0.5 * (u_act + u_str)

#     # --- quasi-steady forcing ---
#     # (this is the key coupling to momentum theory)
#     Uqs = Ct * Uinf**2 / (4 * Ustr)

#     # --- Glauert correction (optional) ---
#     if glauert:
#         Induction_Glauert = 1 - np.sqrt(1.816) / 2
#         mask = (Ustr < Uref * (1 - Induction_Glauert)) & (Uqs > 0)
#         Uqs[mask] = (
#             -1.88254912
#             - 1.54029217 * np.sqrt(Uqs[mask])
#             + 4.08622347 * Uqs[mask]**0.25
#         )

#     # --- convection time scales ---
#     inv_tau_act_1 = (Uinf - 0.5 * u_act - uturb) / len_act
#     inv_tau_act_2 = (Uref - 0.5 * u_act) / len_act
#     inv_tau_str   = (Uref - 0.5 * u_str) / len_str

#     # --- update actuator induction ---
#     u_act_new = (
#         u_act * np.exp(-dt * inv_tau_act_1)
#         + Uqs * (1 - np.exp(-dt * inv_tau_act_2))
#     )

#     # --- update streamtube induction ---
#     u_str_new = (
#         u_str * np.exp(-dt * inv_tau_str)
#         + Uqs * (1 - np.exp(-dt * inv_tau_str))
#     )

#     return u_act_new, u_str_new, Uref

# %%
# # Example rotor parameters
# Uinf = 10.0         # Freestream wind [m/s]
# R = 50.0            # Rotor radius [m]
# Ct = 0.8            # Thrust coefficient

# # Initialize inductions
# a = Ct / (4 + Ct)   # axial induction factor
# u_act = np.array([a * Uinf])
# u_str = np.copy(u_act)        # starting guess
# Uref = Uinf          # initial streamtube reference velocity
# uturb = 0.0          # optional turbulent induction contribution

# k = 5
# dt = 0.1  # time step [s]
# time = np.arange(0, t_final, dt)

# # store history for analysis
# history = []

# for t in time:
#     # Step the dynamic inflow
#     Ct = 0.8 + 0.5 * np.cos(2 * np.pi * f * time)
#     u_act, u_str, Uref = dynamic_inflow_step(
#         Ct=Ct,
#         Uinf=Uinf,
#         Uref=Uref,
#         R=R,
#         dt=dt,
#         u_act=u_act,
#         u_str=u_str,
#         uturb=uturb,
#         glauert=True
#     )
    
#     # Compute streamtube velocity for forcing
#     Ustr = Uref - 0.5 * (u_act + u_str)
    
#     # (Optional) compute quasi-steady thrust or other rotor quantities
#     Uqs = Ct * Uinf**2 / (4 * Ustr)
    
#     # Save for post-processing
#     history.append(u_act/ Uref)
    
#     # Advance time
#     time += dt

# %%
# plt.plot(df_stats["phase"], df_stats["an"], label="LES $a_n$")
# plt.plot(df_stats["phase"], df_stats["an_umm"], label="UMM $a_n$")
# plt.plot(df_stats["phase"], df_stats["an_dyn"], label="UMM with Dynamic Inflow $a_n$")

# %%
f = 0.5
Av = 0.3
Ax = Av / (2 * f * np.pi)
xlim = [-2.5, 12]
ylim =  0
zlim = [-8.5, 8.5]
phases = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
turbX = Ax * np.sin(2 * np.pi * phases)
turbV = Av *  np.cos(2 * np.pi * phases)
plt.plot(phases, turbX, label = "Location")
plt.plot(phases, turbV, label = "Velocity")
plt.legend()

# %%
xlim = [-2.5, 12.5]
zlim = [-3, 3]
phases[0]

# %%
zlim

# %%
time_ds = sim.slice(budget_terms=["ubar", "pbar"], xlim = xlim, ylim = ylim, zlim = zlim)
npoints = len(time_ds["ubar"])
phase_ds = sim.slice(budget_terms=["ubar", "pbar"], xlim = xlim, ylim = ylim, zlim = zlim, phase = phases[5])
phase_max_ubar = np.max(np.abs(phase_ds["ubar"]))
# subtract time-averaged fields
ubar_wave = phase_ds["ubar"] - time_ds["ubar"]
wave_max_ubar = np.max(np.abs(ubar_wave))

phase_ds["ubar"].imshow(cmap = "bwr", vmin = 0, vmax = 2)


img2 = phase_ds["ubar"].imshow(cmap = "bwr", vmin = 0, vmax = 2)
ax = plt.gca()  # current axes
ax.tick_params(axis='both', which='major', labelsize=16)
cbar = img2.colorbar
cbar.ax.tick_params(labelsize=16)
ax.set_ylabel("$z/D$", fontsize=16)
ax.set_xlabel("$x/D$", fontsize=16)
cbar.set_label("$\\langle u \\rangle_{\phi = 0.5}/U_\infty$", fontsize=16)
# Save at high DPI
plt.gcf().savefig("ubar_wave_phi_05.png", dpi=600, bbox_inches="tight")

# %%
phase_max_pbar = np.max(np.abs(phase_ds["pbar"]))
# subtract time-averaged fields
pbar_wave = phase_ds["pbar"] - time_ds["pbar"]
wave_max_pbar = np.max(np.abs(pbar_wave))

phase_ds["pbar"].imshow(cmap = "bwr", vmin = -phase_max_pbar, vmax = phase_max_pbar)
pbar_wave.imshow(cmap = "bwr", vmin = -wave_max_pbar, vmax = wave_max_pbar)

# %%
time_ds = sim.slice(budget_terms=["ubar", "pbar"], xlim = xlim, ylim = 0, zlim = 0)
plt.plot(time_ds["ubar"], label = "Velocity")
plt.plot(time_ds["pbar"], label = "Pressure")
plt.legend()

# %%
for i in range(10):
    time_ds = sim.slice(budget_terms=["ubar", "pbar"], xlim = xlim, ylim = 0, zlim = 0)
    phase_ds = sim.slice(budget_terms=["ubar", "pbar"], xlim = xlim, ylim = 0, zlim = 0, phase = phases[i])
    # subtract time-averaged fields
    ubar_wave = phase_ds["ubar"] - time_ds["ubar"]
    pbar_wave = phase_ds["pbar"] - time_ds["pbar"]

    plt.plot(np.linspace(0, 17, npoints) - turbX[i], phase_ds["pbar"], label = phases[i])
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlim(4, 6)
plt.axvline(5, ymin=-2, ymax=2, c="k", linestyle="--")

# %%
for i in range(10):
    time_ds = sim.slice(budget_terms=["ubar", "pbar"], xlim = xlim, ylim = 0, zlim = 0)
    phase_ds = sim.slice(budget_terms=["ubar", "pbar"], xlim = xlim, ylim = 0, zlim = 0, phase = phases[i])
    # subtract time-averaged fields
    ubar_wave = phase_ds["ubar"] - time_ds["ubar"]
    pbar_wave = phase_ds["pbar"] - time_ds["pbar"]

    plt.plot(np.linspace(0, 17, npoints) - turbX[i], phase_ds["ubar"], label = phases[i])
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlim(4, 6)
plt.axvline(5, ymin=-2, ymax=2, c="k", linestyle="--")

# %%

# %%
phase = 4
xT = turbX[phase]
phase_slice = sim.slice(budget_terms=["ubar", "vbar", "wbar", "pbar"], xlim = [-2, 16], ylim = [-2, 2], zlim = [-2, 2], phase = phases[phase])
stream = streamtube.Streamtube(phase_slice["x"], phase_slice["y"], phase_slice["z"], phase_slice["ubar"], phase_slice["vbar"], phase_slice["wbar"])
traj = stream.compute_streamtube(return_trajectories=True)

# %%
from scipy.interpolate import RegularGridInterpolator

phase_bern_rem_upsteam = []
phase_bern_rem_downstream = []
phase_bern_rem_upsteam_std = []
phase_bern_rem_downstream_std = []
phase_bern_upsteam_far = []
phase_bern_downstream_far = []
phase_bern_before_disk = []
phase_bern_after_disk = []

for (i, phase_val) in enumerate(phases):
    xT = turbX[i]

    # --- slice data ---
    phase_slice = sim.slice(
        budget_terms=["ubar", "vbar", "wbar", "pbar"],
        xlim=[-5, 15], ylim=[-5, 5], zlim=[-5, 5],
        phase=phases[i]
    )

    # --- streamlines ---
    stream = streamtube.Streamtube(
        phase_slice["x"],
        phase_slice["y"],
        phase_slice["z"],
        phase_slice["ubar"],
        phase_slice["vbar"],
        phase_slice["wbar"]
    )
    traj = stream.compute_streamtube(return_trajectories=True)

    # --- interpolators ---
    xg, yg, zg = phase_slice["x"], phase_slice["y"], phase_slice["z"]

    interp_u = RegularGridInterpolator((xg, yg, zg), phase_slice["ubar"])
    interp_v = RegularGridInterpolator((xg, yg, zg), phase_slice["vbar"])
    interp_w = RegularGridInterpolator((xg, yg, zg), phase_slice["wbar"])
    interp_p = RegularGridInterpolator((xg, yg, zg), phase_slice["pbar"])

    rho = 1.0
    g = 0.0   # probably negligible in your case

    def bernoulli_at_points(pts):
        u = interp_u(pts)
        v = interp_v(pts)
        w = interp_w(pts)
        p = interp_p(pts)
        umag2 = u**2 + v**2 + w**2
        return p / rho + 0.5 * umag2 + g * pts[:, 2]

    # --- helper to get value at a specific x location along streamline ---
    def sample_at_x(x_target, x, y, z):
        idx = np.argmin(np.abs(x - x_target))
        return np.array([x[idx], y[idx], z[idx]])

    # --- storage ---
    bernoulli_upstream_far = []
    bernoulli_before_disk = []

    bernoulli_after_disk = []
    bernoulli_downstream_far = []

    bernoulli_rem_upstream = []
    bernoulli_rem_downstream = []

    # --- loop over streamlines ---
    for streamline in traj:

        x, y, z = streamline

        pts = np.vstack([x, y, z]).T

        # --- sample locations ---
        pt_upstream_far   = sample_at_x(xT - 2.0, x, y, z)
        pt_before_disk    = sample_at_x(xT - 0.1, x, y, z)
        pt_after_disk     = sample_at_x(xT + 0.1, x, y, z)
        pt_downstream_far = sample_at_x(xT + 2.0, x, y, z)

        # --- compute Bernoulli values ---
        B_upstream_far   = bernoulli_at_points(pt_upstream_far[None, :])[0]
        B_before_disk    = bernoulli_at_points(pt_before_disk[None, :])[0]
        B_after_disk     = bernoulli_at_points(pt_after_disk[None, :])[0]
        B_downstream_far = bernoulli_at_points(pt_downstream_far[None, :])[0]

        bernoulli_upstream_far.append(B_upstream_far)
        bernoulli_before_disk.append(B_before_disk)
        bernoulli_after_disk.append(B_after_disk)
        bernoulli_downstream_far.append(B_downstream_far)

        # --- residuals (your "unsteady contribution") ---
        rem_upstream = B_before_disk - B_upstream_far
        rem_downstream = B_downstream_far - B_after_disk

        bernoulli_rem_upstream.append(rem_upstream)
        bernoulli_rem_downstream.append(rem_downstream)

    mean_rem_upsteam = np.mean(bernoulli_rem_upstream)
    std_rem_upsteam = np.std(bernoulli_rem_upstream)
    mean_rem_downstream = np.mean(bernoulli_rem_downstream)
    std_rem_downstream = np.std(bernoulli_rem_downstream)

    phase_bern_rem_upsteam.append(mean_rem_upsteam)
    phase_bern_rem_downstream.append(mean_rem_downstream)
    phase_bern_rem_upsteam_std.append(std_rem_upsteam)
    phase_bern_rem_downstream_std.append(std_rem_downstream)

    mean_upsteam_far = np.mean(bernoulli_upstream_far)
    mean_before_disk = np.mean(bernoulli_before_disk)
    mean_after_disk = np.mean(bernoulli_after_disk)
    mean_downstream_far = np.mean(bernoulli_downstream_far)

    phase_bern_upsteam_far.append(mean_upsteam_far)
    phase_bern_before_disk.append(mean_before_disk)
    phase_bern_after_disk.append(mean_after_disk)
    phase_bern_downstream_far.append(mean_downstream_far)

# %%
phase_bern_rem_upsteam_std = np.array(phase_bern_rem_upsteam_std)

# %%
phase_bern_rem_downstream_std = np.array(phase_bern_rem_downstream_std)

# %%
plt.plot(phases, phase_bern_rem_upsteam, label="upstream remainder")
plt.fill_between(
    phases,
    phase_bern_rem_upsteam - phase_bern_rem_upsteam_std,
    phase_bern_rem_upsteam + phase_bern_rem_upsteam_std,
    alpha=0.3
)

plt.plot(phases, phase_bern_rem_downstream, label="downstream remainder")
plt.fill_between(
    phases,
    phase_bern_rem_downstream - phase_bern_rem_downstream_std,
    phase_bern_rem_downstream + phase_bern_rem_downstream_std,
    alpha=0.3
)

plt.plot(phases, phase_bern_upsteam_far, label="upstream far")
plt.plot(phases, phase_bern_before_disk, label="before disk")

plt.plot(phases, phase_bern_downstream_far, label="downstream far")
plt.plot(phases, phase_bern_after_disk, label="after disk")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
plt.plot(phases, phase_bern_rem_upsteam, label = "upsteam remainder")
plt.plot(phases, phase_bern_rem_downstream, label = "downstream remainder")

plt.plot(phases, phase_bern_upsteam_far, label = "upsteam far")
plt.plot(phases, phase_bern_before_disk, label = "before disk")

plt.plot(phases, phase_bern_downstream_far, label = "downstream far")
plt.plot(phases, phase_bern_after_disk, label = "after disk")
plt.legend()

# %%
# sort indices by phase
sort_idx = np.argsort(phases)

# sorted x-axis (turbine velocity)
turbV_sorted = np.array(turbV)[sort_idx]

# sort all your data consistently
bern_rem_up_sorted = np.array(phase_bern_rem_upsteam)[sort_idx]
bern_rem_down_sorted = np.array(phase_bern_rem_downstream)[sort_idx]

bern_up_far_sorted = np.array(phase_bern_upsteam_far)[sort_idx]
bern_before_sorted = np.array(phase_bern_before_disk)[sort_idx]

bern_down_far_sorted = np.array(phase_bern_downstream_far)[sort_idx]
bern_after_sorted = np.array(phase_bern_after_disk)[sort_idx]

# %%
plt.figure()

plt.plot(turbV_sorted, bern_rem_up_sorted, label="upstream remainder")
plt.plot(turbV_sorted, bern_rem_down_sorted, label="downstream remainder")

plt.plot(turbV_sorted, bern_up_far_sorted, label="upstream far")
plt.plot(turbV_sorted, bern_before_sorted, label="before disk")

plt.plot(turbV_sorted, bern_down_far_sorted, label="downstream far")
plt.plot(turbV_sorted, bern_after_sorted, label="after disk")

plt.xlabel("Turbine Velocity")
plt.ylabel("Bernoulli Terms")
# plt.legend()
plt.grid()

# %%
rho = 1.0

phase_E_up = []
phase_E_down = []
phase_delta_E = []
phase_power = []
phase_residual = []

for (i, phase_val) in enumerate(phases):

    xT = turbX[i]

    # --- slice full domain ---
    phase_slice = sim.slice(
        budget_terms=["ubar", "vbar", "wbar", "pbar"],
        xlim=[-5, 15], ylim=[-5, 5], zlim=[-5, 5],
        phase=phase_val
    )

    x = phase_slice["x"]
    y = phase_slice["y"]
    z = phase_slice["z"]

    u = phase_slice["ubar"]
    v = phase_slice["vbar"]
    w = phase_slice["wbar"]
    p = phase_slice["pbar"]

    # --- find nearest planes ---
    x_up = xT - 0.5
    x_down = xT + 0.5

    x_np = x.values
    i_up = np.argmin(np.abs(x_np - x_up))
    i_down = np.argmin(np.abs(x_np - x_down))

    # --- extract planes ---
    u_up = u[i_up, :, :]
    v_up = v[i_up, :, :]
    w_up = w[i_up, :, :]
    p_up = p[i_up, :, :]

    u_down = u[i_down, :, :]
    v_down = v[i_down, :, :]
    w_down = w[i_down, :, :]
    p_down = p[i_down, :, :]

    # --- compute energy density ---
    umag2_up = u_up**2 + v_up**2 + w_up**2
    umag2_down = u_down**2 + v_down**2 + w_down**2

    E_up_density = p_up / rho + 0.5 * umag2_up
    E_down_density = p_down / rho + 0.5 * umag2_down

    # --- energy flux (u ⋅ n = u since normal is x) ---
    flux_up = E_up_density * u_up
    flux_down = E_down_density * u_down

    # --- grid spacing ---
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dA = dy * dz

    # --- integrate ---
    E_up = np.sum(flux_up) * dA
    E_down = np.sum(flux_down) * dA

    delta_E = E_up - E_down

    # =========================
    # --- turbine power estimate ---
    # =========================

    # planes just before/after disk
    i_before = np.argmin(np.abs(x_np - (xT - 0.05)))
    i_after  = np.argmin(np.abs(x_np - (xT + 0.05)))

    p_before = p[i_before, :, :]
    p_after  = p[i_after, :, :]

    u_disk = u[i_before, :, :]  # velocity at disk

    # pressure jump
    dp = p_before - p_after

    # power density = Δp * u
    power_density = dp * u_disk

    P = np.sum(power_density) * dA

    # --- residual ---
    residual = delta_E - P

    # --- store ---
    phase_E_up.append(E_up)
    phase_E_down.append(E_down)
    phase_delta_E.append(delta_E)
    phase_power.append(P)
    phase_residual.append(residual)

# %%
plt.figure()
plt.plot(phases, phase_delta_E, label="Δ Energy Flux")
plt.plot(phases, phase_power, label="Turbine Power")
plt.plot(phases, phase_residual, label="Residual")
plt.legend()
plt.xlabel("Phase")
plt.ylabel("Energy")
plt.grid()
