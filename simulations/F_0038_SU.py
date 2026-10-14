from pathlib import Path
import itertools

import jinja_sim_utils as ju
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 4-run matrix for one-case F_0038:
# (1) sinusoidal surge + constant CT'
# (2) timeseries surge + constant CT'
# (3) sinusoidal surge + timeseries CT'
# (4) timeseries surge + timeseries CT'
#
# Uses one OpenFAST-derived seed/case from:
#   - F_0038_metadata.csv
#   - F_0038_timeseries.csv
#
# HWindSpeed,WaveHs,WaveTp = 10.0,5.0,8.0
# ============================================================

sim_template = ju.TEMPLATE_PATH.joinpath("sim_template.jinja")
turb_template = ju.TEMPLATE_PATH.joinpath("turb_template.jinja")
run_template = ju.TEMPLATE_PATH.joinpath("run_template.jinja")
default_inputs = ju.DEFAULTS_PATH.joinpath("floating_defaults.json")

# ---- user files ----
meta_file = Path("/scratch/10264/sgering/SimsPadeOps/simulations/F_0038_metadata.csv")
ts_file = Path("/scratch/10264/sgering/SimsPadeOps/simulations/F_0038_timeseries.csv")

# base/domain reference
domain_file = Path("/scratch/10264/sgering/SimsPadeOps/simulations/F_0029_SU.csv")

# ---- timing controls ----
spinup_time = 75.0
blend_time = 5.0
dt_signal = 0.005      # signal grid for input-file generation
# NOTE: solver dt remains from domain_file (e.g., 0.00025)

# ============================================================
# Read metadata / timeseries
# ============================================================
meta = pd.read_csv(meta_file)
if len(meta) != 1:
    raise ValueError(f"Expected exactly one row in metadata, found {len(meta)}.")
m = meta.iloc[0]

df_raw = pd.read_csv(ts_file)

# If file has multiple case_names, keep only metadata case_name
if "case_name" in df_raw.columns:
    df_raw = df_raw[df_raw["case_name"] == m["case_name"]].copy()

if df_raw.empty:
    raise ValueError("No matching timeseries rows found for metadata case_name.")

# Sort by time, drop duplicate times
df_raw = df_raw.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

# ---- pull metadata values ----
case_name = str(m["case_name"])
v_A = float(m["v_A"])                    # for simple periodic mode: surge_amplitude
v_f = float(m["v_f"])                    # for simple periodic mode: surge_freq
CT_prime_mean = float(m["CT_prime_mean"])

# ============================================================
# Domain / simulation values
# ============================================================
df_dom = pd.read_csv(domain_file)
nx, ny, nz = np.unique(df_dom["nx"]), np.unique(df_dom["ny"]), np.unique(df_dom["nz"])
Lx, Ly, Lz = np.unique(df_dom["Lx"]), np.unique(df_dom["Ly"]), np.unique(df_dom["Lz"])
dt_solver = float(np.unique(df_dom["dt"])[0])

h = np.sqrt((Lx[0] / nx[0])**2 + (Ly[0] / ny[0])**2 + (Lz[0] / nz[0])**2)
filterWidth = 1.5 * h
useCorrection = True

# ============================================================
# Build uniform signal-time grid for provided timeseries
# ============================================================
t_raw = df_raw["time"].to_numpy()
t0, t1 = float(t_raw.min()), float(t_raw.max())

t_data = np.arange(t0, t1 + 0.5 * dt_signal, dt_signal)

# linear interpolation to dt_signal
disp_data = np.interp(t_data, t_raw, df_raw["displacement"].to_numpy())
vel_data = np.interp(t_data, t_raw, df_raw["velocity"].to_numpy())
ctp_data = np.interp(t_data, t_raw, df_raw["CT_prime"].to_numpy())

# Demean displacement and velocity (as discussed)
disp_data = disp_data - np.nanmean(disp_data)
vel_data = vel_data - np.nanmean(vel_data)

data_duration = t_data[-1] - t_data[0]

# total simulation time: spinup + measured duration
tstop = spinup_time + data_duration
n_hrs = int(np.ceil(tstop / 80 * 3) + 4)

# ============================================================
# Align measured signals to local post-spinup clock
# ============================================================
# measured clock starts at 0
t_data0 = t_data - t_data[0]

disp_data0 = disp_data.copy()
vel_data0 = vel_data.copy()
ctp_data0 = ctp_data.copy()

# full simulation clock
t_full = np.arange(0.0, tstop + 0.5 * dt_signal, dt_signal)

# sinusoidal reference (spinup source + blended source)
omega = 2.0 * np.pi * v_f
surge_sin = (v_A / omega) * np.sin(omega * t_full) if omega != 0 else np.zeros_like(t_full)
ut_sin = v_A * np.cos(omega * t_full)

# local time relative to end of spinup
tau = t_full - spinup_time

# evaluate measured data only on valid local range
valid = (tau >= 0.0) & (tau <= t_data0[-1])

disp_ts_full = np.full_like(t_full, np.nan, dtype=float)
vel_ts_full = np.full_like(t_full, np.nan, dtype=float)
ctp_ts_full = np.full_like(t_full, np.nan, dtype=float)

disp_ts_full[valid] = np.interp(tau[valid], t_data0, disp_data0)
vel_ts_full[valid] = np.interp(tau[valid], t_data0, vel_data0)
ctp_ts_full[valid] = np.interp(tau[valid], t_data0, ctp_data0)

# Edge handling: hold first/last value outside measured interval
before = tau < 0.0
after = tau > t_data0[-1]
if np.any(before):
    disp_ts_full[before] = disp_data0[0]
    vel_ts_full[before] = vel_data0[0]
    ctp_ts_full[before] = ctp_data0[0]
if np.any(after):
    disp_ts_full[after] = disp_data0[-1]
    vel_ts_full[after] = vel_data0[-1]
    ctp_ts_full[after] = ctp_data0[-1]

# ============================================================
# Smooth blend weight (cosine ramp)
# ============================================================
w = np.zeros_like(t_full)
i_mid = (t_full >= spinup_time) & (t_full <= spinup_time + blend_time)
i_hi = t_full > (spinup_time + blend_time)

s = (t_full[i_mid] - spinup_time) / blend_time
w[i_mid] = 0.5 * (1.0 - np.cos(np.pi * s))
w[i_hi] = 1.0

# ============================================================
# Blended signals
# ============================================================
surge_blend = (1.0 - w) * surge_sin + w * disp_ts_full
ut_blend = (1.0 - w) * ut_sin + w * vel_ts_full
ctp_blend = (1.0 - w) * CT_prime_mean + w * ctp_ts_full
pitch_zeros = np.zeros_like(t_full)

# ============================================================
# Output files
# ============================================================
curr_script_name = Path(__file__).with_suffix("").name
output_dir = Path(ju.DATA_PATH) / f"{curr_script_name}_Files"
output_dir.mkdir(parents=True, exist_ok=True)

timeseries_surge_file = output_dir / "timeseries_surge_blend.csv"
timeseries_cT_const_file = output_dir / "timeseries_cT_const.csv"
timeseries_cT_ts_file = output_dir / "timeseries_cT_blend.csv"

# motion timeseries file: time surge pitch ut
pd.DataFrame(
    {"time": t_full, "surge": surge_blend, "pitch": pitch_zeros, "ut": ut_blend}
).to_csv(timeseries_surge_file, index=False, sep=" ", header=False)

# constant CT' timeseries (debug / optional)
pd.DataFrame(
    {"time": t_full, "cT": np.ones_like(t_full) * CT_prime_mean}
).to_csv(timeseries_cT_const_file, index=False, sep=" ", header=False)

# blended CT' timeseries
pd.DataFrame(
    {"time": t_full, "cT": ctp_blend}
).to_csv(timeseries_cT_ts_file, index=False, sep=" ", header=False)

print(f"Generated: {timeseries_surge_file}")
print(f"Generated: {timeseries_cT_const_file}")
print(f"Generated: {timeseries_cT_ts_file}")
print(f"tstop={tstop:.6f}, dt_signal={dt_signal}, dt_solver={dt_solver}, n_hrs={n_hrs}")

# ============================================================
# Base inputs
# ============================================================
single_inputs = dict(
    sim=dict(
        inputdir=ju.DATA_PATH + curr_script_name + "_Files",
        outputdir=ju.DATA_PATH + curr_script_name + "_Files",
        CFL=-1,
        do_time_budgets=False,
        do_multi_phase_budgets=False,
        t_dataDump=100,
        nx=int(nx[0]),
        ny=int(ny[0]),
        nz=int(nz[0]),
        Lx=float(Lx[0]),
        Ly=float(Ly[0]),
        Lz=float(Lz[0]),
        dt=float(dt_solver),
        tstop=float(tstop),
    ),
    turb=dict(
        yLoc=float(Ly[0] / 2),
        zLoc=float(Lz[0] / 2),
        pitch_amplitude=0.0,
        filterWidth=float(filterWidth),
        useCorrection=bool(useCorrection),
        cT=float(CT_prime_mean),               # default constant CT'
        surge_amplitude=float(v_A),            # used if use_simple_periodic=True
        surge_freq=float(v_f),                 # used if use_simple_periodic=True
        motion_timeseries_file=str(timeseries_surge_file),   # used if enabled
        cT_timeseries_file=str(timeseries_cT_ts_file),       # used if enabled
        use_quals_style = False,
    ),
    run=dict(
        problem_dir="turbines",
        problem_name="AD_coriolis_shear",
        job_name=f"{case_name}_2x2",
        build_folder="build_opti_phase",
        queue="spr",
        n_hrs=int(n_hrs),
    ),
)

# ============================================================
# 4 exact combinations requested
# ============================================================
# (1) simple periodic surge + constant CT'
# (2) timeseries surge + constant CT'
# (3) simple periodic surge + timeseries CT'
# (4) timeseries surge + timeseries CT'
use_simple_periodic   = [True,  False, True,  False]
use_motion_timeseries = [False, True,  False, True ]
use_cT_timeseries     = [False, False, True,  True ]

varied_inputs = list(zip(use_simple_periodic, use_motion_timeseries, use_cT_timeseries))
varied_header = ["use_simple_periodic", "use_motion_timeseries", "use_cT_timeseries"]

for v in varied_inputs:
    print(v)

# write all simulation files
ju.write_padeops_suite(
    single_inputs,
    varied_inputs,
    varied_header=varied_header,
    default_input=default_inputs,
    sim_template=sim_template,
    run_template=run_template,
    turb_template=turb_template,
    node_cap=8,
)

print("Done: wrote 4-run simulation suite.")

# ============================================================
# Diagnostic plot: displacement, velocity, cT' (3x1)
# ============================================================
fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axs[0].plot(t_full, surge_blend, lw=1.2, label="surge_blend")
axs[0].plot(t_full, surge_sin, "--", lw=1.0, alpha=0.7, label="surge_sin (spinup ref)")
axs[0].set_ylabel("Displacement")
axs[0].set_title(f"{case_name}: blended inputs")
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc="best")

axs[1].plot(t_full, ut_blend, lw=1.2, label="ut_blend")
axs[1].plot(t_full, ut_sin, "--", lw=1.0, alpha=0.7, label="ut_sin (spinup ref)")
axs[1].set_ylabel("Velocity")
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc="best")

axs[2].plot(t_full, ctp_blend, lw=1.2, label="CT'_blend")
axs[2].axhline(CT_prime_mean, ls="--", lw=1.0, alpha=0.7, label="CT'_mean")
axs[2].set_ylabel("CT'")
axs[2].set_xlabel("Non-dimensional time")
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc="best")

for ax in axs:
    ax.axvline(spinup_time, color="k", ls=":", lw=1.0, alpha=0.8)
    ax.axvline(spinup_time + blend_time, color="k", ls=":", lw=1.0, alpha=0.8)
    ax.set_xlim(70, tstop)

fig.tight_layout()

plot_file = output_dir / "inputs_3x1.png"
fig.savefig(plot_file, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot: {plot_file}")