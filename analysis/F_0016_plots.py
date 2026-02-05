# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: simspadeops-Wg-7Zt3Y-py3.11
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Variables and Helper Functions

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
from UnifiedMomentumModel.Momentum import UnifiedMomentum, ThrustBasedUnified
from matplotlib.colors import ListedColormap
from scipy.stats import describe
from scipy.interpolate import interp1d
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# %matplotlib inline
mpl.rcParams['figure.dpi'] = 100

def calc_an(df, ud_key, uinf_key):
    return 1 - (df[ud_key] / df[uinf_key])

def calc_ct(df, ud_key, uinf_key):
    return np.sign(df[ud_key]) * df["Local Thrust Coefficient"] * (df[ud_key])**2 / (df[uinf_key] * np.cos(np.deg2rad(df["Tilt"]))**2)

def calc_ctp(df, ud_key, uinf_key):
    return (df["ct_Turb"]) / ((df[ud_key])**2 / (df[uinf_key])**2)

def calc_cp(df, uinf_key = "UInf_Turb"):  # power in PadeOps and UMM is calculated in the turbine frame of reference!
    return df.Power / (0.5 * rho * math.pi * (D/2)**2 * (df[uinf_key])**3)

def round_phases(phase):
    bin_width = 0.05
    n_bins = int(1 / bin_width)
    phase_idx = np.round(phase / bin_width).astype(int)
    phase_idx = np.clip(phase_idx, 0, n_bins)
    phase_bin = phase_idx * bin_width
    return phase_bin


rho, uinf, D, dt = 1, 1, 1, 0.05
Ap_vals = [0, 4, 8, 12, 16]
As_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
f_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
ct_vals = [1.33, 1.66, 2.00, 2.33]

# get UMM values
N_periods = 3
def get_tilt_vals(Ap, n_phase=100):
    phases = np.tile(np.linspace(0, 1, n_phase, endpoint=False), N_periods)
    theta = 2 * np.pi * phases
    tilt_deg = Ap * np.sin(theta)
    tilt_rad = np.deg2rad(tilt_deg)
    return tilt_rad, phases
 
def get_uturb_vals(As, f, n_phase=100):
    omega = 2 * np.pi * f
    phases = np.tile(np.linspace(0, 1, n_phase, endpoint=False), N_periods)
    theta = 2 * np.pi * phases
    uturb = As * np.cos(theta)              # velocity signal
    dx = (As / omega) * np.sin(theta)       # displacement
    return uturb, dx, phases


# %% [markdown]
# # Get UMM Data

# %%
model = UnifiedMomentum()
umm_stationary_Cts = [(ctp, model(ctp, yaw = 0.0, tilt = 0.0).Ct) for ctp in ct_vals]
umm_stationary_Cts = dict(umm_stationary_Cts)

umm_stationary_ans = [(ctp, model(ctp, yaw = 0.0, tilt = 0.0).an) for ctp in ct_vals]
umm_stationary_ans = dict(umm_stationary_Cts)

# %%
col_names = ["Local Thrust Coefficient", "Amplitude", "Distance", "Tilt", "Frequency", "UTurb", "an_Turb", "Ct_Turb", "Cp_Turb", "Phase", "Time"]
# get pitch values
pitch_umm_vals = []
for ct in ct_vals:
    for a in Ap_vals:
        for f in f_vals:
            tilts, phases = get_tilt_vals(a,)
            time = [i * dt for i in range(len(phases))]
            for (i, tilt) in enumerate(tilts):
                phase = phases[i]
                tilted_vals = model(ct, yaw = 0.0, tilt = tilt)
                vals = (ct, a, np.zeros_like(a), tilt, f, np.zeros_like(a), tilted_vals.an, tilted_vals.Ct, tilted_vals.Cp, phase, time[i])
                pitch_umm_vals.append(vals)
                
pitch_umm_vals = np.array(pitch_umm_vals)
df_umm_pitch = pd.DataFrame(pitch_umm_vals, columns = col_names)

# %%
# get surge amplitude
surge_umm_vals = []
for ct in ct_vals:
    umm_stationary = model(ct, yaw = 0.0, tilt = 0.0)
    umm_an_stationary = umm_stationary.an
    umm_Cp_stationary = umm_stationary.Cp
    umm_Ct_stationary = umm_stationary.Ct
    for a in As_vals:
        a = np.round(a, decimals = 1)
        for f in f_vals:
            if f >= 1 or a >= 1:
                dt = 0.025
            else:
                dt = 0.025
            uturb, dx, phases = get_uturb_vals(a, f)
            time = [i * dt for i in range(len(phases))]
            for (i, ut) in enumerate(uturb):
                x = dx[i]
                phase = phases[i]
                uinf_t = 1 - ut
                Cp = umm_Cp_stationary * (uinf_t**3)
                Ct = np.sign(uinf_t) * umm_Ct_stationary * (uinf_t**2)
                vals = (ct, a, a / (2 * np.pi * f), 0, f, ut, umm_an_stationary, Ct, Cp, phase, time[i])
                surge_umm_vals.append(vals)
surge_umm_vals = np.array(surge_umm_vals)
df_umm_surge = pd.DataFrame(surge_umm_vals, columns = col_names)

# %%
# create combined pitch + surge dataframe
df_umm_surge["Movement"] = "Surge"
df_umm_pitch["Movement"] = "Pitch"
df_umm = pd.concat([df_umm_surge, df_umm_pitch], ignore_index=True)
df_umm["UInf_Turb"] = (uinf - df_umm["UTurb"]) * np.cos(df_umm["Tilt"])
df_umm["UDisk_Turb"] = (1 - df_umm["an_Turb"]) * df_umm["UInf_Turb"]
df_umm["UDisk_Ground"] = df_umm["UDisk_Turb"] + df_umm["UTurb"]
df_umm["Phase_Rounded"] = round_phases(df_umm["Phase"])


# %% [markdown]
# # Get LES Data
#
# ## Filter LES Data

# %%
def keep_sim(g):
    if (
        g["Movement"].iloc[0] == "Surging"
        and (
            g["Frequency"].iloc[0] >= 1
            or g["Amplitude"].iloc[0] >= 1
        )
    ):
        t = np.sort(g["Time"].values)
        dt = t[1] - t[0]
        return dt < 0.05
    return True


# %%
def get_clean_les_data(file, sim_keys):
    # read in LES data, remove "bad" data, and rename key columns
    df_les = pd.read_csv(file)
    df_les = df_les.dropna()
    df_les["Model"] = "LES"
    df_les = df_les.rename(columns={'UDisk': 'UDisk_Turb', 'Thrust Coefficient': 'Local Thrust Coefficient'}) # disk velocity in the turbine frame of reference

    # calculate needed LES quantities per datapoint/timestep
    df_les["UDisk_Ground"] = df_les["UDisk_Turb"] + df_les["UTurb"] # disk velocity in the ground frame of reference
    df_les["UInf_Turb"] = (uinf - df_les["UTurb"]) * np.cos(df_les["Tilt"])
    df_les["UInf_Ground"] = uinf * np.cos(df_les["Tilt"])

    df_les["an_Turb"] = calc_an(df_les, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb")
    df_les["Ct_Turb"] = calc_ct(df_les, ud_key = "UDisk_Turb", uinf_key = "UInf_Ground")
    df_les["Cp_Turb"] = calc_cp(df_les, uinf_key = "UInf_Ground")

    les_stationary_runs = df_les[(df_les["Frequency"] == 0) & (df_les["Amplitude"] == 0)]
    if len(les_stationary_runs) > 0:
        les_stationary_ct_vals = les_stationary_runs.groupby(sim_keys)["Ct_Turb"].mean().reset_index()
        les_stationary_Cts = dict([
            (ctp, les_stationary_ct_vals[les_stationary_ct_vals["Local Thrust Coefficient"] == ctp]["Ct_Turb"].iloc[0])
            for ctp in ct_vals
        ])
    else:
        les_stationary_Cts = dict()

    # df_les = df_les[df_les["Frequency"] != 0.0]
    # df_les = df_les[(((df_les["Movement"] == "Pitch") & (df_les["Amplitude"] < 20)) | ((df_les["Movement"] == "Surge") & (df_les["Amplitude"] < 1)))]

    df_les = (
        df_les
        .groupby(["Movement", "Frequency", "Amplitude", "Local Thrust Coefficient"])
        .filter(lambda g: (g["Ct_Turb"].max() <= 100))
    )

    df_les = (
        df_les
        .groupby(["Movement", "Frequency", "Amplitude", "Local Thrust Coefficient"])
        .filter(keep_sim)
    )

    df_les["Phase"] = (df_les["Time"] * df_les["Frequency"]) % 1.0
    df_les["Phase_Rounded"] = round_phases(df_les["Phase"])
    return df_les, les_stationary_Cts


# %%
sim_keys = ["Movement", "Frequency", "Amplitude", "Local Thrust Coefficient"]
df_les, les_stationary_Cts = get_clean_les_data("/Users/sky/src/HowlandLab/data/sim_16_all_runs_data_points_02_01_26.csv", sim_keys)


# %% [markdown]
# ## Initial LES Analysis (not phase-binned)
#
# My idea here is that the UMM generated data has a very standardized shape. The maximum $C_T$ value is at $\phi = 0.5$, the minimum value is at $\phi = 0.0$, and the values cross over the mean $C_T$ value at $\phi = 0.25$ and $\phi = 0.75$.
#
# This isn't neccesarily true for the LES data. Therefore, I want to record the mean and max $C_T$ value for each period within a simulation and the phase that the min and max occur at. This should then be averaged over all periods within a simulation. I also want to record the phase as which the $C_T$ values within a simulation go from under the mean of the entire simulation to over the mean (around 0.25) and back from over the mean to under the mean (around 0.75). 

# %%
def add_period_index(g):
    g = g.sort_values("Time")
    phase = g["Phase"].values
    # new period whenever phase jumps backwards
    period = np.zeros(len(phase), dtype=int)
    period[1:] = np.cumsum(np.diff(phase) < 0)
    g["Period"] = period
    return g


# %%
def strict_mean_crossings(phase, val, val_norm):    
    up_crossings = []
    down_crossings = []
    for i in range(len(val) - 1):
        # UP CROSSING
        if val[i] < val_norm and val[i+1] >= val_norm:
            phi0, phi1 = phase[i], phase[i+1]
            val0, val1 = val[i], val[i+1]

            # handle wrap
            if phi1 < phi0:
                phi1 += 1.0

            f = (val_norm - val0) / (val1 - val0)
            phi_cross = phi0 + f * (phi1 - phi0)
            up_crossings.append(phi_cross % 1.0)

        # DOWN CROSSING
        if val[i] > val_norm and val[i+1] <= val_norm:
            phi0, phi1 = phase[i], phase[i+1]
            val0, val1 = val[i], val[i+1]

            if phi1 < phi0:
                phi1 += 1.0

            f = (val_norm - val0) / (val1 - val0)
            phi_cross = phi0 + f * (phi1 - phi0)
            down_crossings.append(phi_cross % 1.0)

    # enforce exactly one crossing per period outside
    if len(up_crossings) == 1 and len(down_crossings) == 1:
        return up_crossings[0], down_crossings[0]
    else:
        return np.nan, np.nan



# %%
def circular_mean(phases):
    """Compute mean of phases in [0,1) accounting for wrap-around."""
    angles = 2 * np.pi * np.array(phases)
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    mean_phase = mean_angle / (2 * np.pi)
    if mean_phase < 0:
        mean_phase += 1.0
    return mean_phase

def circular_std(phases):
    """Compute std of phases in [0,1) accounting for wrap-around."""
    angles = 2 * np.pi * np.array(phases)
    R = np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2)
    circ_std = np.sqrt(-2 * np.log(R)) / (2*np.pi)  # normalized to [0,1]
    return circ_std

def circ_diff(a, b):
    """Compute minimal signed difference between circular phases [0,1)."""
    delta = a - b
    # wrap to [-0.5, 0.5)
    delta = (delta + 0.5) % 1.0 - 0.5
    return delta


# %%
def analyze_simulation(
    g,
    stationary_cts,
    phase_key="Phase",
    normalize=False
):
    g = g.reset_index(drop=True)
    g = g.sort_values("Time")
    g = add_period_index(g)

    # Determine stationary CT
    local_stationary_ct = g["Local Thrust Coefficient"].iloc[0]
    ct_stationary = stationary_cts[local_stationary_ct]
    ct_stationary_norm = 1 if normalize else ct_stationary

    # Remove last period if incomplete
    period_counts = g["Period"].value_counts()
    last_period = g["Period"].max()
    full_count = period_counts.max()
    if period_counts[last_period] < full_count:
        g = g[g["Period"] != last_period]

    # Per-period storage
    mean_cts = []
    max_cts, max_phases = [], []
    min_cts, min_phases = [], []

    up_cross_phases = []
    down_cross_phases = []

    # Per-period phase differences
    extrema_phase_diffs = []
    crossover_phase_diffs = []
    up_to_max_diffs = []
    max_to_down_diffs = []
    skew = []

    # Loop through each period
    for _, p in g.groupby("Period"):
        phase = p[phase_key].values
        ct = p["Ct_Turb"].values

        if normalize:
            ct = ct / ct_stationary

        # Mean CT
        mean_ct = np.mean(ct)
        mean_cts.append(mean_ct)

        # Max CT
        i_max = np.argmax(ct)
        phi_max = phase[i_max]
        max_cts.append(ct[i_max])
        max_phases.append(phi_max)

        # Min CT
        i_min = np.argmin(ct)
        phi_min = phase[i_min]
        min_cts.append(ct[i_min])
        min_phases.append(phi_min)

        # Phase difference between extrema (circular)
        extrema_phase_diffs.append(circ_diff(phi_max, phi_min))

        # Mean crossings
        up, down = strict_mean_crossings(phase, ct, mean_ct)
        if not np.isnan(up) and not np.isnan(down):
            up_cross_phases.append(up)
            down_cross_phases.append(down)

            # Phase span between crossings
            crossover_phase_diffs.append(circ_diff(up, down))

            # Phase difference between crossings and maximum
            d_up = circ_diff(phi_max, up)
            d_down = circ_diff(down, phi_max)
            up_to_max_diffs.append(d_up)
            max_to_down_diffs.append(d_down)
            skew.append(d_up - d_down)

    return pd.Series({
        # Linear CT statistics
        "avg_mean_CT": np.mean(mean_cts),
        "std_mean_CT": np.std(mean_cts, ddof=1),

        "avg_max_CT": np.mean(max_cts),
        "std_max_CT": np.std(max_cts, ddof=1),

        "avg_min_CT": np.mean(min_cts),
        "std_min_CT": np.std(min_cts, ddof=1),

        # Circular phase statistics
        "avg_phase_max_CT": circular_mean(max_phases),
        "std_phase_max_CT": circular_std(max_phases),

        "avg_phase_min_CT": circular_mean(min_phases),
        "std_phase_min_CT": circular_std(min_phases),

        "avg_phase_under_to_over_mean": circular_mean(up_cross_phases),
        "std_phase_under_to_over_mean": circular_std(up_cross_phases),

        "avg_phase_over_to_under_mean": circular_mean(down_cross_phases),
        "std_phase_over_to_under_mean": circular_std(down_cross_phases),

        "avg_phase_extrema_diff": circular_mean(extrema_phase_diffs),
        "std_phase_extrema_diff": circular_std(extrema_phase_diffs),

        "avg_phase_crossover_diff": circular_mean(crossover_phase_diffs),
        "std_phase_crossover_diff": circular_std(crossover_phase_diffs),

        "avg_peak_skew_diff": circular_mean(skew),
        "std_peak_skew_diff": circular_std(skew),

        # Bookkeeping
        "n_periods": g["Period"].nunique(),
        "n_valid_crossings": len(crossover_phase_diffs),
    })



# %%
def remove_high_variation_sims(df, err):
    clean_df = df.copy()
    clean_df = clean_df[np.abs((clean_df["std_mean_CT"] / clean_df["avg_mean_CT"])) < err]
    clean_df = clean_df[np.abs((clean_df["std_max_CT"] / clean_df["avg_max_CT"])) < err]
    return clean_df


# %%
normalize = False

ct_metrics = [
    ("avg_mean_CT", "std_mean_CT"),
    ("avg_max_CT", "std_max_CT"),
    ("avg_min_CT", "std_min_CT"),
    ("avg_phase_max_CT", "std_phase_max_CT"),
    ("avg_phase_min_CT", "std_phase_min_CT"),
    ("avg_phase_under_to_over_mean", "std_phase_under_to_over_mean"),
    ("avg_phase_over_to_under_mean", "std_phase_over_to_under_mean"),
    ("avg_phase_extrema_diff", "std_phase_extrema_diff"),
    ("avg_phase_crossover_diff", "std_phase_crossover_diff"),
    ("avg_peak_skew_diff", "std_peak_skew_diff")
]


# %%
def get_les_umm_summary_data(df_les, df_umm, les_stationary_vals, umm_stationary_vals, metrics, movement_key = "Surge", phase_key = "Phase", normalize = False, err = 0.05):
    # get LES summaries
    les_surge_sim_summary = (
        df_les[df_les["Movement"] == movement_key]
        .groupby(sim_keys, as_index=False)
        .apply(
            analyze_simulation,
            stationary_cts=les_stationary_vals,
            phase_key = phase_key,
            normalize=normalize,
            include_groups=True
        )
        .reset_index(drop=False)   
    )
    les_surge_sim_summary = remove_high_variation_sims(les_surge_sim_summary, err)
    # get UMM summaries
    umm_surge_sim_summary = (
        df_umm[df_umm["Movement"] == movement_key]
        .groupby(sim_keys, as_index=False)
        .apply(
            analyze_simulation,
            stationary_cts=umm_stationary_vals,
            phase_key = phase_key,
            normalize=normalize,
            include_groups=True
        )
        .reset_index(drop=False)   
    )
    umm_surge_sim_summary = remove_high_variation_sims(umm_surge_sim_summary, err)

    # merge summaries
    diff_surge_sim_summary = (
        umm_surge_sim_summary
        .merge(
            les_surge_sim_summary,
            on=sim_keys,
            suffixes=("_UMM", "_LES"),
            how="inner"
        )
    )

    # compute differences and quadrature std
    for mean_col, std_col in metrics:
        diff_col = f"{mean_col}_diff"
        std_diff_col = f"{std_col}_diff"

        if "phase" in mean_col:
            # use circular difference
            diff_surge_sim_summary[diff_col] = circ_diff(
                diff_surge_sim_summary[f"{mean_col}_UMM"],
                diff_surge_sim_summary[f"{mean_col}_LES"]
            )
        else:
            # linear difference for CT metrics
            diff_surge_sim_summary[diff_col] = (
                diff_surge_sim_summary[f"{mean_col}_UMM"]
                - diff_surge_sim_summary[f"{mean_col}_LES"]
            )

        # std adds in quadrature
        diff_surge_sim_summary[std_diff_col] = np.sqrt(
            diff_surge_sim_summary[f"{std_col}_UMM"]**2
            + diff_surge_sim_summary[f"{std_col}_LES"]**2
        )
    return diff_surge_sim_summary, les_surge_sim_summary, umm_surge_sim_summary


# %%
df_les

# %%
diff_surge_sim_summary, les_surge_sim_summary, umm_surge_sim_summary = get_les_umm_summary_data(df_les, df_umm, les_stationary_Cts, umm_stationary_Cts,
                                                                                                ct_metrics, movement_key = "Surge", phase_key = "Phase",
                                                                                                normalize = normalize, err = 0.05)

# %%
ct_ylabel_raw = {
    "avg_mean_CT": r"$\overline{C_T}$",
    "avg_max_CT": r"$\overline{\max(C_T)}$",
    "avg_min_CT": r"$\overline{\min(C_T)}$",

    "avg_phase_max_CT": r"$\overline{\phi_{\max(C_T)}}$",
    "avg_phase_min_CT": r"$\overline{\phi_{\min(C_T)}}$",

    "avg_phase_under_to_over_mean":
        r"$\overline{\phi_{C_T \uparrow C_{T_{\text{FB}}}}}$",

    "avg_phase_over_to_under_mean":
        r"$\overline{\phi_{C_T \downarrow C_{T_{\text{FB}}}}}$",

    "avg_phase_extrema_diff":
        r"$\overline{\phi_{\mathrm{max-min}}}$",

    "avg_phase_crossover_diff":
        r"$\overline{\phi_{\mathrm{\uparrow - \downarrow}}}$",

    "avg_phase_up_to_max_diff":
        r"$\overline{\phi_{\mathrm{\uparrow - max}}}$",

    "avg_phase_max_to_down_diff":
        r"$\overline{\phi_{\mathrm{max - \downarrow}}}$",

    "avg_peak_skew_diff":
        r" $\overline{\Delta \phi_{\uparrow\to\max} - \Delta \phi_{\max\to\downarrow}}$"

}
ct_ylabel_normalized = {
    "avg_mean_CT":
        r"$\overline{C_T / C_{T_{FB}}}$",

    "avg_max_CT":
        r"$\overline{\max\!\left(C_T / C_{T_{FB}}\right)}$",

    "avg_min_CT":
        r"$\overline{\min\!\left(C_T / C_{T_{FB}}\right)}$",

    "avg_phase_max_CT":
        r"$\overline{\phi_{\max(C_T)}}$",

    "avg_phase_min_CT":
        r"$\overline{\phi_{\min(C_T)}}$",

    "avg_phase_under_to_over_mean":
        r"$\overline{\phi_{C_T \uparrow C_{T_{FB}}}}$",

    "avg_phase_over_to_under_mean":
        r"$\overline{\phi_{C_T \downarrow C_{T_{FB}}}}$",

    "avg_phase_extrema_diff":
        r"$\overline{\phi_{\mathrm{max-min}}}$",

    "avg_phase_crossover_diff":
        r"$\overline{\phi_{\mathrm{\uparrow - \downarrow}}}$",

    "avg_phase_up_to_max_diff":
        r"$\overline{\phi_{\mathrm{\uparrow - max}}}$",

    "avg_phase_max_to_down_diff":
        r"$\overline{\phi_{\mathrm{max - \downarrow}}}$",

    "avg_peak_skew_diff":
        r" $\overline{\Delta \phi_{\uparrow\to\max} - \Delta \phi_{\max\to\downarrow}}$"
}

def get_ct_ylabel(metric, normalize=False):
    return (
        ct_ylabel_normalized[metric]
        if normalize
        else ct_ylabel_raw[metric]
    )


# %%
def umm_les_data_plot(df,
    y_key="avg_max_CT",
    y_err_key="std_max_CT",   # <-- column with SEM or std
    y_label = "$\\overline{C_{T_{\\text{LES}}}}$",
    linewidth = 1,
    title = ""
):
    g = sns.FacetGrid(
        df,
        col="Local Thrust Coefficient",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.2,
        col_wrap=2
    )

    # Global frequency ordering
    freqs_all = np.sort(df["Frequency"].unique())
    freqs_plot = freqs_all[::-1]   # largest plotted first

    # Consistent colors across facets
    colors = sns.color_palette("tab10", len(freqs_all))
    color_map = dict(zip(freqs_all, colors))

    for ltc, ax in g.axes_dict.items():
        df_ltc = df[df["Local Thrust Coefficient"] == ltc]

        for freq in freqs_plot:
            df_sub = (
                df_ltc[df_ltc["Frequency"] == freq]
                .sort_values("Amplitude")
            )

            if df_sub.empty:
                continue

            # --- error bars + line (transparent) ---
            ax.errorbar(
                df_sub["Amplitude"],
                df_sub[y_key],
                yerr=df_sub[y_err_key] if y_err_key in df_sub else None,
                color=color_map[freq],
                linestyle="solid",
                linewidth=linewidth,
                marker=None,
                capsize=3,
                alpha=0.5,
                label=f"{freq:.2f} St",
                zorder = 1
            )

            # --- markers only (opaque) ---
            ax.plot(
                df_sub["Amplitude"],
                df_sub[y_key],
                linestyle="None",
                marker="o",
                color=color_map[freq],
                zorder = 2,
                alpha=0.65,
            )

    # Axis + titles
    g.set_axis_labels("$A_S = \max{(U_{t} / U_\infty)}$", y_label)
    g.set_titles(col_template=r"$C_T'$ = {col_name}")
    g.fig.suptitle(title, y = 1.125)

    # --- Create custom legend ---
    color_handles = [
        mlines.Line2D([], [], color=color_map[f], linestyle='solid', linewidth=1, marker = "o")
        for f in freqs_plot[::-1]
    ]
    color_labels = [f"{f:.2f} St" for f in freqs_plot[::-1]]

    g.fig.legend(
        color_handles,
        color_labels,
        title="Frequency",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=6,
        frameon=False,
    )


# %%
for metric in ct_metrics:
    y_key, y_err_key = metric

    y_key = y_key
    y_err_key = y_err_key
    y_label = "UMM: " + get_ct_ylabel(y_key, normalize=normalize)

    # umm_les_data_plot(umm_surge_sim_summary, y_key, y_err_key, y_label)
    

# %%
for metric in ct_metrics:
    y_key, y_err_key = metric

    y_key = y_key
    y_err_key = y_err_key
    y_label = "LES: " + get_ct_ylabel(y_key, normalize=normalize)

    # umm_les_data_plot(les_surge_sim_summary, y_key, y_err_key, y_label)
    

# %%
for metric in ct_metrics:
    y_key, y_err_key = metric

    y_label = "$\Delta$" + get_ct_ylabel(y_key, normalize=normalize)

    y_key = y_key + "_diff"
    y_err_key = y_err_key + "_diff"

    umm_les_data_plot(diff_surge_sim_summary, y_key, y_err_key, y_label, title = "Surging Turbine Difference between UMM and LES")

# %%
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
CONTOUR_LABEL_FONTSIZE = 12
LINE_WIDTH = 1.5

# ykey = "avg_mean_CT_diff"
ykey = "avg_max_CT_diff"
# ykey = "avg_min_CT_diff"
# ykey = "avg_phase_max_CT_diff"

# ---- Get thrust coefficient values ----
ct_vals = np.sort(diff_surge_sim_summary["Local Thrust Coefficient"].unique())
assert len(ct_vals) == 4, "Expected exactly four Local Thrust Coefficient values"

# ---- Global contour levels (even spacing) ----
vmin = diff_surge_sim_summary[ykey].min()
vmax = diff_surge_sim_summary[ykey].max()
levels = np.linspace(vmin, vmax, 12)


# ---- Figure setup ----
fig, axes = plt.subplots(
    2, 2,
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, ct in zip(axes, ct_vals):
    sub = diff_surge_sim_summary[
        diff_surge_sim_summary["Local Thrust Coefficient"] == ct
    ]

    # Pivot to grid
    Z = sub.pivot(
        index="Frequency",
        columns="Amplitude",
        values=ykey
    )

    X = Z.columns.values
    Y = Z.index.values
    Xg, Yg = np.meshgrid(X, Y)

    # ---- Filled contours ----
    cf = ax.contourf(
        Xg, Yg, Z.values,
        levels=levels,
        extend="both"
    )

    # ---- Contour lines ----
    cs = ax.contour(
        Xg, Yg, Z.values,
        levels=levels,
        linewidths=LINE_WIDTH,
        colors="k"
    )

    ax.clabel(
        cs,
        inline=True,
        fontsize=CONTOUR_LABEL_FONTSIZE,
        fmt="%.2f"
    )
    non_nan_mask = ~(np.isnan(Z.values))
    ax.scatter(Xg[non_nan_mask], Yg[non_nan_mask], marker = "+", color = "k")

    # ---- Labels & formatting ----
    ax.set_title(f"$C_T' = {ct}$", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Amplitude", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Frequency", fontsize=LABEL_FONTSIZE)

    ax.tick_params(labelsize=TICK_FONTSIZE)

# ---- Shared colorbar ----
# Leave space on the right for the colorbar
fig.subplots_adjust(right=0.86)
# Add a new axis for the colorbar
cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
cbar = fig.colorbar(cf, cax=cax)
cbar.set_label("avg_max_CT_diff", fontsize=LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

fig.subplots_adjust(
    left=0.08,    # space on left
    right=0.86,   # leave room for colorbar on right
    bottom=0.1,   # space at bottom for x-labels
    top=0.92,     # space at top for titles
    wspace=0.3,   # horizontal spacing between subplots
    hspace=0.35   # vertical spacing between subplots
)


# %%
def check_movement_values_grid_dual(
    df_les,
    df_umm,
    x_key="Phase_Rounded",
    y_key="Ct_Turb",
    ylabel=None,
):
    # Get unique facet values
    freqs = sorted(df_les["Frequency"].unique())
    ltcs = sorted(df_les["Local Thrust Coefficient"].unique())
    amps = sorted(df_les["Amplitude"].unique())

    # Create FacetGrid by row=Frequency, col=LTC
    g = sns.FacetGrid(
        df_les,
        row="Frequency",
        col="Local Thrust Coefficient",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.2
    )

    # Consistent color map by amplitude
    palette = sns.color_palette("tab10", n_colors=len(amps))
    color_map = dict(zip(amps, palette))

    # Marker/linestyle map for dataset
    dataset_styles = {
        "LES": {"linestyle": "-", "marker": "o"},
        "UMM": {"linestyle": "--", "marker": "s"}
    }

    # Loop over all facets
    for (freq, ltc), ax in g.axes_dict.items():
        df_les_sub = df_les[
            (df_les["Frequency"] == freq) &
            (df_les["Local Thrust Coefficient"] == ltc)
        ].sort_values("Phase_Rounded")
        df_umm_sub = df_umm[
            (df_umm["Frequency"] == freq) &
            (df_umm["Local Thrust Coefficient"] == ltc)
        ].sort_values("Phase_Rounded")

        ltc_val = les_stationary_Cts[ltc]

        for amp in amps:
            # fixed bottom
            ax.hlines(y = ltc_val, xmin = 0, xmax = 1, label = "Fixed Bottom $C_T$", color = "k", linestyle = "--")
            # LES
            df_amp_les = df_les_sub[df_les_sub["Amplitude"] == amp]
            if not df_amp_les.empty:
                ax.plot(
                    df_amp_les[x_key],
                    df_amp_les[y_key],
                    color=color_map[amp],
                    linestyle=dataset_styles["LES"]["linestyle"],
                    marker=dataset_styles["LES"]["marker"],
                    alpha=0.7,
                    zorder = 2
                )

            # UMM
            df_amp_umm = df_umm_sub[df_umm_sub["Amplitude"] == amp]
            if not df_amp_umm.empty:
                ax.plot(
                    df_amp_umm[x_key],
                    df_amp_umm[y_key],
                    color=color_map[amp],
                    linestyle=dataset_styles["UMM"]["linestyle"],
                    marker=dataset_styles["UMM"]["marker"],
                    alpha=0.7,
                    zorder = 2
                )

    # Axis labels and titles
    if ylabel is None:
        ylabel = y_key
    g.set_axis_labels("Phase $\phi$", ylabel, size=12)
    g.set_titles(col_template=r"$C_T'$ = {col_name}", row_template="f = {row_name} Hz", size=14)
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

    # --- Create two-part legend ---
    import matplotlib.lines as mlines

    # 1) Colors for amplitudes
    amp_handles = [
        mlines.Line2D([], [], color=color_map[amp], marker='o', linestyle='None', markersize=8)
        for amp in amps
    ]
    amp_labels = [f"$A_S = {amp:.1f}$" for amp in amps]

    # 2) Shapes/linestyles for datasets
    dataset_handles = (
        [mlines.Line2D([], [], color='k', marker=dataset_styles[ds]["marker"],
                    linestyle=dataset_styles[ds]["linestyle"], markersize=8)
        for ds in ["LES", "UMM"]] +
        [mlines.Line2D([], [], color='k', linestyle='--', markersize=8)]
    )

    dataset_labels = ["LES", "UMM", "Fixed Bottom $C_T$"]


    # Add both legends
    leg1 = g.fig.legend(amp_handles, amp_labels, loc="upper center",
                        ncol=3, frameon=False, fontsize=12, title_fontsize=12)

    leg2 = g.fig.legend(dataset_handles, dataset_labels, loc="upper center",
                        ncol=len(dataset_handles), frameon=False, fontsize=12, title_fontsize=12,
                        bbox_to_anchor=(0.5, 0.96))
    

    # Make sure both are visible
    g.fig.add_artist(leg1)

# %%
diff_surge_sim_summary

# %%
diff_pitch_sim_summary, les_pitch_sim_summary, umm_pitch_sim_summary = get_les_umm_summary_data(df_les, df_umm, les_stationary_Cts, umm_stationary_Cts,
                                                                                                ct_metrics, movement_key = "Pitch", phase_key = "Phase",
                                                                                                normalize = normalize, err = 0.05,
)

# %%
for metric in ct_metrics:
    y_key, y_err_key = metric

    y_label = "$\Delta$" + get_ct_ylabel(y_key, normalize=normalize)

    y_key = y_key + "_diff"
    y_err_key = y_err_key + "_diff"

    if "phase" not in y_key:
        umm_les_data_plot(diff_pitch_sim_summary, y_key, y_err_key, y_label, title = "Pitching Turbine Difference between UMM and LES")
    

# %% [markdown]
# ## Surging + Pitching Runs

# %%
df_both_les, _ = get_clean_les_data("/Users/sky/src/HowlandLab/data/sim_20_all_runs_data_points_02_01_26.csv", sim_keys + ["PitchAmp"])

df_both_les = df_both_les[(df_both_les["Amplitude"] < 0.8) & (df_both_les["Frequency"] < 1)]


# %%
def add_surge_only_to_both(df_both_les, df_les):
    """
    Add surge-only simulations from df_single_les into df_both_les.
    Sets Pitch_Amp = 0 and matches on Frequency, Surge_Amp, and Local Thrust Coefficient.
    Avoids duplicates.
    """
    df_surge_les = df_les[df_les["Movement"] == "Surge"]
    # Columns to match
    match_cols = ["Frequency", "Amplitude", "Local Thrust Coefficient"]

    # Only keep rows in df_surge that exist in df_both_les (matching all three columns)
    matching_surge = df_surge_les.merge(
        df_both_les[match_cols],
        on=match_cols,
        how="inner"
    )

    # Set Pitch_Amp to zero for these rows
    matching_surge = matching_surge.copy()
    matching_surge["PitchAmp"] = 0

    # Add to df_both_les
    df_both_les = pd.concat([df_both_les, matching_surge], ignore_index=True)
    return df_both_les


# %%
les_surge_pitch_sim_summary = (
    df_both_les
    .groupby(sim_keys + ["PitchAmp"])
    .apply(
        analyze_simulation,
        stationary_cts=les_stationary_Cts,
        phase_key = "Phase",
        normalize=normalize,
        include_groups=True
    )
    .reset_index()
)
les_surge_pitch_sim_summary = remove_high_variation_sims(les_surge_pitch_sim_summary, 0.05)

# %%
les_surge_pitch_sim_summary = add_surge_only_to_both(les_surge_pitch_sim_summary, les_surge_sim_summary)


# %%
def umm_les_both_data_plot(df, y_key, y_err_key, y_label, title = "Pitching Turbine Difference between UMM and LES"):
    g = sns.FacetGrid(
        df,
        col="Local Thrust Coefficient",
        row = "Frequency",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.2,
    )

    # Global frequency ordering
    pitch_amp_all = np.sort(df["PitchAmp"].unique())
    pitch_amp_plot = pitch_amp_all[::-1]   # largest plotted first

    # Consistent colors across facets
    colors = sns.color_palette("tab10", len(pitch_amp_all))
    color_map = dict(zip(pitch_amp_all, colors))

    for (freq, ltc), ax in g.axes_dict.items():
        df_ltc_freq = df[(df["Local Thrust Coefficient"] == ltc) & (df["Frequency"] == freq)]
        for pamp in pitch_amp_plot:
            df_sub = (
                df_ltc_freq[df_ltc_freq["PitchAmp"] == pamp]
                .sort_values("Amplitude")
            )

            if df_sub.empty:
                continue

            # --- error bars + line (transparent) ---
            ax.errorbar(
                df_sub["Amplitude"],
                df_sub[y_key],
                yerr=df_sub[y_err_key] if y_err_key in df_sub else None,
                color=color_map[pamp],
                linestyle="solid",
                linewidth=1,
                marker=None,
                capsize=3,
                alpha=0.5,
                label=f"{pamp:.2f} Pitch Amplitude",
                zorder = 1
            )

            # --- markers only (opaque) ---
            ax.plot(
                df_sub["Amplitude"],
                df_sub[y_key],
                linestyle="None",
                marker="o",
                color=color_map[pamp],
                zorder = 2,
                alpha=0.65,
            )

    # Axis + titles
    g.set_axis_labels("$A_S = \max{(U_{t} / U_\infty)}$", y_label)
    g.set_titles(col_template=r"$C_T'$ = {col_name}")
    g.fig.suptitle(title, y = 1.125)

    # --- Create custom legend ---
    color_handles = [
        mlines.Line2D([], [], color=color_map[p], linestyle='solid', linewidth=1, marker = "o")
        for p in pitch_amp_plot[::-1]
    ]
    color_labels = [f"{f:.2f} degrees" for f in pitch_amp_plot[::-1]]

    g.fig.legend(
        color_handles,
        color_labels,
        title="Pitch Amplitude",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=6,
        frameon=False,
    )


# %%
def umm_les_both_data_plot_freq(df, y_key, y_err_key, y_label, title="Pitching Turbine Difference between UMM and LES"):
    g = sns.FacetGrid(
        df,
        col="Local Thrust Coefficient",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.2,
        col_wrap=2
    )

    # Pitch amplitudes as colors
    pitch_amp_all = np.sort(df["PitchAmp"].unique())
    colors = sns.color_palette("tab10", len(pitch_amp_all))
    pitch_color_map = dict(zip(pitch_amp_all, colors))

    # Frequencies as marker shapes
    freqs_all = np.sort(df["Frequency"].unique())
    markers = ['o', 's', 'P', 'd']  # extend if needed
    freq_marker_map = dict(zip(freqs_all, markers[:len(freqs_all)]))

    for ltc, ax in g.axes_dict.items():
        df_ltc = df[df["Local Thrust Coefficient"] == ltc]

        for freq in freqs_all:
            df_freq = df_ltc[df_ltc["Frequency"] == freq]
            for pamp in pitch_amp_all:
                df_sub = df_freq[df_freq["PitchAmp"] == pamp].sort_values("Amplitude")
                if df_sub.empty:
                    continue

                # Error bars + line (transparent)
                ax.errorbar(
                    df_sub["Amplitude"],
                    df_sub[y_key],
                    yerr=df_sub[y_err_key] if y_err_key in df_sub else None,
                    color=pitch_color_map[pamp],
                    linestyle='solid',
                    linewidth=1,
                    marker=None,
                    capsize=3,
                    alpha=0.5,
                    zorder=1
                )

                # Markers only (opaque)
                ax.plot(
                    df_sub["Amplitude"],
                    df_sub[y_key],
                    linestyle='None',
                    marker=freq_marker_map[freq],
                    color=pitch_color_map[pamp],
                    zorder=2,
                    markeredgecolor='k',
                    markeredgewidth=0.5,
                )


    # Axis + titles
    g.set_axis_labels("$A_S = \max(U_t / U_\infty)$", y_label)
    g.set_titles(col_template=r"$C_T'$ = {col_name}")
    g.fig.suptitle(title, y=1.25)

    # --- Pitch amplitude legend (colors) ---
    pitch_handles = [
        mlines.Line2D([], [], color=pitch_color_map[p], marker='o', linestyle='None', markersize=6)
        for p in pitch_amp_all
    ]
    pitch_labels = [f"{int(round(p))}°" for p in pitch_amp_all]


    # --- Frequency legend (shapes) ---
    freq_handles = [
        mlines.Line2D([], [], color='k', marker=freq_marker_map[f], linestyle='None', markersize=6)
        for f in freqs_all
    ]
    freq_labels = [f"{f:.2f} St" for f in freqs_all]

    # Combine legends
    g.fig.legend(
        pitch_handles, pitch_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=4,
        frameon=False,
        title_fontsize=10,
        handletextpad=0.5
    )
    g.fig.legend(
        freq_handles, freq_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        title_fontsize=10,
        handletextpad=0.5
    )


# %%
for metric in ct_metrics:
    y_key, y_err_key = metric

    y_label = get_ct_ylabel(y_key, normalize=normalize)

    y_key = y_key
    y_err_key = y_err_key

    # umm_les_both_data_plot(les_surge_pitch_sim_summary, y_key, y_err_key, y_label, title = "Surging and Pitching Turbine LES")

# %%
for metric in ct_metrics:
    y_key, y_err_key = metric

    y_label = get_ct_ylabel(y_key, normalize=normalize)

    y_key = y_key
    y_err_key = y_err_key

    umm_les_both_data_plot_freq(les_surge_pitch_sim_summary, y_key, y_err_key, y_label, title = "Surging and Pitching Turbine LES")

# %% [markdown]
# ## Phase-Rounded Analysis

# %%
df_les["Phase_Rounded"] = round_phases(df_les["Phase"])


# %%
def first_max_phase(
    df,
    var_col="Ct_Turb",
):
    results = []

    grouped = df.groupby(["Local Thrust Coefficient", "Frequency", "Amplitude",  "Movement"])
    
    for (ltc, freq, amp, move), group in grouped:
        group = group.sort_values("Phase")
        phases = group["Phase"].values
        values = group[var_col].values

        if len(values) == 0:
            continue

        # find index of max Ct_Turb
        idx_max = np.argmax(values)

        # get the phase at that index
        phase_max = phases[idx_max]

        results.append({
            "Local Thrust Coefficient": ltc,
            "Frequency": freq,
            "Amplitude": amp,
            "Movement": move,
            "Phase_Diff": 0.5 - phase_max,
        })

    return pd.DataFrame(results)



# %%
phase_max_les_vals = first_max_phase(df_les[df_les["Movement"] == "Surge"])

g = sns.FacetGrid(
    phase_max_les_vals,
    col="Local Thrust Coefficient",
    margin_titles=True,
    sharex=True,
    sharey=True,
    height=3,
    aspect=1.2,
    col_wrap=2
)

# Global frequency ordering
freqs_all = np.sort(phase_max_les_vals["Frequency"].unique())
freqs_plot = freqs_all[::-1]   # largest plotted first
print(freqs_plot)

# Consistent colors across facets
colors = sns.color_palette("tab10", len(freqs_all))
color_map = dict(zip(freqs_all, colors))

marker_map = dict(zip(freqs_all, ["P", "v", "*", "d", "o", "s", "p", "H"]))

for ltc, ax in g.axes_dict.items():
    df_ltc = phase_max_les_vals[phase_max_les_vals["Local Thrust Coefficient"] == ltc]

    for freq in freqs_plot:
        df_sub = (
            df_ltc[df_ltc["Frequency"] == freq]
            .sort_values("Amplitude")
        )

        if df_sub.empty:
            continue

        ax.scatter(
            df_sub["Amplitude"],
            df_sub["Phase_Diff"],
            color=color_map[freq],
            marker=marker_map[freq],
            linewidth=2,
            label=f"{freq:.2f} St",
            alpha = 0.4
        )


# Axis + titles
g.set_axis_labels(
    "$A_S = \max{(U_{t} / U_\infty)}$",
    "Phase of First Maximum $C_T$"
)
g.set_titles(col_template=r"$C_T'$ = {col_name}")

# --- Create custom legend ---
# 1. Color handles for frequency
color_handles = [
    mlines.Line2D([], [], color=color_map[f], linestyle='solid', linewidth=3)
    for f in freqs_plot[::-1]  # show smallest first
]
color_labels = [f"{f:.2f} St" for f in freqs_plot[::-1]]


# Place both legends
g.fig.legend(
    color_handles,
    color_labels,
    title="Frequency",
    loc="upper center",
    ncol=3,
    frameon=False,
)

plt.tight_layout(rect=[0, 0, 1, 0.9])


# %%
# group LES and UMM by movement, frequency, amplitude, and rounded phase
group_keys = ["Local Thrust Coefficient", "Frequency", "Amplitude", "Movement", "Phase_Rounded"]
value_keys = ["an_Turb", "Ct_Turb", "Cp_Turb"]
df_umm_avg = df_umm.groupby(group_keys)[value_keys].mean().reset_index()
df_les_avg = df_les.groupby(group_keys)[value_keys].mean().reset_index()

# Select only the columns we need for comparison
cols_to_merge = group_keys + value_keys
df_umm_sub = df_umm_avg[cols_to_merge]
df_les_sub = df_les_avg[cols_to_merge]


# %%
def check_movement_values_grid(df, x_key = "Phase_Rounded", y_key = "Ct_Turb", ylabel = None):
    g = sns.FacetGrid(
        df,
        row="Frequency",
        col="Local Thrust Coefficient",
        hue="Amplitude",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.2,
    )

    g.map_dataframe(
        sns.lineplot,
        x=x_key,
        y=y_key,
        marker="o"
    )

    g.add_legend(title="Amplitude")

    if ylabel is None:
        ylabel = y_key

    g.set_axis_labels("Phase", ylabel)
    g.set_titles(col_template=r"$C_T'$ = {col_name}",
                row_template="f = {row_name} Hz")


# %%
df_umm_surge = df_umm_sub[df_umm_sub["Movement"] == "Surge"]
df_umm_pitch = df_umm_sub[df_umm_sub["Movement"] == "Pitch"]
check_movement_values_grid(df_umm_pitch, y_key = "Ct_Turb")

# %%
df_les_surge = df_les_sub[df_les_sub["Movement"] == "Surge"]
check_movement_values_grid(df_les_surge, y_key = "Ct_Turb")

# %%
df_les_surge["Shifted_Phase_Rounded"] = np.mod(df_les_surge["Phase_Rounded"] + 0.5, 1)
check_movement_values_grid(df_les_surge, x_key = "Shifted_Phase_Rounded", y_key = "Ct_Turb")

# %%
df_les_pitch = df_les_sub[df_les_sub["Movement"] == "Pitch"]
check_movement_values_grid(df_les_pitch, y_key = "Ct_Turb")

# %%
sub_les = df_les_surge[
    ((df_les_surge["Local Thrust Coefficient"] == 1.33) | (df_les_surge["Local Thrust Coefficient"] == 2.00)) &
    ((df_les_surge["Frequency"] == 0.4) | (df_les_surge["Frequency"] == 0.8)) &
    ((df_les_surge["Amplitude"] == 0.2) | (df_les_surge["Amplitude"] == 0.6) | (df_les_surge["Amplitude"] == 1.2))
]
sub_umm = df_umm_surge[
    ((df_umm_surge["Local Thrust Coefficient"] == 1.33) | (df_umm_surge["Local Thrust Coefficient"] == 2.00)) &
    ((df_umm_surge["Frequency"] == 0.4) | (df_umm_surge["Frequency"] == 0.8)) &
    ((df_umm_surge["Amplitude"] == 0.2) | (df_umm_surge["Amplitude"] == 0.6) | (df_umm_surge["Amplitude"] == 1.2))
]

check_movement_values_grid_dual( #TODO: it would be good to swap these to 
    sub_les,
    sub_umm,
    y_key="Ct_Turb",
    ylabel = "$C_T$",
    )

# %%
sub_les = df_les_pitch[
    ((df_les_pitch["Local Thrust Coefficient"] == 1.33) | (df_les_pitch["Local Thrust Coefficient"] == 2.00)) &
    ((df_les_pitch["Frequency"] == 0.4) | (df_les_pitch["Frequency"] == 0.8)) &
    ((df_les_pitch["Amplitude"] == 4) | (df_les_pitch["Amplitude"] == 8) | (df_les_pitch["Amplitude"] == 16))
]
sub_umm = df_umm_pitch[
    ((df_umm_pitch["Local Thrust Coefficient"] == 1.33) | (df_umm_pitch["Local Thrust Coefficient"] == 2.00)) &
    ((df_umm_pitch["Frequency"] == 0.4) | (df_umm_pitch["Frequency"] == 0.8)) &
    ((df_umm_pitch["Amplitude"] == 4) | (df_umm_pitch["Amplitude"] == 8) | (df_umm_pitch["Amplitude"] == 16))
]

check_movement_values_grid_dual( #TODO: it would be good to swap these to 
    sub_les,
    sub_umm,
    y_key="Ct_Turb",
    ylabel = "$C_T$",
    )

# %%
# Merge on the grouping keys
df_merged = pd.merge(
    df_les_sub,
    df_umm_sub,
    on=group_keys,
    suffixes=("_LES", "_UMM")
)

df_merged["an_diff"] =  df_merged["an_Turb_UMM"] - df_merged["an_Turb_LES"]
df_merged["Ct_diff"] =  df_merged["Ct_Turb_UMM"] - df_merged["Ct_Turb_LES"]
df_merged["Cp_diff"] =  df_merged["Cp_Turb_UMM"] - df_merged["Cp_Turb_LES"]

# %%
df_merge_surge = df_merged[df_merged["Movement"] == "Surge"]
check_movement_values_grid(df_merge_surge, y_key = "Ct_diff", ylabel = "$C_{T_\\text{UMM}} - C_{T_\\text{LES}}$")
# df_merge_pitch = df_merged[df_merged["Movement"] == "Pitch"]
# check_movement_values_grid(df_merge_pitch, y_key = "Ct_diff")

# %%
# Aggregate function dictionary
result_keys = ["Local Thrust Coefficient", "Frequency", "Amplitude", "Movement"]
agg_dict = {
    "an_Turb_LES": ["mean", "max", "min"],
    "Ct_Turb_LES": ["mean", "max", "min"],
    "Cp_Turb_LES": ["mean", "max", "min"],
    "an_Turb_UMM": ["mean", "max", "min"],
    "Ct_Turb_UMM": ["mean", "max", "min"],
    "Cp_Turb_UMM": ["mean", "max", "min"],
    "an_diff": ["mean", "max", "min"],
    "Ct_diff": ["mean", "max", "min"],
    "Cp_diff": ["mean", "max", "min"]
}

# Group and aggregate
df_stats = df_merged.groupby(result_keys).agg(agg_dict).reset_index()
df_stats.columns = [
    f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
    for col in df_stats.columns
]

# %%
# # Assuming Phase_Rounded is normalized 0-1
# left_mask = df_merged["Phase_Rounded"] < 0.5
# right_mask = df_merged["Phase_Rounded"] >= 0.5
# center_mask = (df_merged["Phase_Rounded"] >= 0.25) & (df_merged["Phase_Rounded"] <= 0.75)
# edge_mask = (df_merged["Phase_Rounded"] < 0.25) | (df_merged["Phase_Rounded"] > 0.75)

# %%
# # Max left and right
# LES_max_left = df_merged[left_mask].groupby(result_keys)["Ct_Turb_LES"].max().rename("Ct_Turb_LES_max_left")
# LES_max_right = df_merged[right_mask].groupby(result_keys)["Ct_Turb_LES"].max().rename("Ct_Turb_LES_max_right")
# UMM_max_left = df_merged[left_mask].groupby(result_keys)["Ct_Turb_UMM"].max().rename("Ct_Turb_UMM_max_left")
# UMM_max_right = df_merged[right_mask].groupby(result_keys)["Ct_Turb_UMM"].max().rename("Ct_Turb_UMM_max_right")
# # Min center and edge
# LES_min_center = df_merged[center_mask].groupby(result_keys)["Ct_Turb_LES"].min().rename("Ct_Turb_LES_min_center")
# LES_min_edge = df_merged[edge_mask].groupby(result_keys)["Ct_Turb_LES"].min().rename("Ct_Turb_LES_min_edge")
# UMM_min_center = df_merged[center_mask].groupby(result_keys)["Ct_Turb_UMM"].min().rename("Ct_Turb_UMM_min_center")
# UMM_min_edge = df_merged[edge_mask].groupby(result_keys)["Ct_Turb_UMM"].min().rename("Ct_Turb_UMM_min_edge")

# # Merge in the extra extrema
# df_stats = df_stats.merge(LES_max_left.reset_index(), on=result_keys)
# df_stats = df_stats.merge(LES_max_right.reset_index(), on=result_keys)
# df_stats = df_stats.merge(UMM_max_left.reset_index(), on=result_keys)
# df_stats = df_stats.merge(UMM_max_right.reset_index(), on=result_keys)
# df_stats = df_stats.merge(LES_min_center.reset_index(), on=result_keys)
# df_stats = df_stats.merge(LES_min_edge.reset_index(), on=result_keys)
# df_stats = df_stats.merge(UMM_min_center.reset_index(), on=result_keys)
# df_stats = df_stats.merge(UMM_min_edge.reset_index(), on=result_keys)

# %%
df_stats["Ct_mean_diff"] = df_stats["Ct_Turb_UMM_mean"] - df_stats["Ct_Turb_LES_mean"]
df_stats["Ct_min_diff"] = df_stats["Ct_Turb_UMM_min"] - df_stats["Ct_Turb_LES_min"]
df_stats["Ct_max_diff"] = df_stats["Ct_Turb_UMM_max"] - df_stats["Ct_Turb_LES_max"] 

# %%
# df_stats["Ct_min_diff_center"] = df_stats["Ct_Turb_UMM_min_center"] - df_stats["Ct_Turb_LES_min_center"]
# df_stats["Ct_min_diff_edge"] = df_stats["Ct_Turb_UMM_min_edge"] - df_stats["Ct_Turb_LES_min_edge"]
# df_stats["Ct_max_diff_right"] = df_stats["Ct_Turb_UMM_max_right"] - df_stats["Ct_Turb_LES_max_right"] 
# df_stats["Ct_max_diff_left"] = df_stats["Ct_Turb_UMM_max_left"] - df_stats["Ct_Turb_LES_max_left"] 

# %%
df_stats_surge = df_stats[df_stats["Movement"] == "Surge"]


# %%
def diff_grid_shaded(
    df,
    y_mean="Ct_diff_mean_norm",
    y_min="Ct_diff_min_norm",
    y_max="Ct_diff_max_norm"
):
    g = sns.FacetGrid(
        df,
        row="Frequency",
        col="Local Thrust Coefficient",
        hue = "Frequency",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.2
    )

    for (freq, ltc), ax in g.axes_dict.items():
        df_sub = df[
            (df["Frequency"] == freq) &
            (df["Local Thrust Coefficient"] == ltc)
        ].sort_values("Amplitude")

        # Mean line
        ax.plot(
            df_sub["Amplitude"],
            df_sub[y_mean],
            marker="o"
        )

        # Min–max shading
        ax.fill_between(
            df_sub["Amplitude"],
            df_sub[y_min],
            df_sub[y_max],
            alpha=0.3
        )

    g.set_axis_labels("Amplitude", y_mean)
    g.set_titles(
        row_template="f = {row_name} Hz",
        col_template=r"$\Delta C_T'$ = {col_name}"
    )

    plt.tight_layout()
    plt.show()


# %%
def diff_grid_shaded_freq(
    df,
    y_mean="Ct_diff_mean_norm",
    y_min="Ct_diff_min_norm",
    y_max="Ct_diff_max_norm"
):
    g = sns.FacetGrid(
        df,
        col="Local Thrust Coefficient",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.2,
        col_wrap=2
    )

    # Global frequency ordering
    freqs_all = np.sort(df["Frequency"].unique())
    freqs_plot = freqs_all[::-1]   # largest plotted first

    # Consistent colors across facets
    colors = sns.color_palette("tab10", len(freqs_all))
    color_map = dict(zip(freqs_all, colors))

    for ltc, ax in g.axes_dict.items():
        df_ltc = df[df["Local Thrust Coefficient"] == ltc]

        for freq in freqs_plot:
            df_sub = (
                df_ltc[df_ltc["Frequency"] == freq]
                .sort_values("Amplitude")
            )

            if df_sub.empty:
                continue

            # Fade higher frequencies
            alpha = 0.15 + 0.5 * (freq == freqs_all.min())

            ax.plot(
                df_sub["Amplitude"],
                df_sub[y_mean],
                color=color_map[freq],
                linestyle="--",
                linewidth=2,
                label=f"{freq:.2f} St"
            )

            ax.fill_between(
                df_sub["Amplitude"],
                df_sub[y_min],
                df_sub[y_max],
                color=color_map[freq],
                alpha=alpha * 0.7
            )

    # Axis + titles
    g.set_axis_labels("$A_S = \max{(U_{t} / U_\infty)}$", "$\Delta C_T = C_{T_{\\text{LES}}} -  C_{T_{\\text{UMM}}}$")
    g.set_titles(col_template=r"$C_T'$ = {col_name}")

    # Legend ordered from smallest to largest frequency
    handles, labels = ax.get_legend_handles_labels()
    handles_labels = sorted(
        zip(handles, labels),
        key=lambda hl: float(hl[1].split()[0])
    )
    handles, labels = zip(*handles_labels)

    g.fig.legend(
        handles,
        labels,
        title="Frequency",
        loc="upper center",
        ncol=len(labels),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()



# %%
def diff_grid_shaded_amp(
    df,
    y_mean="Ct_diff_mean_norm",
    y_min="Ct_diff_min_norm",
    y_max="Ct_diff_max_norm"
):
    g = sns.FacetGrid(
        df,
        col="Local Thrust Coefficient",
        margin_titles=True,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1.2,
        col_wrap=2
    )

    # Global frequency ordering
    amps_all = np.sort(df["Amplitude"].unique())
    amps_plot = amps_all[::-1]   # largest plotted first

    # Consistent colors across facets
    colors = sns.color_palette("tab10", len(amps_all))
    color_map = dict(zip(amps_all, colors))

    for ltc, ax in g.axes_dict.items():
        df_ltc = df[df["Local Thrust Coefficient"] == ltc]

        for amp in amps_plot:
            df_sub = (
                df_ltc[df_ltc["Amplitude"] == amp]
                .sort_values("Amplitude")
            )

            if df_sub.empty:
                continue

            # Fade higher frequencies
            alpha = 0.15 + 0.5 * (amp == amps_all.min())

            ax.plot(
                df_sub["Frequency"],
                df_sub[y_mean],
                color=color_map[amp],
                linestyle="--",
                linewidth=2,
                label=f"{amp:.2f}"
            )

            ax.fill_between(
                df_sub["Frequency"],
                df_sub[y_min],
                df_sub[y_max],
                color=color_map[amp],
                alpha=alpha * 0.7
            )

    # Axis + titles
    g.set_axis_labels("$St$", "$\Delta C_T = C_{T_{\\text{LES}}} -  C_{T_{\\text{UMM}}}$")
    g.set_titles(col_template=r"$C_T'$ = {col_name}")

    # Legend ordered from smallest to largest frequency
    handles, labels = ax.get_legend_handles_labels()
    handles_labels = sorted(
        zip(handles, labels),
        key=lambda hl: float(hl[1].split()[0])
    )
    handles, labels = zip(*handles_labels)

    g.fig.legend(
        handles,
        labels,
        title="$A_S = \max{(U_{t} / U_\infty)}$",
        loc="upper center",
        ncol=len(labels),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()



# %%
diff_grid_shaded_freq(df_stats_surge,
    y_mean="Ct_diff_mean",
    y_min="Ct_diff_min",
    y_max="Ct_diff_max"
)

# %%
diff_grid_shaded_amp(df_stats_surge,
    y_mean="Ct_diff_mean",
    y_min="Ct_diff_min",
    y_max="Ct_diff_max"
)


# %%
def diff_grid_mean_min_max(
    df,
    y_mean="Ct_diff_mean_norm",
    y_min="Ct_diff_min_norm",
    y_max="Ct_diff_max_norm",
    ylabel = "$\Delta C_T = C_{T_{\\text{UMM}}} -  C_{T_{\\text{LES}}}$",
    ylim = None,
):
    g = sns.FacetGrid(
        df,
        col="Local Thrust Coefficient",
        margin_titles=True,
        sharex=True,
        sharey=False,
        height=3,
        aspect=1.2,
        col_wrap=2
    )

    # Global frequency ordering
    freqs_all = np.sort(df["Frequency"].unique())
    freqs_plot = freqs_all[::-1]   # largest plotted first

    # Consistent colors across facets
    colors = sns.color_palette("tab10", len(freqs_all))
    color_map = dict(zip(freqs_all, colors))

    for ltc, ax in g.axes_dict.items():
        df_ltc = df[df["Local Thrust Coefficient"] == ltc]

        for freq in freqs_plot:
            df_sub = (
                df_ltc[df_ltc["Frequency"] == freq]
                .sort_values("Amplitude")
            )

            if df_sub.empty:
                continue

            # Fade higher frequencies
            alpha = 0.15 + 0.5 * (freq == freqs_all.min())

            ax.plot(
                df_sub["Amplitude"],
                df_sub[y_mean],
                color=color_map[freq],
                linestyle="solid",
                linewidth=2,
                label=f"{freq:.2f} St"
            )

            ax.plot(
                df_sub["Amplitude"],
                df_sub[y_min],
                color=color_map[freq],
                linestyle="dotted",
                linewidth=2,
                label=f"{freq:.2f} St"
            )

            ax.plot(
                df_sub["Amplitude"],
                df_sub[y_max],
                color=color_map[freq],
                linestyle="dashed",
                linewidth=2,
                label=f"{freq:.2f} St"
            )

 # Axis + titles
    g.set_axis_labels("$A_S = \max{(U_{t} / U_\infty)}$", ylabel)
    g.set_titles(col_template=r"$C_T'$ = {col_name}")

    # --- Create custom legend ---
    # 1. Color handles for frequency
    color_handles = [
        mlines.Line2D([], [], color=color_map[f], linestyle='solid', linewidth=3)
        for f in freqs_plot[::-1]  # show smallest first
    ]
    color_labels = [f"{f:.2f} St" for f in freqs_plot[::-1]]

    # 2. Line style handles for mean/min/max
    style_handles = [
        mlines.Line2D([], [], color="black", linestyle="solid", linewidth=2),
        mlines.Line2D([], [], color="black", linestyle="dashed", linewidth=2),
        mlines.Line2D([], [], color="black", linestyle="dotted", linewidth=2)
    ]
    style_labels = ["Mean", "Max", "Min"]

    # Place both legends
    g.fig.legend(
        color_handles,
        color_labels,
        title="Frequency",
        loc="upper center",
        ncol=3,
        frameon=False
    )

    g.fig.legend(
        style_handles,
        style_labels,
        title="Statistic",
        loc="upper right",
        frameon=False
    )
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout(rect=[0, 0, 1, 0.9])



# %%
diff_grid_mean_min_max(df_stats_surge,
    y_mean="Ct_mean_diff",
    y_min="Ct_min_diff",
    y_max="Ct_max_diff"
)

# %%
diff_grid_mean_min_max(df_stats_surge,
    y_mean="Ct_mean_diff",
    y_min="Ct_min_diff",
    y_max="Ct_max_diff"
)

# %%
diff_grid_mean_min_max(df_stats_surge,
    y_mean="Ct_Turb_UMM_mean",
    y_min="Ct_Turb_UMM_min",
    y_max="Ct_Turb_UMM_max",
    ylabel = "$C_{T_{\\text{UMM}}}$",
    ylim = (-0.5, 3.5),
)

# %%
diff_grid_mean_min_max(df_stats_surge,
    y_mean="Ct_Turb_LES_mean",
    y_min="Ct_Turb_LES_min",
    y_max="Ct_Turb_LES_max",
    ylabel = "$C_{T_{\\text{LES}}}$",
    ylim = (-0.5, 3.5),
)

# %%
# df = df_stats_surge
# g = sns.FacetGrid(
#     df,
#     col="Local Thrust Coefficient",
#     margin_titles=True,
#     sharex=True,
#     sharey=True,
#     height=3,
#     aspect=1.2,
#     col_wrap=2
# )

# # Global frequency ordering
# freqs_all = np.sort(df["Frequency"].unique())
# freqs_plot = freqs_all[::-1]   # largest plotted first

# # Consistent colors across facets
# colors = sns.color_palette("tab10", len(freqs_all))
# color_map = dict(zip(freqs_all, colors))

# for ltc, ax in g.axes_dict.items():
#     df_ltc = df[df["Local Thrust Coefficient"] == ltc]

#     for freq in freqs_plot:
#         df_sub = (
#             df_ltc[df_ltc["Frequency"] == freq]
#             .sort_values("Amplitude")
#         )

#         if df_sub.empty:
#             continue

#         # Fade higher frequencies
#         alpha = 0.15 + 0.5 * (freq == freqs_all.min())

#         ax.plot(
#             df_sub["Amplitude"],
#             df_sub["Ct_mean_diff"],
#             color=color_map[freq],
#             linestyle="solid",
#             linewidth=2,
#             label=f"{freq:.2f} St"
#         )

#         ax.scatter(
#             df_sub["Amplitude"],
#             df_sub["Ct_min_diff_center"],
#             color=color_map[freq],
#             # linestyle="dotted",
#             # linewidth=2,
#             label=f"{freq:.2f} St",
#             marker = "."
#         )
#         ax.scatter(
#             df_sub["Amplitude"],
#             df_sub["Ct_min_diff_edge"],
#             color=color_map[freq],
#             # linestyle="dotted",
#             # linewidth=2,
#             label=f"{freq:.2f} St",
#             marker = "x"
#         )

#         ax.scatter(
#             df_sub["Amplitude"],
#             df_sub["Ct_max_diff_right"],
#             color=color_map[freq],
#             # linestyle="dashed",
#             # linewidth=2,
#             label=f"{freq:.2f} St",
#             marker = "+"
#         )

#         ax.scatter(
#             df_sub["Amplitude"],
#             df_sub["Ct_max_diff_left"],
#             color=color_map[freq],
#             # linestyle="dashed",
#             # linewidth=2,
#             label=f"{freq:.2f} St",
#             marker = "2"
#         )

# # Axis + titles
# g.set_axis_labels("$A_S = \max{(U_{t} / U_\infty)}$", "$\Delta C_T = C_{T_{\\text{UMM}}} -  C_{T_{\\text{LES}}}$")
# g.set_titles(col_template=r"$C_T'$ = {col_name}")

# # --- Create custom legend ---
# # 1. Color handles for frequency
# color_handles = [
#     mlines.Line2D([], [], color=color_map[f], linestyle='solid', linewidth=3)
#     for f in freqs_plot[::-1]  # show smallest first
# ]
# color_labels = [f"{f:.2f} St" for f in freqs_plot[::-1]]

# # 2. Line style handles for mean/min/max
# style_handles = [
#     mlines.Line2D([], [], color="black", linestyle="solid", linewidth=2),
#     mlines.Line2D([], [], color="black", linestyle="dashed", linewidth=2),
#     mlines.Line2D([], [], color="black", linestyle="dotted", linewidth=2)
# ]
# style_labels = ["Mean", "Max", "Min"]

# # Place both legends
# g.fig.legend(
#     color_handles,
#     color_labels,
#     title="Frequency",
#     loc="upper center",
#     ncol=3,
#     frameon=False
# )

# g.fig.legend(
#     style_handles,
#     style_labels,
#     title="Statistic",
#     loc="upper right",
#     frameon=False
# )

# plt.tight_layout(rect=[0, 0, 1, 0.9])


# %%
df_both_les.keys()

# %%
sub = df_both_les[(df_both_les["Frequency"] == 0.4) & (df_both_les["Amplitude"] == 0.4) & (df_both_les["Local Thrust Coefficient"] == 1.33) & (df_both_les["PitchAmp"] == 8)]

# %%
plt.scatter(sub["Time"], sub["Ct_Turb"])
plt.xlim(200, 225)

# %%
df_both_les_avg = df_both_les.groupby(group_keys + ["PitchAmp"])[value_keys].mean().reset_index()
# df_both_les_sub = df_both_les_avg[cols_to_merge + ["PitchAmp"]]

# %%
df_both_les_sub = df_both_les_avg[df_both_les_avg["Frequency"] == 0.4]

# %%
df_both_les_sub

# %%
# Get unique facet values
freqs = sorted(df_both_les_sub["Frequency"].unique())
ltcs = sorted(df_both_les_sub["Local Thrust Coefficient"].unique())
amps = sorted(df_both_les_sub["Amplitude"].unique())
pitch_amps = sorted(df_both_les_sub["PitchAmp"].unique())

# Create FacetGrid by row=Frequency, col=LTC
g = sns.FacetGrid(
    df_both_les_sub,
    row="PitchAmp",
    col="Local Thrust Coefficient",
    margin_titles=True,
    sharex=True,
    sharey=True,
    height=3,
    aspect=1.2
)

# Consistent color map by amplitude
palette = sns.color_palette("tab10", n_colors=len(amps))
color_map = dict(zip(amps, palette))

# Loop over all facets
for (pamp, ltc), ax in g.axes_dict.items():
    df_les_sub = df_both_les_sub[
        (df_both_les_sub["PitchAmp"] == pamp) &
        (df_both_les_sub["Local Thrust Coefficient"] == ltc)
    ].sort_values("Phase_Rounded")
    print(df_les_sub)
    
    ltc_val = les_stationary_Cts[ltc]

    for amp in amps:
        # fixed bottom
        ax.hlines(y = ltc_val, xmin = 0, xmax = 1, label = "Fixed Bottom $C_T$", color = "k", linestyle = "--")
        # LES
        df_amp_les = df_both_les_sub[df_both_les_sub["Amplitude"] == amp]
        if not df_amp_les.empty:
            ax.scatter(
                df_amp_les["Phase_Rounded"],
                df_amp_les["Ct_Turb"],
                color=color_map[amp],
                alpha=0.7,
                zorder = 2
            )

g.set_axis_labels("Phase $\phi$", "$C_T$", size=12)
g.set_titles(col_template=r"$C_T'$ = {col_name}", row_template="$A_\phi = {row_name} ^\circ$", size=14)
for ax in g.axes.flatten():
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

# --- Create two-part legend ---
import matplotlib.lines as mlines

# 1) Colors for amplitudes
amp_handles = [
    mlines.Line2D([], [], color=color_map[amp], marker='o', linestyle='None', markersize=8)
    for amp in amps
]
amp_labels = [f"$A_S = {amp:.1f}$" for amp in amps]

# 2) Shapes/linestyles for datasets
dataset_handles = (
    [mlines.Line2D([], [], color='k',  markersize=8)
    for ds in ["LES",]] +
    [mlines.Line2D([], [], color='k', linestyle='--', markersize=8)]
)

dataset_labels = ["LES", "Fixed Bottom $C_T$"]


# Add both legends
leg1 = g.fig.legend(amp_handles, amp_labels, loc="upper center",
                    ncol=3, frameon=False, fontsize=12, title_fontsize=12)

leg2 = g.fig.legend(dataset_handles, dataset_labels, loc="upper center",
                    ncol=len(dataset_handles), frameon=False, fontsize=12, title_fontsize=12,
                    bbox_to_anchor=(0.5, 0.96))


# Make sure both are visible
g.fig.add_artist(leg1)

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()

# %%
