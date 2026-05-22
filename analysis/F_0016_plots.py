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
# # Analysis of Rotor Aerodynamics for Surging and Pitching Turbines

# %% [markdown]
# ## Variables and Helper Functions

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from UnifiedMomentumModel.Momentum import UnifiedMomentum
from UnifiedMomentumModel.Utilities.Geometry import calc_eff_yaw
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# %matplotlib inline
mpl.rcParams['figure.dpi'] = 100

# %% [markdown]
# Here I define the functions that I use to calculate the induction, thrust coefficent, and power:
#
# I defined induction with respect to the relative normal velocity at the disk ($u_{d, \text{rel}} = u_d - u_t$) and the relative inflow velocity normal to the disk ($U_{\infty, \text{rel}} = U_\infty - u_t$), where $u_t$ is the turbine surge velocity and $\phi$ is the instantaneous tilt of the turbine.
#
# $$
# a_n = 1 - \frac{u_{d, \text{rel}} \cdot \hat{n}}{U_{\infty, \text{rel}}\cos{\phi}}
# $$
#
# $$
# C_T' = \frac{T}{\frac{1}{2} \rho \pi A_d (u_{d, \text{rel}} \cdot \hat{n})^2}
# $$
#
# $$
# C_T = \frac{T}{\frac{1}{2}\rho \pi A_d U_\infty^2} = \frac{C_T' \cdot \frac{1}{2} \rho \pi A_d (u_{d, \text{rel}} \cdot \hat{n})^2}{\frac{1}{2}\rho \pi A_d U_\infty^2} = C_T' (\frac{u_{d, \text{rel}} \cdot \hat{n}}{U_\infty})^2
# $$
#
# $$
# C_P = \frac{P}{\frac{1}{2}\rho \pi A_d U_\infty^3}
# $$
#
# Note that for $a_n$, $U_{\infty, \text{rel}}$ is in the denominator, while for $C_T$ and $C_P$ just $U_\infty$ is in the denominator. These are just definitional decisions. For induction, I decided to define it with respect to the relative inflow that the turbine actually sees, while for $C_T$ and $C_P$ I defined them with respect to the freestream velocity. This is also why $C_T$ and $C_P$ don't have a factor of $\cos{\phi}$ in the denominator.

# %%
rho, uinf, D, dt = 1, 1, 1, 0.05

# calculate needed values
def calc_an(df, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb"): # calcualte an with respect to relative-inflow
    u_inf = df[uinf_key]* np.cos(np.deg2rad(df["Tilt"])) # TODO: add in yaw
    return 1 - (df[ud_key] / u_inf)

def calc_ct(df, ud_key = "UDisk_Turb", uinf_key = "UInf_Ground"): # calcualte CT with respect to freestream inflow
    u_inf = df[uinf_key]
    return np.sign(df[ud_key]) * df["Ct_prime"] * ((df[ud_key])**2 / (u_inf)**2)

def calc_cp(df, uinf_key = "UInf_Ground"): # calcualte CT with respect to freestream inflow
    u_inf = df[uinf_key]
    return df.Power / (0.5 * rho * math.pi * (D/2)**2 * (u_inf)**3)


# %% [markdown]
# I also then have a helper function to determine bin the phase of the LES data, given a user-defined bin width.

# %%
# def round_phases(phase, bin_width=0.05):
#     phase = np.mod(phase, 1.0)          # enforce circularity
#     phase_idx = np.floor(phase / bin_width).astype(int)
#     phase_bin = (phase_idx + 0.5) * bin_width
#     return phase_bin

# %% [markdown]
# Finally, I define all amplitudes and frequencies that were run in LES and thus need to be run in UMM to compare. I have been considering the following values
#
# - $C_T'$ values: $1.33$, $1.66$, $2.00$, and $2.33$
# - Non-dimensional frequencies / Strouhal numers: $0.0 - 1.2$, with a spacing of $0.1$
# - Non-dimensional surge amplitudes ($A_S$): $0.0 - 1.2$, with a spacing of $0.1$
# - Pitch amplitudes ($A_P$) in degrees: $0.0 - 20^\circ$, with a spacing of $4^\circ$

# %%
ct_vals = [1.33, 1.66, 2.00, 2.33]
f_vals  = [round(0.1 * i, 1) for i in range(13)]
As_vals = [round(0.1 * i, 1) for i in range(13)]
Ap_vals = [0, 4, 8, 12, 16, 20]


# %% [markdown]
# # Get LES Data
#
# ## Filter LES Data
#
# The first steps here are to clean the LES data. I have several runs with different timesteps / filter widths. I select the ones that I want with the `keep_sim` function.

# %%
def keep_smallest_dt(df, sim_keys):
    """
    Keep only the rows corresponding to the minimum dt for each simulation (sim_keys).
    Assumes dt varies within a simulation.
    """
    # 1) Find the minimum dt per simulation
    min_dt_per_sim = df.groupby(sim_keys)["dt"].transform("min")
    
    # 2) Keep only the rows that correspond to the min dt
    return df[df["dt"] == min_dt_per_sim].reset_index(drop=True)



# %% [markdown]
# Within the `get_clean_les_data`, I remove any `NaN` data, rename key columns, and calculate key quantities. I also calcualte the mean induction, thrust coeffienct, and power coeffient for the stationary runs, which can be used to normalize various quantities.

# %%
sim_keys = ["Movement", "Frequency", "Amplitude", "Ct_prime"]
run_keys = ["dt"]


# %%
def get_clean_les_data(file, sim_keys, phase_decimals = 3):
    # read in LES data, remove "bad" data, and rename key columns
    df_les = pd.read_csv(file)
    df_les = df_les.dropna()
    df_les["Model"] = "LES"
    df_les = df_les.rename(columns={'UDisk': 'UDisk_Turb', 'Thrust Coefficient': "Ct_prime"}) # disk velocity in the turbine frame of reference

    # calculate needed LES quantities per datapoint/timestep
    df_les["UDisk_Ground"] = df_les["UDisk_Turb"] + df_les["UTurb"] # disk velocity in the ground frame of reference
    df_les["UInf_Turb"] = (uinf - df_les["UTurb"])
    df_les["UInf_Ground"] = uinf

    df_les["an_Turb"] = calc_an(df_les)
    df_les["Ct_Turb"] = calc_ct(df_les)
    df_les["Cp_Turb"] = calc_cp(df_les)

    # grabbing the stationary run values
    les_stationary_vals = dict()
    les_stationary_runs = df_les[(df_les["Frequency"] == 0) & (df_les["Amplitude"] == 0)]
    if len(les_stationary_runs) > 0:
        les_stationary_ans = les_stationary_runs.groupby(sim_keys + run_keys)["an_Turb"].mean().reset_index()
        les_stationary_cts = les_stationary_runs.groupby(sim_keys + run_keys)["Ct_Turb"].mean().reset_index()
        les_stationary_cps = les_stationary_runs.groupby(sim_keys + run_keys)["Cp_Turb"].mean().reset_index()
        les_stationary_vals = dict()
        for ctp in ct_vals:
            les_stationary_vals[ctp] = {
                "an_Turb": les_stationary_ans[les_stationary_ans["Ct_prime"] == ctp]["an_Turb"].iloc[0],
                "Ct_Turb": les_stationary_cts[les_stationary_cts["Ct_prime"] == ctp]["Ct_Turb"].iloc[0],
                "Cp_Turb": les_stationary_cps[les_stationary_cps["Ct_prime"] == ctp]["Cp_Turb"].iloc[0],
            }

    # remove any cases where the CT values are very unstable
    df_les = (
        df_les
        .groupby(sim_keys + run_keys)
        .filter(lambda g: (g["Ct_Turb"].max() <= 100))
    )
    # remove any cases that don't meet standards in `keep_sim`
    df_les = keep_smallest_dt(df_les, sim_keys)
    # add in phase and rounded phase 
    df_les["Phase"] = (df_les["Time"] * df_les["Frequency"]) % 1.0
    df_les["Phase_Rounded"] = df_les["Phase"].round(phase_decimals)
    return df_les, les_stationary_vals


# %%
phase_decimals = 2
df_les_16, les_stationary_vals = get_clean_les_data("/Users/sky/src/HowlandLab/data/sim_0016_all_02_09_26.csv", sim_keys, phase_decimals = phase_decimals)
df_les_23, _ = get_clean_les_data("/Users/sky/src/HowlandLab/data/sim_0023_all_02_09_26.csv", sim_keys, phase_decimals = phase_decimals)
df_les = pd.concat([df_les_16, df_les_23])

# %%
sim_keys = sim_keys + ["Phase_Rounded"]


# %% [markdown]
# We now get the phases, tilts, and turbine velocities represented in the LES runs. We can repeate these unique pairs with the UMM below.

# %%
def collect_unique_sim_values(df, sim_keys):
    # 1) Phase-conditioned statistics
    phase_stats = (
        df
        .groupby(sim_keys, as_index=False)
        .agg(
            Tilt_mean=("Tilt", "mean"),
            Tilt_std=("Tilt", "std"),
            UTurb_mean=("UTurb", "mean"),
            UTurb_std=("UTurb", "std"),
        )
        .sort_values("Phase_Rounded")
    )
    return phase_stats

# Example usage:
df_umm = collect_unique_sim_values(df_les, sim_keys + run_keys)


# %%
def plot_phase_conditioned_grid(df, yvar="UTurb", movement="Surge", has_mean_std = True):
    # filter by movement
    df = df[df["Movement"] == movement]

    freqs = np.sort(df["Frequency"].unique())
    cts   = np.sort(df["Ct_prime"].unique())
    amps  = np.sort(df["Amplitude"].unique())

    nrow, ncol = len(freqs), len(cts)

    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(4*ncol, 3*nrow),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    # ------------------------------------------------------------------
    # DISCRETE color mapping for Amplitude
    # ------------------------------------------------------------------
    base_cmap = cm.tab10  # up to 10 discrete colors
    colors = base_cmap.colors[:len(amps)]
    amp_to_color = dict(zip(amps, colors))

    # ------------------------------------------------------------------
    # Loop over frequency/CT grid
    # ------------------------------------------------------------------
    for i, f in enumerate(freqs):
        for j, ct in enumerate(cts):
            ax = axes[i, j]

            sub = df[(df["Frequency"] == f) & (df["Ct_prime"] == ct)]

            for _, row in sub.iterrows():
                x = row["Phase_Rounded"]
                amp = row["Amplitude"]
                color = amp_to_color[amp]

                if has_mean_std:
                    y = row[f"{yvar}_mean"]
                    yerr = row[f"{yvar}_std"]
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt='o',
                        markersize=4,
                        markerfacecolor='none',
                        markeredgecolor=color,
                        ecolor=color,
                        alpha=0.9,
                        capsize=2,
                        linestyle='none'
                    )
                else:
                    y =  row[f"{yvar}"]
                    ax.scatter(
                        x,
                        y,
                        s=6,
                        color=color,
                        alpha=0.9,
                    )

            if i == 0:
                ax.set_title(rf"$C_T'={ct}$")
            if j == 0:
                ax.set_ylabel(f"{yvar}\n$f={f}$")
            if i == nrow - 1:
                ax.set_xlabel("Phase")

    # ------------------------------------------------------------------
    # Discrete legend for Amplitude
    # ------------------------------------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='none',
            markeredgecolor=amp_to_color[a],
            markersize=6,
            label=f"A = {a}",
        )
        for a in amps
    ]

    fig.legend(
        handles=legend_handles,
        title="Amplitude",
        loc="upper right",
        bbox_to_anchor=(1.02, 0.98),
        frameon=True,
    )

    fig.suptitle(f"{yvar} vs Phase — {movement}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])


# %%
sub = df_umm[(df_umm["Movement"] == "Surge") & (df_umm["Frequency"] == 0.8)]
plot_phase_conditioned_grid(
    sub,
    yvar="UTurb",
    movement="Surge"
)

# %%
sub = df_umm[(df_umm["Movement"] == "Pitch") & (df_umm["Frequency"] == 0.4)]
plot_phase_conditioned_grid(
    sub,
    yvar="Tilt",
    movement="Pitch"
)


# %% [markdown]
# # Get UMM Data

# %% [markdown]
# I then needed to get the UMM values to compare to the equivalent LES simulations. Given that the LES simulations have _many_ timesteps and hundreds of periods, I did not run the UMM for each of the corresponding timesteps. Instead I choose to run it for just a few periods, but I also in-built the functionality to have a fixed number of points per period (i.e. n = 100), regardless of frequency. I compare these techniques below.
#
# The sinusoidal turbine motion (either pitch in degrees of surge in non-dimensional distance) is
#
# $$
# A\sin(2 \pi f t) = A\sin(2 \pi \frac{t}{T}) = A\sin(2 \pi \psi)
# $$
#
# where $f = \frac{1}{T}$ is the frequency and $\psi \in [0, 1)$ is the phase. To compare with LES, I will bin the values over the phases $\psi$ or compare period-averaged values.
#
# Note that to calculate the induction, I just take the value straight from the UMM. The UMM already accounts for pitch and yaw, and induction is independent of inflow speed. Since surge is equivalent to a uniform rescaling of the relative velocity, it does not effect the induction under the quasi-steady assumption.
#
# On the other hand, for $C_T$ and $C_P$ I multiply the UMM returned value by the relative velocity squared or cubed, respectively. I do this because the definitions I choose for the induction, thrust coefficent, and power coefficent. This is discussed above, but to summarize, induction is defined with respect to the relative inflow that the turbine sees, while for $C_T$ and $C_P$ are defined with respect to the freestream velocity.
#
# Thus, if we re-dimensionalize the thrust in the no-surge case, for example, $T = C_T \cdot \frac{1}{2}\rho A_d U_\infty^2$. However, in the surge case, the turbine actually "sees" $U_{\infty, \text{rel}} = U_\infty - U_\text{turb}$. Note that thrust scales with the velocity into the turbine rotor.
#
# Thus: $T_\text{surge} = C_T \cdot \frac{1}{2}\rho A_d U_{\infty, \text{rel}}^2$ and $C_{T_{\text{surge}}} = \frac{T_\text{surge}}{\frac{1}{2}\rho A_d U_\infty^2} = \frac{C_T \cdot \frac{1}{2}\rho A_d U_{\infty, \text{rel}}^2}{\frac{1}{2}\rho A_d U_\infty^2} = C_T U_{\infty, \text{rel}}^2$ since $U_\infty = 1$. A similar logic can be repeated for power.

# %%
def get_umm_solution(model, cache, ct, tilt):
    key = (ct, tilt)
    if key not in cache:
        sol = model(ct, yaw=0.0, tilt=tilt)
        cache[key] = {
            "an_Turb": sol.an,
            "Ct_Turb": sol.Ct,
            "Cp_Turb": sol.Cp,
        }
    return cache[key]


# %%
def get_umm_data(df):
    model = UnifiedMomentum()
    cache = {}
    an, Ct, Cp = [], [], []
    for row in df.itertuples(index=False):
        Ct_prime = row.Ct_prime
        tilt = row.Tilt_mean
        uinf_t = 1 - row.UTurb_mean
        sol = get_umm_solution(model, cache, Ct_prime, tilt)
        an.append(sol["an_Turb"])
        Ct.append(np.sign(uinf_t) * sol["Ct_Turb"] * uinf_t**2)
        Cp.append(sol["Cp_Turb"] * uinf_t**3)

    df = df.copy()
    df["an_Turb"] = an
    df["Ct_Turb"] = Ct
    df["Cp_Turb"] = Cp
    return df


# %%
df_umm = get_umm_data(df_umm)

# %%
sub = df_umm[(df_umm["Movement"] == "Surge") & (df_umm["Frequency"] == 0.8)]
plot_phase_conditioned_grid(
    sub,
    yvar="Ct_Turb",
    movement="Surge",
    has_mean_std = False
)

# %%
sub = df_umm[(df_umm["Movement"] == "Pitch") & (df_umm["Frequency"] == 0.4)]
plot_phase_conditioned_grid(
    sub,
    yvar="Ct_Turb",
    movement="Pitch",
    has_mean_std = False
)

# %%
df_les.keys(), df_umm.keys()

# %%
join_keys = [
    "Movement",
    "Frequency",
    "Amplitude",
    "Ct_prime",
    "Phase_Rounded",
]

df_sims = df_les.merge(
    df_umm[
        join_keys + ["an_Turb", "Ct_Turb", "Cp_Turb"] + ['Tilt_mean', 'Tilt_std', 'UTurb_mean', 'UTurb_std']
    ],
    on=join_keys,
    how="left",
    suffixes=("_LES", "_UMM"),
)

df_sims["an_Turb_diff"] = df_sims["an_Turb_LES"] - df_sims["an_Turb_UMM"]
df_sims["Ct_Turb_diff"] = df_sims["Ct_Turb_LES"] - df_sims["Ct_Turb_UMM"]
df_sims["Cp_Turb_diff"] = df_sims["Cp_Turb_LES"] - df_sims["Cp_Turb_UMM"]

# %%
col_names = ["Ct_prime", "Amplitude", "Pitch_Amp", "Frequency", "Phase", "Time", "Distance", "Tilt", "UTurb", "an_Turb", "Ct_Turb", "Cp_Turb"]

def get_phases(*, n_phase=None, f=None, dt=None, N_periods=5):
    """
    Returns phase in [0, 1) over N_periods.
    Exactly one of:
      - n_phase
      - (f and dt)
    must be provided.
    """
    if n_phase is not None:
        base = np.linspace(0, 1, n_phase, endpoint=False)
        phases = np.tile(base, N_periods)
        t = range(len(phases)) # note that this isn't a real time and is just used for later sorting
        return phases, t

    if f is not None and dt is not None:
        if f == 0:
            # stationary: single phase repeated
            return np.zeros(N_periods), np.zeros(N_periods)
        T = 1.0 / f
        t = np.arange(0, N_periods * T, dt)
        phases = np.mod(f * t, 1.0)
        return phases, t

    raise ValueError("Provide either n_phase or (f and dt)")

def get_umm_solution(model, cache, ct, tilt):
    key = (ct, tilt)
    if key not in cache:
        sol = model(ct, yaw=0.0, tilt=tilt)
        cache[key] = {
            "an_Turb": sol.an,
            "Ct_Turb": sol.Ct,
            "Cp_Turb": sol.Cp,
        }
    return cache[key]

def get_umm_vals(ct_vals, f_vals, As_vals, Ap_vals, *, n_phase=None, dt=None, N_periods=5): # TODO: update this to use df_umm
    # check for non-zero surge values
    pitching = False
    if np.all(np.array(As_vals) == 0):
        pitching = True
    # insantiate needed model/cache
    model = UnifiedMomentum()
    cache = {}
    umm_vals = []
    for ct in ct_vals:
        for f in f_vals:
            # get phases and calculate needed phase values
            phases, times = get_phases(n_phase=n_phase, f=f if dt is not None else None, dt=dt, N_periods=N_periods)
            phi = 2 * np.pi * phases
            sinphi = np.sin(phi)
            cosphi = np.cos(phi)
            omega = 2 * np.pi * f
            # loop through each phase
            for i, phase in enumerate(phases):
                for a_p in Ap_vals:
                    # solve UMM for pitch
                    tilt = np.deg2rad(a_p * sinphi[i])
                    sol = get_umm_solution(model, cache, ct, tilt)
                    an_val = sol["an_Turb"]
                    ct0 = sol["Ct_Turb"]
                    cp0 = sol["Cp_Turb"]
                    # for each amplitude, update UMM results by relative velocity
                    for a_s in As_vals:
                        ut = a_s * cosphi[i]
                        uinf_t = 1.0 - ut
                        Ct_val = np.sign(uinf_t) * ct0 * uinf_t**2
                        Cp_val = cp0 * uinf_t**3
                        dist = (a_s / omega) * sinphi[i] if (omega != 0) else 0.0
                        amp = a_p if pitching else a_s
                        # assemble output tuple here
                        # ("Ct_prime", "Amplitude", "Pitch_Amp", "Frequency",  "Phase", "Time, "Distance", "Tilt", "UTurb", "an_Turb", "Ct_Turb", "Cp_Turb")
                        umm_vals.append((ct, amp, a_p, f, phase, times[i], dist, tilt, ut, an_val, Ct_val, Cp_val))
    # combine into a dataframe and add a few more columns!
    df_umm = pd.DataFrame(np.array(umm_vals), columns = col_names)
    df_umm["UInf_Turb"] = (uinf - df_umm["UTurb"])
    df_umm["UDisk_Turb"] = (1 - df_umm["an_Turb"]) * df_umm["UInf_Turb"] * np.cos(calc_eff_yaw(yaw = 0.0, tilt = df_umm["Tilt"])) # TODO: update with yaw and static tilt!
    df_umm["Phase_Rounded"] = round_phases(df_umm["Phase"])
    df_umm["Model"] = "UMM"

    # get stationary umm values
    umm_stationary_vals = dict()
    for ct in ct_vals:
        sol = get_umm_solution(model, {}, ct, tilt = 0.0)
        umm_stationary_vals[ct] = sol
    return df_umm, umm_stationary_vals



# %% [markdown]
# I then ran the UMM for the surging and pitching cases, as well as the cases where it is both surging and pitching. 
#
# Eventually, I will add in the cases where we have surging and pitching, as well as static tilt and yaw here as well.

# %%
df_umm_pitch, umm_stationary_vals = get_umm_vals(ct_vals, f_vals, [0], Ap_vals, dt = 0.05, N_periods=10)
df_umm_pitch["Movement"] = "Pitch"

df_umm_surge, _ = get_umm_vals(ct_vals, f_vals, As_vals, [0.0], dt = 0.05, N_periods=10)
df_umm_surge["Movement"] = "Surge"

# %% [markdown]
# We can then combine these two datasets. 
#
# Note here that we calculate the disk velocity from the induction. We first calculate the relative freesteam with the inclusion of the turbine movement. From that, and the definitions above, we can calculate the relative velocity normal to the disk.

# %%
# create combined pitch + surge dataframe
df_umm = pd.concat([df_umm_surge, df_umm_pitch], ignore_index=True)

# %%
sub_les = df_les[(df_les["Ct_prime"] == 1.33) & (df_les["Frequency"] == 0.2) & (df_les["Amplitude"] == 0.2)]
sub_umm = df_umm[(df_umm["Ct_prime"] == 1.33) & (df_umm["Frequency"] == 0.2) & (df_umm["Amplitude"] == 0.2)]
sns.scatterplot(data = sub_les, x = "Phase", y = "Cp_Turb", color = "tab:orange")
sns.scatterplot(data = sub_umm, x = "Phase", y = "Cp_Turb", color = "tab:blue")
len(np.unique(np.round(sub_les["Phase"], decimals = 4))), len(sub_umm["Phase"])

# %% [markdown]
# ## Initial LES Analysis (not phase-binned)
#
# My idea here is that the UMM generated data has a very standardized shape. The maximum $C_T$ value is at $\phi = 0.5$, the minimum value is at $\phi = 0.0$, and the values cross over the mean $C_T$ value at $\phi = 0.25$ and $\phi = 0.75$.
#
# This isn't neccesarily true for the LES data. Therefore, I want to record the mean and max $C_T$ value for each period within a simulation and the phase that the min and max occur at. This should then be averaged over all periods within a simulation. I also want to record the phase as which the $C_T$ values within a simulation go from under the mean of the entire simulation to over the mean (around 0.25) and back from over the mean to under the mean (around 0.75). 

# %%
normalize = False
n_bins = 10


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

def circular_std(phase):
    phase = np.asarray(phase)
    phase = phase[~np.isnan(phase)]

    if phase.size < 2:
        return np.nan

    theta = 2 * np.pi * phase
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C**2 + S**2)

    if R <= 0 or R > 1:
        return np.nan

    return np.sqrt(-2 * np.log(R)) / (2 * np.pi)


def circ_diff(a, b):
    """Compute minimal signed difference between circular phases [0,1)."""
    delta = a - b
    # wrap to [-0.5, 0.5)
    delta = (delta + 0.5) % 1.0 - 0.5
    return delta

def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)

def safe_std(x):
    return np.nan if len(x) < 2 else np.std(x, ddof=1)


# %%
def period_avg_stats(
    g,
    val_key,
    stationary_dict,
    phase_key="Phase",
    normalize=False
):
    g = g.reset_index(drop=True)
    g = add_period_index(g)

    # Determine stationary values for current CT'
    local_stationary_ctp = g["Ct_prime"].iloc[0]
    stationary_vals = stationary_dict[local_stationary_ctp][val_key]

    # Remove last period if incomplete
    period_counts = g["Period"].value_counts()
    last_period = g["Period"].max()
    full_count = period_counts.max()
    if period_counts[last_period] < full_count:
        g = g[g["Period"] != last_period]

    # Per-period storage
    mean_vals = []
    max_vals, max_phases = [], []
    min_vals, min_phases = [], []

    # Per-period phase differences
    extrema_phase_diffs = []
    crossover_phase_diffs = []
    peak_skew_diff = []

    # Loop through each period
    for _, p in g.groupby("Period"):
        phase = p[phase_key].values
        vals = p[val_key].values

        if normalize:
            vals = vals / stationary_vals

        # Mean CT
        mean_val = np.mean(vals)
        mean_vals.append(mean_val)

        # Max CT
        i_max = np.argmax(vals)
        phi_max = phase[i_max]
        max_vals.append(vals[i_max])
        max_phases.append(phi_max)

        # Min CT
        i_min = np.argmin(vals)
        phi_min = phase[i_min]
        min_vals.append(vals[i_min])
        min_phases.append(phi_min)

        # Phase difference between extrema (circular)
        extrema_phase_diffs.append(circ_diff(phi_max, phi_min))

        # Mean crossings
        up, down = strict_mean_crossings(phase, vals, mean_val)
        if not np.isnan(up) and not np.isnan(down):
            # Phase difference between crossings and maximum
            d_up = circ_diff(phi_max, up)
            d_down = circ_diff(down, phi_max)
            peak_skew_diff.append(d_up - d_down)

    return pd.Series({
        # Linear CT statistics
        "avg_mean": np.mean(mean_vals),
        "std_mean": np.std(mean_vals, ddof=1),

        "avg_max": np.mean(max_vals),
        "std_max": np.std(max_vals, ddof=1),

        "avg_min": np.mean(min_vals),
        "std_min": np.std(min_vals, ddof=1),

        # Circular phase statistics
        "avg_phase_max": circular_mean(max_phases),
        "std_phase_max": circular_std(max_phases),

        "avg_phase_min": circular_mean(min_phases),
        "std_phase_min": circular_std(min_phases),

        "avg_phase_extrema_diff": circular_mean(extrema_phase_diffs),
        "std_phase_extrema_diff": circular_std(extrema_phase_diffs),

        "avg_peak_skew_diff": circular_mean(peak_skew_diff),
        "std_peak_skew_diff": circular_std(peak_skew_diff),

        # Bookkeeping
        "n_periods": g["Period"].nunique(),
        "n_valid_crossings": len(crossover_phase_diffs),
    })


# %%
def phase_avg_stats(
    g,
    val_key,
    stationary_dict,
    phase_key="Phase",
    normalize=False,
    n_bins=10,
):
    import numpy as np
    import pandas as pd

    # Reset index
    g = g.reset_index(drop=True)

    # Add period index
    g = add_period_index(g)

    # Stationary normalization
    local_ct = g["Ct_prime"].iloc[0]
    stationary_val = stationary_dict[local_ct][val_key]

    # Remove incomplete last period
    period_counts = g["Period"].value_counts()
    last_period = g["Period"].max()
    full_count = period_counts.max()
    if period_counts[last_period] < full_count:
        g = g[g["Period"] != last_period]

    # Extract values
    vals = g[val_key].values
    phase = g[phase_key].values

    if normalize:
        vals = vals / stationary_val

    # Bin phases (phase already 0–1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(phase, bins, right=False) - 1
    bin_idx[bin_idx == n_bins] = n_bins - 1  # clip exact 1.0 values

    # Create DataFrame for aggregation
    df = pd.DataFrame({"bin": bin_idx, "val": vals})

    # Compute mean, std, count per phase bin
    df = (
        df.groupby("bin", group_keys=False)
        .agg(
            mean=("val", safe_mean),
            std=("val", safe_std),
            n=("val", "count"),
        )
        .reset_index()
    )

    # Assign phase centers and number of periods
    df["phase"] = bin_centers[df["bin"].values]
    df["n_periods"] = g["Period"].nunique()

    # Attach all group columns automatically
    sim_keys = ["Movement", "Frequency", "Amplitude", "Ct_prime"]
    for col in sim_keys:
        if col in g.columns:
            df[col] = g[col].iloc[0]

    # Drop the 'bin' column and reorder nicely
    df = df[["phase", "mean", "std", "n", "n_periods"] + sim_keys]

    # Clean any leftover level_0 / level_1 columns
    df = df.loc[:, [c for c in df.columns if not c.startswith("level_")]]

    return df



# %%
def remove_high_variation_sims(df, qc_metrics, err, eps=1e-12,
):
    """
    Remove rows where selected QC metrics have high relative variation.
    """
    mask = np.ones(len(df), dtype=bool)

    for mean_col, std_col in qc_metrics:
        rel_var = np.abs(df[std_col] / (np.abs(df[mean_col]) + eps))
        mask &= rel_var < err

    return df[mask]


# %%
period_avg_metrics = [
    ("avg_mean", "std_mean"),
    ("avg_max", "std_max"),
    ("avg_min", "std_min"),
    ("avg_phase_max", "std_phase_max"),
    ("avg_phase_min", "std_phase_min"),
    ("avg_phase_extrema_diff", "std_phase_extrema_diff"),
    ("avg_peak_skew_diff", "std_peak_skew_diff")
]

phase_avg_metrics = [
    ("mean", "std")
]


# %%
def get_les_umm_summary_data(df_les, df_umm, les_stationary_vals, umm_stationary_vals, metrics, qc_metrics, reducer, reducer_kwargs = None,
                            val_key = "Ct_Turb", movement_key = "Surge", phase_key = "Phase", normalize = False, err = 0.05,
):
    # setup to be able to get both LES and UMM values with same code!
    reducer_kwargs = reducer_kwargs or {}
    def run(df, stationary_vals):
        return (
            df[df["Movement"] == movement_key]
            .groupby(sim_keys, as_index=False)
            .apply(
                reducer,
                val_key,
                stationary_vals,
                phase_key=phase_key,
                normalize=normalize,
                **reducer_kwargs,
                include_groups=True,
            )
            .reset_index(drop=True)
        )

    les_sim_summary = run(df_les, les_stationary_vals)
    umm_sim_summary = run(df_umm, umm_stationary_vals)
    
    # les_sim_summary = remove_high_variation_sims(les_sim_summary, qc_metrics, err)
    # umm_sim_summary = remove_high_variation_sims(umm_sim_summary, qc_metrics, err)

    merged = les_sim_summary.merge(
        umm_sim_summary,
        on=sim_keys,
        how="left",
        suffixes=("_LES", "_UMM"),
    )
    diff_sim_summary = merged[sim_keys].copy()

    for mean_col, std_col in metrics:
        if "phase" in mean_col.lower():
            diff_sim_summary[f"{mean_col}"] = circ_diff(
                merged[f"{mean_col}_UMM"],
                merged[f"{mean_col}_LES"],
            )
        else:
            diff_sim_summary[f"{mean_col}"] = (
                merged[f"{mean_col}_UMM"] - merged[f"{mean_col}_LES"]
            )

        diff_sim_summary[f"{std_col}"] = np.sqrt(
            merged[f"{std_col}_LES"]**2 + merged[f"{std_col}_UMM"]**2
        )

    return diff_sim_summary, les_sim_summary, umm_sim_summary


# %%
period_diff_ct_surge_sim_summary, period_les_ct_surge_sim_summary, period_umm_ct_surge_sim_summary = get_les_umm_summary_data(df_les, df_umm, les_stationary_vals, umm_stationary_vals,
                                                                                                period_avg_metrics, period_avg_metrics[:2], reducer=period_avg_stats,
                                                                                                val_key = "Ct_Turb", movement_key = "Surge", phase_key = "Phase",
                                                                                                normalize = normalize, err = 0.05)

# %%
# phase_diff_ct_surge_sim_summary, phase_les_ct_surge_sim_summary, phase_umm_ct_surge_sim_summary = get_les_umm_summary_data(df_les, df_umm, les_stationary_vals, umm_stationary_vals,
#                                                                                                 phase_avg_metrics, phase_avg_metrics, reducer=phase_avg_stats, reducer_kwargs={"n_bins": n_bins},
#                                                                                                 val_key = "Ct_Turb", movement_key = "Surge", phase_key = "Phase",
#                                                                                                 normalize = normalize, err = 0.05)

# %%
# Templates with escaped braces for str.format()
ylabel_templates_raw = {
    "avg_mean": r"$\overline{{{val}}}$",
    "avg_max": r"$\overline{{\max({val})}}$",
    "avg_min": r"$\overline{{\min({val})}}$",

    "avg_phase_max": r"$\overline{{\phi_{{\max({val})}}}}$",
    "avg_phase_min": r"$\overline{{\phi_{{\min({val})}}}}$",

    "avg_phase_under_to_over_mean": r"$\overline{{\phi_{{{val} \uparrow {val}_{{\mathrm{{FB}}}}}}}}$",
    "avg_phase_over_to_under_mean":  r"$\overline{{\phi_{{{val} \downarrow {val}_{{\mathrm{{FB}}}}}}}}$",

    "avg_phase_extrema_diff": r"$\overline{{\phi_{{\mathrm{{max-min}}}}}}$",
    "avg_phase_crossover_diff": r"$\overline{{\phi_{{\uparrow - \downarrow}}}}$",
    "avg_phase_up_to_max_diff": r"$\overline{{\phi_{{\uparrow - max}}}}$",
    "avg_phase_max_to_down_diff": r"$\overline{{\phi_{{max - \downarrow}}}}$",
    "avg_peak_skew_diff": r"$\overline{{\Delta \phi_{{\uparrow\to\max}} - \Delta \phi_{{\max\to\downarrow}}}}$"
}

ylabel_templates_normalized = {
    "avg_mean": r"$\overline{{{val} / {val}_{{\mathrm{{FB}}}}}}$",
    "avg_max": r"$\overline{{\max\!\left({val} / {val}_{{\mathrm{{FB}}}}\right)}}$",
    "avg_min": r"$\overline{{\min\!\left({val} / {val}_{{\mathrm{{FB}}}}\right)}}$",

    "avg_phase_max": r"$\overline{{\phi_{{\max({val})}}}}$",
    "avg_phase_min": r"$\overline{{\phi_{{\min({val})}}}}$",

    "avg_phase_under_to_over_mean": r"$\overline{{\phi_{{{val} \uparrow {val}_{{\mathrm{{FB}}}}}}}}$",
    "avg_phase_over_to_under_mean": r"$\overline{{\phi_{{{val} \downarrow {val}_{{\mathrm{{FB}}}}}}}}$",

    "avg_phase_extrema_diff": r"$\overline{{\phi_{{\mathrm{{max-min}}}}}}$",
    "avg_phase_crossover_diff": r"$\overline{{\phi_{{\uparrow - \downarrow}}}}$",
    "avg_phase_up_to_max_diff": r"$\overline{{\phi_{{\uparrow - max}}}}$",
    "avg_phase_max_to_down_diff": r"$\overline{{\phi_{{max - \downarrow}}}}$",
    "avg_peak_skew_diff": r"$\overline{{\Delta \phi_{{\uparrow\to\max}} - \Delta \phi_{{\max\to\downarrow}}}}$"
}

def get_ylabel(metric, val="C_T", normalize=False):
    templates = ylabel_templates_normalized if normalize else ylabel_templates_raw
    if metric not in templates:
        raise KeyError(f"Metric '{metric}' not found in label templates")
    return templates[metric].format(val=val)



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
        col="Ct_prime",
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
        df_ltc = df[df["Ct_prime"] == ltc]

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
for metric in period_avg_metrics:
    y_key, y_err_key = metric

    y_key = y_key
    y_err_key = y_err_key
    y_label = "UMM: " + get_ylabel(y_key, normalize=normalize)

    # umm_les_data_plot(umm_surge_sim_summary, y_key, y_err_key, y_label)


# %%
for metric in period_avg_metrics:
    y_key, y_err_key = metric

    y_key = y_key
    y_err_key = y_err_key
    y_label = "LES: " + get_ylabel(y_key, normalize=normalize)

    # umm_les_data_plot(les_surge_sim_summary, y_key, y_err_key, y_label)


# %%
for metric in period_avg_metrics:
    y_key, y_err_key = metric

    y_label = "$\Delta$" + get_ylabel(y_key, normalize=normalize)

    umm_les_data_plot(period_diff_ct_surge_sim_summary, y_key, y_err_key, y_label, title = "Surging Turbine Difference between UMM and LES")

# %%
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
CONTOUR_LABEL_FONTSIZE = 12
LINE_WIDTH = 1.5

# ykey = "avg_mean_CT"
ykey = "avg_max"
# ykey = "avg_min_CT"
# ykey = "avg_phase_max_CT"

# ---- Get thrust coefficient values ----
ct_vals = np.sort(period_diff_ct_surge_sim_summary["Ct_prime"].unique())
assert len(ct_vals) == 4, "Expected exactly four Local Thrust Coefficient values"

# ---- Global contour levels (even spacing) ----
vmin = period_diff_ct_surge_sim_summary[ykey].min()
vmax = period_diff_ct_surge_sim_summary[ykey].max()
levels = np.linspace(vmin, vmax, 12)


# ---- Figure setup ----
fig, axes = plt.subplots(
    2, 2,
    sharex=True,
    sharey=True
)

axes = axes.flatten()

for ax, ct in zip(axes, ct_vals):
    sub = period_diff_ct_surge_sim_summary[
        period_diff_ct_surge_sim_summary["Ct_prime"] == ct
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
cbar.set_label("avg_max", fontsize=LABEL_FONTSIZE)
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
    ltcs = sorted(df_les["Ct_prime"].unique())
    amps = sorted(df_les["Amplitude"].unique())

    # Create FacetGrid by row=Frequency, col=LTC
    g = sns.FacetGrid(
        df_les,
        row="Frequency",
        col="Ct_prime",
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
            (df_les["Ct_prime"] == ltc)
        ].sort_values("Phase_Rounded")
        df_umm_sub = df_umm[
            (df_umm["Frequency"] == freq) &
            (df_umm["Ct_prime"] == ltc)
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
                                                                                                period_avg_metrics, movement_key = "Pitch", phase_key = "Phase",
                                                                                                normalize = normalize, err = 0.05,
)

# %%
for metric in period_avg_metrics:
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
    match_cols = ["Frequency", "Amplitude", "Ct_prime"]

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
        period_avg_stats,
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
        col="Ct_prime",
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
        df_ltc_freq = df[(df["Ct_prime"] == ltc) & (df["Frequency"] == freq)]
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
        col="Ct_prime",
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
        df_ltc = df[df["Ct_prime"] == ltc]

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
for metric in period_avg_metrics:
    y_key, y_err_key = metric

    y_label = get_ct_ylabel(y_key, normalize=normalize)

    y_key = y_key
    y_err_key = y_err_key

    # umm_les_both_data_plot(les_surge_pitch_sim_summary, y_key, y_err_key, y_label, title = "Surging and Pitching Turbine LES")

# %%
for metric in period_avg_metrics:
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

    grouped = df.groupby(["Ct_prime", "Frequency", "Amplitude",  "Movement"])
    
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
            "Ct_prime": ltc,
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
    col="Ct_prime",
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
    df_ltc = phase_max_les_vals[phase_max_les_vals["Ct_prime"] == ltc]

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
group_keys = ["Ct_prime", "Frequency", "Amplitude", "Movement", "Phase_Rounded"]
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
        col="Ct_prime",
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
    ((df_les_surge["Ct_prime"] == 1.33) | (df_les_surge["Ct_prime"] == 2.00)) &
    ((df_les_surge["Frequency"] == 0.4) | (df_les_surge["Frequency"] == 0.8)) &
    ((df_les_surge["Amplitude"] == 0.2) | (df_les_surge["Amplitude"] == 0.6) | (df_les_surge["Amplitude"] == 1.2))
]
sub_umm = df_umm_surge[
    ((df_umm_surge["Ct_prime"] == 1.33) | (df_umm_surge["Ct_prime"] == 2.00)) &
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
    ((df_les_pitch["Ct_prime"] == 1.33) | (df_les_pitch["Ct_prime"] == 2.00)) &
    ((df_les_pitch["Frequency"] == 0.4) | (df_les_pitch["Frequency"] == 0.8)) &
    ((df_les_pitch["Amplitude"] == 4) | (df_les_pitch["Amplitude"] == 8) | (df_les_pitch["Amplitude"] == 16))
]
sub_umm = df_umm_pitch[
    ((df_umm_pitch["Ct_prime"] == 1.33) | (df_umm_pitch["Ct_prime"] == 2.00)) &
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
result_keys = ["Ct_prime", "Frequency", "Amplitude", "Movement"]
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
        col="Ct_prime",
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
            (df["Ct_prime"] == ltc)
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
        col="Ct_prime",
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
        df_ltc = df[df["Ct_prime"] == ltc]

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
        col="Ct_prime",
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
        df_ltc = df[df["Ct_prime"] == ltc]

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
        col="Ct_prime",
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
        df_ltc = df[df["Ct_prime"] == ltc]

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
#     col="Ct_prime",
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
#     df_ltc = df[df["Ct_prime"] == ltc]

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
sub = df_both_les[(df_both_les["Frequency"] == 0.4) & (df_both_les["Amplitude"] == 0.4) & (df_both_les["Ct_prime"] == 1.33) & (df_both_les["PitchAmp"] == 8)]

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
ltcs = sorted(df_both_les_sub["Ct_prime"].unique())
amps = sorted(df_both_les_sub["Amplitude"].unique())
pitch_amps = sorted(df_both_les_sub["PitchAmp"].unique())

# Create FacetGrid by row=Frequency, col=LTC
g = sns.FacetGrid(
    df_both_les_sub,
    row="PitchAmp",
    col="Ct_prime",
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
        (df_both_les_sub["Ct_prime"] == ltc)
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
