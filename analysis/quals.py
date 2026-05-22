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
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
import matplotlib as mpl
from scipy.interpolate import griddata
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %matplotlib inline
mpl.rcParams['figure.dpi'] = 100


# %% [markdown]
# # Surge Runs

# %%
def get_surge_runs(sims_f, sims_Av, CT_primes, max_Ax = 0.6, nperiods = 11, Lx = 25, Ly = 6.4, nx = 256, ny = 128, nflow_through = 3):
    # create DataFrame of all combinations
    my_sims_df = pd.DataFrame(
        [(f, 2*np.pi*f, Av, Av/(2*np.pi*f)) for f in sims_f for Av in sims_Av],
        columns=["f", "k", "Av", "Ax"]
    )

    # filter based on Ax
    my_sims_df = my_sims_df[my_sims_df["Ax"] < max_Ax]

    # --- calculate runtime for each f ---
    sims_T = 1 / sims_f
    # round transient up to next full period
    transient_end = np.ceil(Lx * nflow_through / sims_T) * sims_T
    # total runtime = transient + 10 periods
    sims_runtime = transient_end + nperiods * sims_T

    # map runtime to DataFrame based on f
    runtime_dict = dict(zip(sims_f, sims_runtime))
    my_sims_df["runtime"] = my_sims_df["f"].map(runtime_dict)

    # --- calculate timestep for each f ---
    sims_dt = [0.01 if f <= 1 else 0.005 for f in sims_f]
    dt_dict = dict(zip(sims_f, sims_dt))
    my_sims_df["dt"] = my_sims_df["f"].map(dt_dict)

    # add stationary row
    stationary_row = {
        "f": 0.0,
        "k": 0.0,
        "Av": 0.0,
        "Ax": 0.0,
        "runtime": 150.0,
        "dt": 0.01,
    }
    my_sims_df = pd.concat([my_sims_df, pd.DataFrame([stationary_row])], ignore_index=True)

    # --- repeat for multiple CT' values ---
    ct_df = pd.DataFrame({"CT_prime": CT_primes})
    my_sims_df = my_sims_df.merge(ct_df, how="cross")

    # --- add in LES parameters ---
    my_sims_df["Lx"] = Lx
    my_sims_df["Ly"] = Ly
    my_sims_df["Lz"] = Ly

    my_sims_df["nx"] = nx
    my_sims_df["ny"] = ny
    my_sims_df["nz"] = ny

    # --- calculate filterwidth ---
    h = np.sqrt((Lx / nx)**2 + (Ly / ny)**2 + (Ly / ny)**2)
    my_sims_df["filterWidth"] = 1.5*h
    my_sims_df["useCorrection"] = True

    return my_sims_df


# %% [markdown]
# ## Surge Literature

# %%
literature_df = pd.read_csv("/Users/sky/src/HowlandLab/data/quals_data/Surge_Ax_vs_k.csv")
literature_df = literature_df.rename(columns={'x': 'k', ' y': 'Ax'})

# %%
literature_df["f"] = literature_df["k"] / (2 * np.pi)
literature_df["Av"] = literature_df["Ax"] * literature_df["k"]

# %%
0.3/1.17

# %%
# add in wei and dabiri 2022
D = 1.17 # m
U1 = 8.06 # m/s
T = np.array([12, 6, 3, 2, 1, 0.5])
fstar_all = (1 / T) * D / U1
wei_f = np.concatenate([fstar_all[:3], np.repeat(fstar_all[3], 3), fstar_all[4:]])
wei_Av = [0.039, 0.078, 0.156, 0.234, 0.175, 0.117, 0.234, 0.234]
wei_Ax = np.concatenate([np.repeat(0.514, 4), [0.385], np.repeat(0.257, 2), [0.128]])

wei_df = pd.DataFrame({
    "File": ["wei_2022.csv"] * len(wei_f),
    "k": wei_f * 2 * np.pi,
    "Ax": wei_Ax,
    "f": wei_f,
    "Av": wei_Av
})

# %%
0.3 / D

# %%
# add in wen 2018
D = 126 # m
U1 = 11.4 # m/s
f = np.array([0.065, 0.1, 0.2, 0.3]) # Hz
wen_f = f * D / U1 # [-] 
wen_Ax = [2.5 / D for i in range(len(f))] # [-]
wen_Av = [wen_Ax[i] * 2 * np.pi * f for (i, f) in enumerate(wen_f)]

wen_df = pd.DataFrame({
    "File": ["wen_2018.csv"] * len(wen_f),
    "k": wen_f * 2 * np.pi,
    "Ax": wen_Ax,
    "f": wen_f,
    "Av": wen_Av
})

# %%
wen_df

# %%
# add in Ferreira 2021
ferreira_k = np.array([1.43, 2.77, 5.63, 8.66, 1, 3, 5, 10, 15, 20])
ferreira_Ax = np.concatenate([np.repeat(0.063, 4), np.repeat(0.1, 6)])
ferreira_Av = [ferreira_Ax[i] * k for (i, k) in enumerate(ferreira_k)]

ferreira_df = pd.DataFrame({
    "File": ["ferreira_2021.csv"] * len(ferreira_k),
    "k": ferreira_k,
    "Ax": ferreira_Ax,
    "f": ferreira_k / (2 * np.pi),
    "Av": ferreira_Av
})

# %%
literature_df = pd.concat([literature_df, wei_df, wen_df, ferreira_df], ignore_index=True)
# mapping
citation_map = {
    "de_vaal_2014": "de Vaal et al., 2014",
    "mancini_2020": "Mancini et al., 2020",
    "wei_2022": "Wei and Dabiri, 2022",
    "wen_2018": "Wen et al., 2018",
    "ferreira_2021": "Ferreira et al., 2021",
}

# create citation column with fallback
literature_df["citation"] = (
    literature_df["File"]
    .str.replace(".csv", "", regex=False)
    .map(citation_map)
    .fillna("Other studies (2014-2020)")   # <- professional fallback
)

# %%
literature_df[literature_df["File"] == "mancini_2020.csv"]

# %% [markdown]
# ## Surge Stability

# %%
stability_f = np.array([0.2, 2.0])
stability_Av = np.array([0.1, 0.6])
stability_CT_primes = np.array([2.00])
stability_sims_df = get_surge_runs(stability_f, stability_Av, stability_CT_primes, max_Ax = 0.6, nperiods = 11, Lx = 25, Ly = 6.4, nx = 256, ny = 128, nflow_through = 3)
stability_sims_df

# %%
sns.set_theme(
    context="paper",
    style="whitegrid",
    font="serif",
    font_scale=1.3,
)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.3,
    "legend.frameon": True,
})


def make_sensitivity_plots(key, file, max_ref = True):
    rho, uinf, D, dt = 1, 1, 1, 0.01
    df = pd.read_csv(file)
    df["Ct"] = df["CT_prime"] * (df["UDisk"])**2
    df = df[df["Surge_Amplitude"] > 0]

    df_mean = (
        df.groupby(["id","CT_prime","Frequency","Surge_Amplitude", key])
        .agg(Ct_mean=("Ct","mean"))
        .reset_index()
    )
    # g = sns.relplot(
    #     data=df_mean,
    #     x="Frequency",
    #     y="Ct_mean",
    #     hue=key,
    #     style="Surge_Amplitude",
    #     col="CT_prime",
    #     kind="scatter",
    #     marker="o",
    #     s = 60,
    #     facet_kws={"sharey": True, "sharex": True},
    #     palette="tab10"
    # )

    # g.set_axis_labels("Frequency (f)", "$\overline{C_T}$")
    # g.set_titles("CT' = {col_name}")

    # plt.show()

    # g = sns.relplot(
    #     data=df,
    #     x="Phase",
    #     y="Ct",
    #     hue=key,
    #     style="Surge_Amplitude",
    #     row="Frequency",
    #     col="CT_prime",
    #     kind="scatter",
    #     facet_kws={"sharey": True, "sharex": True},
    #     height=3,
    #     aspect=1.3,
    #     alpha = 0.5,
    #     palette="tab10"
    # )

    # g.set_axis_labels("Phase", "$C_T$")
    # g.set_titles("f={row_name}, CT'={col_name}")

    # plt.show()

    # reference resolution per case
    group_cols = ["CT_prime","Frequency","Surge_Amplitude","Period"]

    ref_val = df.groupby(group_cols)[key].transform("max" if max_ref else "min")

    ref_df = (
        df.loc[df[key] == ref_val,
            ["CT_prime","Frequency","Surge_Amplitude","Period","Phase","Ct"]]
        .rename(columns={"Ct": "Ct_ref"})
    )

    # attach reference Ct
    df_diff = df.merge(
        ref_df,
        on=["CT_prime","Frequency","Surge_Amplitude","Period","Phase"],
        how="left"
    )

    df_diff["Ct_diff"] = df_diff["Ct"] - df_diff["Ct_ref"]

    # period statistics
    period_stats = (
        df_diff.groupby(
            ["id","CT_prime","Frequency","Surge_Amplitude",key,"Period"]
        )
        .agg(
            mean_diff=("Ct_diff","mean"),
            max_diff=("Ct","max"),
            min_diff=("Ct","min"),
            max_ref=("Ct_ref","max"),
            min_ref=("Ct_ref","min"),
        )
    )

    period_stats["max_diff"] -= period_stats["max_ref"]
    period_stats["min_diff"] -= period_stats["min_ref"]

    # average across periods
    df_stats = (
        period_stats.groupby(
            ["id","CT_prime","Frequency","Surge_Amplitude",key]
        )[["mean_diff","max_diff","min_diff"]]
        .mean()
        .reset_index()
    )

    df_stats_long = df_stats.melt(
        id_vars=["id","CT_prime","Frequency","Surge_Amplitude",key],
        value_vars=["mean_diff","max_diff","min_diff"],
        var_name="stat",
        value_name="Ct_diff"
    )

    df_stats_long["stat"] = df_stats_long["stat"].map({
        "mean_diff": "Mean",
        "max_diff": "Max",
        "min_diff": "Min"
    })


    g = sns.relplot(
        data=df_stats_long,
        x="stat",
        y="Ct_diff",
        hue=key,
        row="Surge_Amplitude",
        col="Frequency",
        kind="scatter",
        marker="o",
        s = 60,
        facet_kws={"sharey": True, "sharex": True},
        palette="tab10"
    )

    for ax in g.axes.flat:
        ax.axhspan(-0.01, 0.01, color="gray", alpha=0.15, zorder=0)
        ax.axhline(0.01, linestyle="--", linewidth=1, color="black")
        ax.axhline(-0.01, linestyle="--", linewidth=1, color="black")
        ax.axhline(0.02, linestyle="--", linewidth=1, color="grey")
        ax.axhline(-0.02, linestyle="--", linewidth=1, color="grey")

    g.set_axis_labels("Statistic", r"$\Delta C_T$ (vs reference)")
    g.set_titles("f={col_name}, A={row_name}")

    plt.show()

    # --- Phase-resolved resolution convergence ---
    df_phase_diff = (
        df_diff.groupby(
            ["Phase","Frequency","Surge_Amplitude","CT_prime",key]
        )
        .agg(Ct_diff_mean=("Ct_diff","mean"))
        .reset_index()
    )

    g = sns.relplot(
        data=df_phase_diff,
        x="Phase",
        y="Ct_diff_mean",
        hue=key,
        style="Surge_Amplitude",
        row="Surge_Amplitude",
        col="Frequency",
        kind="line",
        facet_kws={"sharey": True, "sharex": True},
        height=3,
        aspect=1.3,
        palette="tab10"
    )

    for ax in g.axes.flat:
        ax.axhspan(-0.01, 0.01, color="gray", alpha=0.15, zorder=0)
        ax.axhline(0.01, linestyle="--", linewidth=1, color="black")
        ax.axhline(-0.01, linestyle="--", linewidth=1, color="black")
        ax.axhline(0.02, linestyle="--", linewidth=1, color="grey")
        ax.axhline(-0.02, linestyle="--", linewidth=1, color="grey")

    g.set_axis_labels("Phase", r"$C_T - C_{T,ref}$")
    g.set_titles("f={col_name}, A={row_name}")

    plt.show()

# %% [markdown]
# ### Periods to Average Over

# %%
# make_sensitivity_plots("nperiods", "/Users/sky/src/HowlandLab/data/quals_data/surge_nperiod_results.csv")

# %% [markdown]
# ### Spinup Time

# %%
# make_sensitivity_plots("spinup_time", "/Users/sky/src/HowlandLab/data/quals_data/surge_spinup_results.csv")

# %% [markdown]
# ### Resolution

# %%
Lx = 25
Ly = 6.4

nx = 256
ny = np.array([256, 128, 64])
n_hrs = [4, 2, 1]

res_runs = []
for (i, n) in enumerate(ny):
    df = stability_sims_df.copy()
    df["ny"] = n
    df["nz"] = n
    df["n_hrs"] = n_hrs[i]
    df["filterWidth"] = 1.5 * np.sqrt((Lx / nx)**2 + 2 * (Ly / n)**2)
    res_runs.append(df)
res_sims_df = pd.concat(res_runs, ignore_index=True)

# -------------------------
# add ny=512 only for A=0.6
# -------------------------
df_high = stability_sims_df[stability_sims_df["Av"] == 0.6].copy()

df_high["ny"] = 512
df_high["nz"] = 512
df_high["n_hrs"] = 8  # adjust if needed
df_high["filterWidth"] = 1.5 * np.sqrt((Lx / nx)**2 + 2 * (Ly / 512)**2)

# -------------------------
# add ny=192 only for A=0.6
# -------------------------
df_mid_red = stability_sims_df[stability_sims_df["Av"] == 0.6].copy()

df_mid_red["ny"] = 192
df_mid_red["nz"] = 192
df_mid_red["n_hrs"] = 3  # adjust if needed
df_mid_red["filterWidth"] = 1.5 * np.sqrt((Lx / nx)**2 + 2 * (Ly / 192)**2)

# append to end so order of existing rows stays unchanged
res_sims_df = pd.concat([res_sims_df, df_high, df_mid_red], ignore_index=True)

res_sims_df.to_csv("surge_res_runs.csv", index=False)

# %%
make_sensitivity_plots("ny", "/Users/sky/src/HowlandLab/data/quals_data/surge_ny_results.csv")

# %% [markdown]
# ### Timesteps

# %%
Lx = 25
Ly = 6.4

nx = 256
ny = 128

n_hrs = [4, 2, 1]

dts = [0.005, 0.01, 0.02]
dt_runs = []
for (i, dt) in enumerate(dts):
    df = stability_sims_df.copy()
    df["dt"] = dt
    df.loc[df["f"] == 2, "dt"] = dt / 2
    df["n_hrs"] = n_hrs[i]
    dt_runs.append(df)

dt_sims_df = pd.concat(dt_runs, ignore_index=True)


# -------------------------
# add dt = 0.0025 only for f=0.2
# -------------------------
df_slow = stability_sims_df[stability_sims_df["f"] == 0.2].copy()

df_slow["dt"] = 0.0025
df_slow["n_hrs"] = 4  # adjust if needed

# append to end so order of existing rows stays unchanged
dt_sims_df = pd.concat([dt_sims_df, df_slow], ignore_index=True)

dt_sims_df.to_csv("surge_dt_runs.csv", index=False)

# %%
make_sensitivity_plots("dt", "/Users/sky/src/HowlandLab/data/quals_data/surge_dt_results.csv", max_ref = False) # min dt is best

# %% [markdown]
# ### Filter Width

# %%
Lx = 25
Ly = 6.4

nx = 256
ny = 128

dx, dy = Lx / nx, Ly / ny
points_across_D = D / dy
points_across_D, dx, dy

h = np.sqrt(dx**2 + dy**2 + dy**2)
fW_M, fW_no_M = 5*h/2, h/2

filterWidth_runs1 = stability_sims_df.copy()
filterWidth_runs1["filterWidth"] = fW_M
filterWidth_runs1["useCorrection"] = True

filterWidth_runs2 = stability_sims_df.copy()
filterWidth_runs2["filterWidth"] = fW_no_M
filterWidth_runs2["useCorrection"] = False

fW_sims_df = pd.concat([filterWidth_runs1, filterWidth_runs2], ignore_index=True)

fW_sims_df.to_csv("surge_fW_runs.csv", index=False)

# %%
phase_grid = np.round(np.linspace(0, 0.99, 100), decimals=2)
phase_grid

# %%
surge_fW_df = pd.read_csv("/Users/sky/src/HowlandLab/data/quals_data/surge_fW_results.csv")
surge_fW_df.keys()

# %%
rho, uinf, D, dt = 1, 1, 1, 0.01
surge_fW_df = pd.read_csv("/Users/sky/src/HowlandLab/data/quals_data/collected_runs_all.csv")
surge_fW_df["Ct"] = surge_fW_df["CT_prime"] * (surge_fW_df["UDisk"])**2
surge_fW_df = surge_fW_df[surge_fW_df["Surge_Amplitude"] > 0]

surge_fW_df_mean = (
    surge_fW_df.groupby(["id","CT_prime","Frequency","Surge_Amplitude","useCorrection"])
      .agg(Ct_mean=("Ct","mean"))
      .reset_index()
)
g = sns.relplot(
    data=surge_fW_df_mean,
    x="Frequency",
    y="Ct_mean",
    hue="useCorrection",
    style="Surge_Amplitude",
    col="CT_prime",
    kind="scatter",
    marker="o",
    facet_kws={"sharey": True, "sharex": True},
    palette="magma"
)

g.set_axis_labels("Frequency (f)", "$\overline{C_T}$")
g.set_titles("CT' = {col_name}")

plt.show()

g = sns.relplot(
    data=surge_fW_df,
    x="Phase",
    y="Ct",
    hue="useCorrection",
    style="Surge_Amplitude",
    row="Frequency",
    col="CT_prime",
    kind="scatter",
    facet_kws={"sharey": True, "sharex": True},
    height=3,
    aspect=1.3,
    alpha = 0.5,
    palette="magma"
)

g.set_axis_labels("Phase", "$C_T$")
g.set_titles("f={row_name}, CT'={col_name}")

plt.show()

# %%
keys = ["Frequency","CT_prime","Surge_Amplitude","Period","Phase"]

df_on = surge_fW_df[surge_fW_df["useCorrection"] == True]
df_off = surge_fW_df[surge_fW_df["useCorrection"] == False]

df_diff = (
    df_on.merge(
        df_off,
        on=keys,
        suffixes=("_on","_off")
    )
)
df_diff["Ct_diff"] = df_diff["Ct_off"] - df_diff["Ct_on"]

g = sns.relplot(
    data=df_diff[df_diff["Surge_Amplitude"] == 0.1],
    x="Phase",
    y="Ct_diff",
    hue="Period",
    row="Frequency",
    col="CT_prime",
    kind="scatter",
    facet_kws={"sharey": True, "sharex": True},
    height=3,
    aspect=1.3,
    palette="viridis_r",
    alpha = 0.5
)
g.figure.suptitle("$C_T$ Difference between Correction Off and On (A* = 0.1)", y = 1.03)
g.set_axis_labels("Phase", r"$\Delta C_T$")
g.set_titles("f={row_name}, CT'={col_name}")

# %% [markdown]
# ### Blockage (<2%)

# %%
D = 1
rotor_area = np.pi * (D/2)**2
domain_area = Ly*Ly
blockage = rotor_area/domain_area
blockage

# %%
Lx = 25
Ly = [12.8, 6.4, 3.2]

nx = 256
ny = np.array([256, 128, 64])
n_hrs = [4, 2, 1]

len_runs = []
for (i, L) in enumerate(Ly):
    df = stability_sims_df.copy()
    df["ny"] = ny[i]
    df["nz"] = ny[i]
    df["Ly"] = Ly[i]
    df["Lz"] = Ly[i]
    df["filterWidth"] = 1.5 * np.sqrt((Lx/nx)**2 + 2*(L / ny[i])**2)
    df["n_hrs"] = n_hrs[i]
    len_runs.append(df)

len_sims_df = pd.concat(len_runs, ignore_index=True)


# -------------------------
# add Ly=8.5 only for A=0.6
# -------------------------
df_mid_red = stability_sims_df[stability_sims_df["Av"] == 0.6].copy()

df_mid_red["ny"] = 170
df_mid_red["nz"] = 170
df_mid_red["Ly"] = 8.5
df_mid_red["Lz"] = 8.5
df_mid_red["n_hrs"] = 3  # adjust if needed
df_mid_red["filterWidth"] = 1.5 * np.sqrt((Lx / nx)**2 + 2 * (8.5 / 170)**2)

# append to end so order of existing rows stays unchanged
len_sims_df = pd.concat([len_sims_df, df_mid_red], ignore_index=True)
len_sims_df.to_csv("surge_len_runs.csv", index=False)

# %%
make_sensitivity_plots("Ly", "/Users/sky/src/HowlandLab/data/quals_data/surge_ly_results.csv", max_ref = True) # min dt is best

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = [
    ("/Users/sky/src/HowlandLab/data/quals_data/surge_ny_results.csv", "ny", [256, 192, 128]),
    ("/Users/sky/src/HowlandLab/data/quals_data/surge_dt_results.csv", "dt", [0.0025, 0.005, 0.01]),
    ("/Users/sky/src/HowlandLab/data/quals_data/surge_ly_results.csv", "Ly", [12.8, 8.5, 6.4]),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

for ax, (file, key, order) in zip(axes, files):
    
    df = pd.read_csv(file)
    df["Ct"] = df["CT_prime"] * (df["UDisk"])**2
    df = df[df["Surge_Amplitude"] == 0.6]
    
    sims = df[["CT_prime", "Frequency"]].drop_duplicates().values
    stats_names = ["Mean", "Max"]
    
    # store results per resolution
    res_values = {val: [] for val in order}
    
    for (ct, f) in sims:
        sub_sim = df[(df["CT_prime"] == ct) & (df["Frequency"] == f)]
        
        for stat in stats_names:
            
            stat_vals = {}
            for val in order:
                sub = sub_sim[sub_sim[key] == val]
                
                if stat == "Mean":
                    v = sub["Ct"].mean()
                else:  # Max
                    v = sub["Ct"].max()
                
                stat_vals[val] = v
            
            # reference per stat (fine resolution)
            ref_val = min(order) if key == "dt" else max(order)
            ref = stat_vals[ref_val]
            
            for val in order:
                res_values[val].append((stat_vals[val] - ref) / ref)
    
    x = list(range(len(stats_names) * len(sims)))
    
    # --- legend formatting ---
    if key == "dt":
        legend_title = r"$\Delta t$"
        label_map = {val: f"{val}" for val in order}
        title = "Timestep"
        
    elif key == "ny":
        legend_title = r"$\Delta y$"
        label_map = {val: f"{6.4/val:.3f}" for val in order}
        title = "Resolution"
        
    elif key == "Ly":
        legend_title = "Blockage"
        label_map = {
            val: f"{(np.pi * (0.5)**2 / (val**2) * 100):.3f}%"
            for val in order
        }
        title = "Blockage"
    
    # jitter so dots don't overlap
    offsets = [0, 0, 0]
    
    for offset, val in zip(offsets, order):
        ax.scatter(
            [xi + offset for xi in x],
            res_values[val],
            s=24,
            label=label_map[val]
        )
    
    # =========================
    # GROUPED X-AXIS LABELS
    # =========================
    
    inner_labels = stats_names * len(sims)
    ax.set_xticks(x)
    ax.set_xticklabels(inner_labels, fontsize=9)
    ax.tick_params(axis='x', labelrotation=30)
    ax.tick_params(axis='y', labelsize=9)
    
    for i, (ct, f) in enumerate(sims):
        center = i * 2 + 0.5  # updated for 2 stats
        
        ax.text(
            center,
            0.95,
            f"f={f:.2f}",
            ha='center',
            va='top',
            transform=ax.get_xaxis_transform(),
            fontsize=10
        )
    
    for i in range(1, len(sims)):
        ax.axvline(i*2 - 0.5, color='k', linewidth=0.5, alpha=0.3)
    
    # =========================
    
    ax.set_title(title, fontsize=10)
    
    ax.axhspan(-0.01, 0.01, alpha=0.1)
    ax.axhline(0, linewidth=0.8)
    
    # ✅ legend outside
    ax.legend(
        title=legend_title,
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

axes[0].set_ylabel(r"$\Delta C_T$", size=11)

# Make room for outside legends
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Optional spacing between subplots
fig.subplots_adjust(wspace=0.4)

plt.savefig("LES_sensitivity.png", dpi=900)

# %% [markdown]
# ## Final Surge Runs

# %%
sims_f = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
sims_Av = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
CT_primes = np.array([1.33, 2.00])
my_sims_df = get_surge_runs(sims_f, sims_Av, CT_primes, max_Ax = 0.6, nperiods = 10, Lx = 25, Ly = 8.5, nx = 256, ny = 256, nflow_through = 3)
my_sims_df["dt"] = 0.005
my_sims_df["n_hrs"] = np.ceil(my_sims_df["runtime"] / 80 * 3) + 1
my_sims_df.to_csv("surge_final_runs.csv", index=False)
my_sims_df

# %%
lit_plot = literature_df
sim_plot = my_sims_df[(my_sims_df["Av"] / my_sims_df["k"] <= 0.5)]

plt.figure(figsize=(6,5), dpi=300)

# plot literature
sns.scatterplot(
    data=lit_plot,
    x="k",
    y="Ax",
    s=40,
    label="Literature"
)

# my simulations
sns.scatterplot(
    data=sim_plot,
    x="k",
    y="Ax",
    color="red",
    s=50,
    label="My Simulations"
)

# avoid divide-by-zero
k_vals = np.linspace(0.05, 15.5, 400)

# Av contour levels
Av_levels = [0.1, 0.25, 0.5, 0.75, 1.0]

x_offsets = [0, 4, 8, 11, 12]
y_offsets = [-0.01, 0.005, 0.028, 0.1, 0.16]
rotations = [0, -5, -10, -22, -30]

for i, Av in enumerate(Av_levels):

    Ax_line = Av / k_vals

    plt.plot(
        k_vals,
        Ax_line,
        "--",
        color="gray",
        linewidth=1
    )

    # alternate vertical offset
    x_offset = x_offsets[i]
    y_offset = y_offsets[i]

    if Av == 1:
        label = f"$A_v={Av:.2f} U_\infty$"
    else:
        label = f"${Av:.2f}$"

    plt.text(
        k_vals[-1] - x_offset,
        Ax_line[-1] + y_offset,
        label,
        fontsize=11,
        color="k",
        rotation = rotations[i]
    )

plt.xlabel("Reduced Frequency $k$", fontsize=14)
plt.ylabel("Displacement Amplitude $A_x$", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlim(-0.05, 17)
plt.ylim(-0.01, 0.6)

# move legend slightly upward
plt.legend(loc="center right", bbox_to_anchor=(1, 0.65), fontsize=12)

plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Remove points with Ax > 0.5
lit_plot = literature_df
sim_plot = my_sims_df[(my_sims_df["Av"] / my_sims_df["k"] <= 0.5)]

plt.figure(figsize=(6,5), dpi=300)

# scatter

sns.scatterplot(
    data=sim_plot,
    x="f",
    y="Av",
    color="tab:orange",
    s=80,
    label="MIT LES Simulations"
)
sns.scatterplot(
    data=lit_plot,
    x="f",
    y="Av",
    s=80,
    # label="Literature",
    style = "citation"
)

# range of k values
f_vals = np.linspace(0, literature_df["f"].max(), 200)
k_vals = np.linspace(0, literature_df["k"].max(), 200)

Ax_levels = np.arange(0.1, 0.6, 0.1)

y_offsets = [0.01, 0.05]  # alternate offsets

for i, Ax in enumerate(Ax_levels):
    Av_line = Ax * k_vals
    mask = Av_line <= 1.1

    plt.plot(
        f_vals[mask],
        Av_line[mask],
        "--",
        color="grey",
        linewidth=1
    )

    # alternate label offset
    y_offset = y_offsets[i % 2]
    if Ax == 0.5:
        xoffset = 0.2
    else:
        xoffset = 0.08

    if Ax == 0.1:
        label = f"$A_x = {Ax:.1f}D$"
    else:
        label = f"${Ax:.1f}D$"

    plt.text(
        f_vals[mask][-1] - xoffset,
        Av_line[mask][-1] + y_offset,
        label,
        fontsize=9,
        color="k"
    )

plt.xlabel("Frequency $f^*$ [-]", fontsize=14)
plt.ylabel("Velocity Amplitude $A_v$ [-]", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.ylim(0, 1.2)

# move legend slightly up
plt.legend(loc="center right", bbox_to_anchor=(0.98, 0.685), fontsize=9)

plt.tight_layout()
# plt.show()
plt.savefig("literature.png", dpi = 600)


# %% [markdown]
# # Analyzing Final Surge Runs

# %%
def umm_les_contour_plot(
    df,
    x_key="Surge_Amplitude",
    y_key="Frequency",
    z_key="Diff_CT_mean",   # adjust as needed
    z_label=r"$\Delta C_T$",
    levels=20,
    show_scatter=True,
    vmin=None,
    vmax=None,
    grid_res=100,   # controls smoothness
):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    TITLE_FONTSIZE = 16
    LABEL_FONTSIZE = 14
    TICK_FONTSIZE = 12

    # ---- Get CT' values ----
    ctp_vals = np.sort(df["CT_prime"].unique())

    # ---- Global color limits ----
    if vmin is None:
        vmin = df[z_key].min()
    if vmax is None:
        vmax = df[z_key].max()

    dz = 0.05
    # snap bounds to multiples of 0.05
    vmin = dz * np.floor(vmin / dz)
    vmax = dz * np.ceil(vmax / dz)

    levels = np.arange(vmin, vmax + dz, dz)

    # ---- Figure ----
    fig, axes = plt.subplots(
        1, 2,
        sharex=True,
        sharey=True,
        figsize=(12, 4),
    )
    axes = axes.flatten()

    cf_last = None

    for ax, ct in zip(axes, ctp_vals):

        sub = df[df["CT_prime"] == ct].copy()

        # remove NaNs
        sub = sub.dropna(subset=[x_key, y_key, z_key])

        x = sub[x_key].values
        y = sub[y_key].values
        z = sub[z_key].values

        # ---- create regular grid ----
        xi = np.linspace(x.min(), x.max(), grid_res)
        yi = np.linspace(y.min(), y.max(), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        # ---- interpolate (smooth field) ----
        Zi = griddata(
            (x, y),
            z,
            (Xi, Yi),
            method="cubic"   # smoothest; use "linear" if unstable
        )

        # ---- filled contour ----
        cf = ax.contourf(
            Xi,
            Yi,
            Zi,
            levels=levels,
            cmap="viridis_r",
            extend="both"
        )

        # ---- contour lines ----
        ax.contour(
            Xi,
            Yi,
            Zi,
            levels=levels,
            colors="k",
            linewidths=1.0
        )

        # ---- scatter points ----
        if show_scatter:
            ax.scatter(x, y, marker="+", color="k", zorder=3)

        # ---- labels ----
        ax.set_title(f"$C_T' = {ct}$", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Phase $\phi$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("$A_v$", fontsize=LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)

        cf_last = cf

    # ---- colorbar ----
    # fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf_last, cax=cax)
    cbar.set_label(z_label, fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.1,
        top=0.92,
        wspace=0.3,
        hspace=0.35,
    )

    return fig, axes


# %%
def umm_les_contour_plot_with_diff(
    df,
    x_key="Surge_Amplitude",
    y_key="Frequency",
    z_key="Diff_CT_mean",   # for individual plots
    ct_diff=(1.33, 2),      # CT' values to subtract for the third axis
    z_label=r"$\Delta C_T$",
    levels=None,
    show_scatter=True,
    vmin=None,
    vmax=None,
    grid_res=100,
):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    TITLE_FONTSIZE = 18
    LABEL_FONTSIZE = 16
    TICK_FONTSIZE = 14

    # ---- Get CT' values ----
    ctp_vals = np.sort(df["CT_prime"].unique())
    if len(ctp_vals) < 2:
        raise ValueError("Need at least two CT_prime values for difference plot.")

    # ---- Global color limits ----
    if vmin is None:
        vmin = df[z_key].min()
    if vmax is None:
        vmax = df[z_key].max()
    vmin = -np.maximum(abs(vmin), abs(vmax))
    vmax = np.maximum(abs(vmin), abs(vmax))
    dz = 0.05
    # snap bounds to multiples of 0.05
    vmin = dz * np.floor(vmin / dz)
    vmax = dz * np.ceil(vmax / dz)

    if levels is None:
        levels = np.arange(vmin, vmax + dz, dz)

    # ---- Figure ----
    fig, axes = plt.subplots(
        1, 3,
        sharex=True,
        sharey=True,
        figsize=(18, 4),
    )
    axes = axes.flatten()

    cf_last = None

    # ---- First two axes: individual CT' plots ----
    for ax, ct in zip(axes[:2], ctp_vals[:2]):
        sub = df[df["CT_prime"] == ct].copy()
        sub = sub.dropna(subset=[x_key, y_key, z_key])
        x = sub[x_key].values
        y = sub[y_key].values
        z = sub[z_key].values

        # create regular grid
        xi = np.linspace(x.min(), x.max(), grid_res)
        yi = np.linspace(y.min(), y.max(), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        # interpolate
        Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

        # filled contour
        cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap="PRGn", extend="both")
        ax.contour(Xi, Yi, Zi, levels=levels, colors="k", linewidths=1.0)

        # scatter
        if show_scatter:
            ax.scatter(x, y, marker="+", color="k", zorder=3)

        # labels
        ax.set_title(f"$C_T' = {ct}$", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Phase $\phi$", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("$A_v$", fontsize=LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)

        cf_last = cf

    # ---- Third axis: difference ----
    ax = axes[2]
    ct_low, ct_high = ct_diff

    sub_low = df[df["CT_prime"] == ct_low].dropna(subset=[x_key, y_key, z_key])
    sub_high = df[df["CT_prime"] == ct_high].dropna(subset=[x_key, y_key, z_key])

    # create grid based on union of x and y ranges
    x_all = np.unique(np.concatenate([sub_low[x_key], sub_high[x_key]]))
    y_all = np.unique(np.concatenate([sub_low[y_key], sub_high[y_key]]))
    xi = np.linspace(x_all.min(), x_all.max(), grid_res)
    yi = np.linspace(y_all.min(), y_all.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi_low = griddata((sub_low[x_key], sub_low[y_key]), sub_low[z_key], (Xi, Yi), method="cubic")
    Zi_high = griddata((sub_high[x_key], sub_high[y_key]), sub_high[z_key], (Xi, Yi), method="cubic")

    Zi_diff = Zi_high - Zi_low

    cf = ax.contourf(Xi, Yi, Zi_diff, levels=levels, cmap="PRGn", extend="both")
    ax.contour(Xi, Yi, Zi_diff, levels=levels, colors="k", linewidths=1.0)
    if show_scatter:
        ax.scatter(sub_low[x_key], sub_low[y_key], marker="+", color="k", zorder=3)
        ax.scatter(sub_high[x_key], sub_high[y_key], marker="x", color="r", zorder=3)

    ax.set_title(f"$\Delta C_T'$", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Phase $\phi$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("$A_v$", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    cf_last = cf

    # ---- colorbar ----
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf_last, cax=cax)
    cbar.set_label(z_label, fontsize=TITLE_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.1,
        top=0.92,
        wspace=0.3,
        hspace=0.35,
    )

    return fig, axes


# %%
final_surge_df = pd.read_csv("/Users/sky/src/HowlandLab/data/quals_data/surge_final_runs_interp.csv")
final_surge_df = final_surge_df.rename(columns={"UDisk": "LES_UDisk", "Power": "LES_Power", "DeltaX": "XTurb"})

# %%
final_surge_df["Surge_Displacement"] = np.where(
    final_surge_df["Frequency"] == 0,
    0,
    final_surge_df["Surge_Amplitude"] / (2 * np.pi * final_surge_df["Frequency"])
)

# %%
final_surge_df = final_surge_df[final_surge_df["Surge_Displacement"] <= 0.4]

# %%
UInf = 1
D = 1
rho = 1

final_surge_df["URel"] = UInf - final_surge_df["UTurb"]
omega = 2 * np.pi * final_surge_df["Frequency"]
final_surge_df["ATurb"] = - (omega**2) * final_surge_df["XTurb"]

final_surge_df["LES_CP"] =  final_surge_df["LES_Power"] / (0.5 * rho * math.pi * (D/2)**2 * (UInf)**3)
final_surge_df["LES_CT"] =  final_surge_df["CT_prime"] * final_surge_df["LES_UDisk"]**2 / (UInf)**2
final_surge_df["LES_an_T"] = 1 - (final_surge_df["LES_UDisk"] / UInf)
final_surge_df["LES_an_G"] = 1 - (final_surge_df["LES_UDisk"] / final_surge_df["URel"])

# %%
model = UnifiedMomentum()
CT_primes = [1.33, 2.00]
umm_sols = [model(ctp, yaw=0.0, tilt=0.0) for ctp in CT_primes]

UMM_UDisk, UMM_an_T, UMM_an_G, UMM_CT, UMM_CP = [], [], [], [], []
for row in final_surge_df.itertuples(index=False):
    ctp = row.CT_prime
    umm_sol = umm_sols[0] if ctp == CT_primes[0] else umm_sols[1]
    ud = (1 - umm_sol.an) * row.URel
    UMM_UDisk.append(ud)
    UMM_an_T.append(1 - ud / row.URel)
    UMM_an_G.append(1 - ud / UInf)
    UMM_CT.append(umm_sol.Ct * row.URel**2)
    UMM_CP.append(umm_sol.Cp * row.URel**3)

final_surge_df["UMM_UDisk"] = UMM_UDisk
final_surge_df["UMM_an_T"] = UMM_an_T
final_surge_df["UMM_an_G"] = UMM_an_G
final_surge_df["UMM_CT"] = UMM_CT
final_surge_df["UMM_CP"] = UMM_CP
final_surge_df["UMM_Power"] = np.array(UMM_CP) * (0.5 * rho * math.pi * (D/2)**2 * (UInf)**3)

# %%
# identify stationary rows
stationary_mask = (final_surge_df["Frequency"] == 0) & (final_surge_df["Surge_Amplitude"] == 0)

# compute reference values (one per CT')
ref_df = (
    final_surge_df[stationary_mask]
    .groupby("CT_prime")[[
        "LES_CP", "LES_CT", "LES_an_T", "LES_an_G", "LES_Power", "LES_UDisk",
        "UMM_CP", "UMM_CT", "UMM_an_T", "UMM_an_G", "UMM_Power", "UMM_UDisk"
    ]]
    .mean()
    .rename(columns=lambda c: f"{c}_ref")
    .reset_index()
)

# %%
ref_df

# %%
(0.761473 - 0.749836) / .761473, (0.912980 - 0.894033) / 0.912980

# %%
final_surge_df = final_surge_df.merge(ref_df, on="CT_prime", how="left", suffixes=("", "_ref"))

# %%
normalize = False

if normalize:
    # LES normalization
    les_cols = ["LES_CP", "LES_CT", "LES_an_T", "LES_an_G", "LES_Power", "LES_UDisk"]

    for col in les_cols:
        final_surge_df[col] = final_surge_df[col] / final_surge_df[f"{col}_ref"]

    # UMM normalization
    umm_cols = ["UMM_CP", "UMM_CT", "UMM_an_T", "UMM_an_G", "UMM_Power", "UMM_UDisk"]

    for col in umm_cols:
        final_surge_df[col] = final_surge_df[col] / final_surge_df[f"{col}_ref"]

    ylabel_mean = r"$\overline{\Delta C^*_T}$"
    ylabel_max = r"$\overline{\Delta \max C^*_T}$"
    ylabel_min = r"$\overline{\Delta \min C^*_T}$"
    ylabel_umm_phase = r"$\langle{C^*_{T_{\text{UMM}}}}\rangle_\phi$"
    ylabel_les_phase = r"$\langle{C^*_{T_{\text{LES}}}}\rangle_\phi$"
    ylabel_diff_phase = r"$\langle{\Delta C^*_T}\rangle_\phi$"
else:
    ylabel_mean = r"$\overline{\Delta C_T}$"
    ylabel_max = r"$\overline{\Delta \max C_T}$"
    ylabel_min = r"$\overline{\Delta \min C_T}$"
    ylabel_umm_phase = r"$\langle{C_{T_{\text{UMM}}}}\rangle_\phi$"
    ylabel_les_phase = r"$\langle{C_{T_{\text{LES}}}}\rangle_\phi$"
    ylabel_diff_phase = r"$\langle{\Delta C_T}\rangle_\phi$"

# %%
final_surge_df["Diff_UDisk"] = final_surge_df["LES_UDisk"] - final_surge_df["UMM_UDisk"]
final_surge_df["Diff_an_T"] = final_surge_df["LES_an_T"] - final_surge_df["UMM_an_T"]
final_surge_df["Diff_an_G"] = final_surge_df["LES_an_G"] - final_surge_df["UMM_an_G"]
final_surge_df["Diff_CT"] = final_surge_df["LES_CT"] - final_surge_df["UMM_CT"]
final_surge_df["Diff_CP"] = final_surge_df["LES_CP"] - final_surge_df["UMM_CP"]
final_surge_df["Diff_Power"] = final_surge_df["LES_Power"] - final_surge_df["UMM_Power"]

# %% [markdown]
# # Mean and Standard Deviation for each Phase, $C_T'$, $f^*$, and $A_S^*$

# %%
phase_group_cols = ["Phase", "CT_prime", "Surge_Amplitude", "Frequency"]
period_group_cols = ["CT_prime", "Surge_Amplitude", "Frequency"]

stat_cols = ["XTurb", "UTurb", "URel", "ATurb",
    "LES_CP", "LES_CT", "LES_an_T", "LES_an_G", "LES_UDisk", "LES_Power",
    "UMM_CP", "UMM_CT", "UMM_an_T", "UMM_an_G", "UMM_UDisk", "UMM_Power",
    "Diff_CP", "Diff_CT", "Diff_an_T", "Diff_an_G", "Diff_UDisk", "Diff_Power"
]

phase_stats = final_surge_df.groupby(phase_group_cols)[stat_cols].agg(["mean", "std"]).reset_index()

# flatten column names
phase_stats.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in phase_stats.columns
]

# %%
plot_df = phase_stats.melt(
    id_vars=["Phase", "CT_prime", "Surge_Amplitude", "Frequency"],
    value_vars=["LES_CT_mean", "UMM_CT_mean", "Diff_CT_mean"],
    var_name="Source",
    value_name="CT"
)

# %%
# Melt dataframe
plot_df = phase_stats.melt(
    id_vars=["Phase", "CT_prime", "Surge_Amplitude", "Frequency"],
    value_vars=["LES_CT_mean", "UMM_CT_mean", "Diff_CT_mean"],
    var_name="Source",
    value_name="CT"
)

plot_df["Source"] = plot_df["Source"].str.replace("_CT_mean", "")
plot_df["Source"] = pd.Categorical(
    plot_df["Source"],
    categories=["UMM", "LES", "Diff"],
    ordered=True
)

sns.set_style("ticks")
sns.set_context("notebook")

# Create FacetGrid without legend
g = sns.relplot(
    data=plot_df[
        (plot_df["Frequency"] == 0.5) &
        (plot_df["Surge_Amplitude"] > 0) &
        (plot_df["CT_prime"] == 2)
    ],
    x="Phase",
    y="CT",
    hue="Surge_Amplitude",
    style="Frequency",
    col="Source",
    kind="line",
    markers=False,
    dashes=False,
    palette=sns.color_palette("viridis", 8),
    height=5,
    aspect=1.2,
    lw=3,
    facet_kws={"sharey": False},
    legend=False
)

# Match first two y-axes
ymin, ymax = g.axes_dict["LES"].get_ylim()
g.axes_dict["UMM"].set_ylim(ymin, ymax)

# Custom titles
label_map = [ylabel_umm_phase, ylabel_les_phase, ylabel_diff_phase]
title_map = ["UMM", "LES", "LES - UMM"]

# Axes labels, ticks, grid
for (i, ax) in enumerate(g.axes.flat):
    ax.set_title(title_map[i], fontsize=26)
    ax.set_xlabel("Phase $\phi$", fontsize=26)
    ax.set_ylabel(label_map[i], fontsize=28)
    ax.tick_params(labelsize=24)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(0.5, linestyle="--", linewidth=2, color="k", alpha = 0.5)

g.axes.flat[2].text(
    0.55, 0.63,
    "axis magnified\nfor visability",
    transform=ax.transAxes,
    ha='left',
    va='bottom',
    fontsize=22,
)

# --- Safe horizontal legend ---
# Draw a temporary line plot to generate legend handles
unique_hues = sorted(plot_df["Surge_Amplitude"].unique())
unique_hues = unique_hues[1:]
handles = [mlines.Line2D([], [], color=sns.color_palette("viridis", len(unique_hues))[i], lw=5)
           for i in range(len(unique_hues))]
labels = [str(s) for s in unique_hues]

g.fig.legend(
    handles, labels,
    title="$A_v$",
    title_fontsize=28,
    fontsize=24,
    loc='upper center',
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.54, 1.2)
)

g.fig.subplots_adjust(top=0.85, right=0.95, left=0.15, wspace=0.4) 
g.fig.suptitle("LES vs UMM $C^*_T$ Response Across Phase: $f^* = 0.5$, $C_T' = 2$", fontsize=26, y = 1.25)

plt.savefig("f_05_lines.png", dpi = 600, bbox_inches='tight')

# %%
sns.set_style("ticks")
sns.set_context("notebook")

# -------------------------
# FILTER DATA
# -------------------------
df_line = plot_df[
    (plot_df["Frequency"] == 0.5) &
    (plot_df["Surge_Amplitude"] > 0) &
    (plot_df["CT_prime"] == 2)
]

df_contour = phase_stats[
    (phase_stats["Frequency"] == 0.5) &
    (phase_stats["Surge_Amplitude"] > 0) &
    (phase_stats["CT_prime"] == 2)
]

# -------------------------
# FIGURE SETUP
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

axs = axes.flatten()

# -------------------------
# COLOR SETUP (shared)
# -------------------------
avs = sorted(df_line["Surge_Amplitude"].unique())
palette = sns.color_palette("viridis", len(avs))
color_map = dict(zip(avs, palette))

# -------------------------
# TOP ROW: LES and UMM
# -------------------------
label_map = [ylabel_umm_phase, ylabel_les_phase, ylabel_diff_phase]

for j, source in enumerate(["UMM", "LES"]):
    ax = axes[0, j]

    sub = df_line[df_line["Source"] == source]

    for av in avs:
        s = sub[sub["Surge_Amplitude"] == av]
        ax.plot(s["Phase"], s["CT"], lw=2.5, color=color_map[av])
        ax.tick_params(labelsize=18)
        ax.set_ylim(-0.1, 3)

    ax.set_title(source + ": " + label_map[j] + " vs $\phi$" , fontsize=20)
    ax.set_ylabel(label_map[j], fontsize = 20)
    ax.grid(True, linestyle="--", alpha=0.3)

axes[0,0].set_xlabel("")
axes[0,1].set_xlabel("")

# -------------------------
# BOTTOM LEFT: LES - UMM LINE
# -------------------------
ax = axes[1,0]

diff_line = df_line[df_line["Source"] == "Diff"]

for av in avs:
    s = diff_line[diff_line["Surge_Amplitude"] == av]
    ax.plot(s["Phase"], s["CT"], lw=2.5, color=color_map[av])

ax.set_title(f"LES - UMM {label_map[2]} vs $\phi$", fontsize=20)
ax.set_xlabel("Phase $\phi$", fontsize = 20)
ax.set_ylabel(label_map[2], fontsize = 20)
ax.grid(True, linestyle="--", alpha=0.3)
ax.tick_params(labelsize=18)

# -------------------------
# BOTTOM RIGHT: CONTOUR
# -------------------------
ax = axes[1,1]

x = df_contour["Phase"].values
y = df_contour["Surge_Amplitude"].values
z = df_contour["Diff_CT_mean"].values

xi = np.linspace(x.min(), x.max(), 150)
yi = np.linspace(y.min(), y.max(), 150)
Xi, Yi = np.meshgrid(xi, yi)

Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

# symmetric color scaling
vmax = np.nanmax(np.abs(Zi))
vmin = -vmax
print(vmax)
levels = np.arange(-0.3, 0.35, 0.05)

cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap="PiYG", extend="both")
ax.contour(Xi, Yi, Zi, levels=levels, colors="k", linewidths=0.6)

ax.set_title("LES - UMM $A_v$ vs $\phi$ Contour", fontsize=20)
ax.set_xlabel("Phase $\phi$", fontsize = 20)
ax.set_ylabel("$A_v$", fontsize = 20)
ax.grid(True, linestyle="--", alpha=0.3)
ax.tick_params(labelsize=18)

cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(label_map[2], fontsize = 20)
cbar.ax.tick_params(labelsize=18)

# -------------------------
# SHARED LEGEND
# -------------------------
handles = [
    mlines.Line2D([], [], color=color_map[av], lw=3, label=f"${av}$")
    for av in avs
]

fig.legend(
    handles=handles,
    loc="upper center",
    ncol=len(avs),
    frameon=False,
    title="$A_v$",
    title_fontsize=22,
    fontsize=18,
    bbox_to_anchor=(0.5, 1.14)
)

# -------------------------
# GLOBAL TITLE + LAYOUT
# -------------------------
fig.suptitle("Phase-Resolved Thrust Coefficient: $f^* = 0.5, C_T' = 2$", fontsize=24, y=1.18)

plt.tight_layout()
fig.subplots_adjust(top=0.90, wspace=0.25, hspace=0.30)

plt.savefig("combined_2x2.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
umm_les_contour_plot(
    phase_stats[phase_stats["Frequency"] == 0.5],
    x_key="Phase",
    y_key="Surge_Amplitude",
    z_key="Diff_CT_mean",   # adjust as needed
    z_label=ylabel_diff_phase,
    show_scatter = False
)
fig.suptitle("Phase-Resolved Thrust Coefficient: $f^* = 0.5$", fontsize=22, y = 1.15)
plt.savefig("f_05_contour.png", dpi = 600, bbox_inches='tight')

# %%
umm_les_contour_plot_with_diff(
    phase_stats[phase_stats["Frequency"] == 0.5],
    x_key="Phase",
    y_key="Surge_Amplitude",
    z_key="Diff_CT_mean",   # adjust as needed
    z_label=ylabel_diff_phase,
    show_scatter = False
)
fig.suptitle("Phase-Resolved Thrust Coefficient: $f^* = 0.5$", fontsize=22, y = 1.15)
plt.savefig("f_05_contour_diff.png", dpi = 600, bbox_inches='tight')

# %%
# Filter data
df = phase_stats[(phase_stats["Frequency"] == 0.5) & (phase_stats["CT_prime"] == 2)].copy()
group_cols = ["CT_prime", "Surge_Amplitude"]

# Normalize per simulation
for name, group in df.groupby(group_cols):
    idx = group.index
    df.loc[idx, "XTurb_norm"] = group["XTurb_mean"] / group["XTurb_mean"].abs().max()
    df.loc[idx, "UTurb_norm"] = group["UTurb_mean"] / group["UTurb_mean"].abs().max()
    df.loc[idx, "ATurb_norm"] = group["ATurb_mean"] / group["ATurb_mean"].abs().max()  # normalized acceleration

# ---- Set up figure and axes: 2 subplots side-by-side ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={'width_ratios':[1,1]})

# ---- Subplot 1: XTurb vs CT error colored by UTurb ----
cmap = plt.get_cmap("viridis_r")
vmin = df["UTurb_norm"].min()
vmax = df["UTurb_norm"].max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

for name, group in df.groupby(group_cols):
    group = group.sort_values("Phase")
    uturb = np.append(group["UTurb_norm"].values, group["UTurb_norm"].values[0])
    ct_err = np.append(group["Diff_UDisk_mean"].values, group["Diff_UDisk_mean"].values[0])
    xturb = np.append(group["XTurb_mean"].values, group["XTurb_mean"].values[0])
    points = np.array([xturb, ct_err]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(uturb)
    lc.set_linewidth(2.5)
    ax1.add_collection(lc)

ax1.set_xlabel("Turbine Position $x$", fontsize=16)
ax1.set_ylabel(r"$\langle u_{d_{\mathrm{LES}}}/u_{d_{\mathrm{LES,0}}} - u_{d_{\mathrm{ADM}}}/u_{d_{\mathrm{ADM,0}}} \rangle$", fontsize=16)
ax1.tick_params(labelsize=14)
ax1.axhline(0, linestyle="--", linewidth=2, color="k", alpha = 0.5)
ax1.grid(True)
ax1.autoscale()

# Colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1)
cbar.set_label("Turbine Velocity $\\frac{\dot{x}(t)}{A_v}$", fontsize=14)
cbar.ax.tick_params(labelsize=14)

# ---- Subplot 2: UTurb vs CT error colored by normalized acceleration ----
cmap2 = plt.get_cmap("plasma_r")
vmin2 = df["ATurb_norm"].min()
vmax2 = df["ATurb_norm"].max()
norm2 = mpl.colors.Normalize(vmin=vmin2, vmax=vmax2)

for name, group in df.groupby(group_cols):
    uturb = np.append(group["UTurb_mean"].values, group["UTurb_mean"].values[0])
    ct_err = np.append(group["Diff_UDisk_mean"].values, group["Diff_UDisk_mean"].values[0])
    aturb = np.append(group["ATurb_norm"].values, group["ATurb_norm"].values[0])
    points = np.array([uturb, ct_err]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap2, norm=norm2)
    lc.set_array(aturb)
    lc.set_linewidth(2.5)
    ax2.add_collection(lc)

ax2.set_xlabel("Turbine Velocity $\dot{x}$", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.grid(True)
ax2.autoscale()
ax2.axhline(0, linestyle="--", linewidth=2, color="k", alpha = 0.5)

# Colorbar
sm2 = mpl.cm.ScalarMappable(cmap=cmap2, norm=norm2)
sm2.set_array([])
cbar2 = fig.colorbar(sm2, ax=ax2)
cbar2.set_label("Normalized Turbine Acceleration $\\frac{\ddot{x}(t)}{A_a}$", fontsize=14)
cbar2.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig("f_05_ud_errs.png", dpi = 600, bbox_inches='tight')

# %%
# Filter data
df = phase_stats[(phase_stats["Frequency"] == 0.5) & (phase_stats["CT_prime"] == 2)].copy()
group_cols = ["CT_prime", "Surge_Amplitude"]

# Normalize per simulation
for name, group in df.groupby(group_cols):
    idx = group.index
    df.loc[idx, "XTurb_norm"] = group["XTurb_mean"] / group["XTurb_mean"].abs().max()
    df.loc[idx, "UTurb_norm"] = group["UTurb_mean"] / group["UTurb_mean"].abs().max()
    df.loc[idx, "ATurb_norm"] = group["ATurb_mean"] / group["ATurb_mean"].abs().max()  # normalized acceleration

# ---- Set up figure and axes: 2 subplots side-by-side ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={'width_ratios':[1,1]})

# ---- Subplot 1: XTurb vs CT error colored by UTurb ----
cmap = plt.get_cmap("viridis_r")
vmin = df["UTurb_norm"].min()
vmax = df["UTurb_norm"].max()
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

for name, group in df.groupby(group_cols):
    group = group.sort_values("Phase")
    uturb = np.append(group["UTurb_norm"].values, group["UTurb_norm"].values[0])
    ct_err = np.append(group["Diff_UDisk_mean"].values, group["Diff_UDisk_mean"].values[0])
    xturb = np.append(group["XTurb_mean"].values, group["XTurb_mean"].values[0])
    points = np.array([xturb, ct_err]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(uturb)
    lc.set_linewidth(2.5)
    ax1.add_collection(lc)

ax1.set_xlabel("Turbine Position $x$", fontsize=16)
ax1.set_ylabel(r"$\langle u_{d_{\mathrm{LES}}}/u_{d_{\mathrm{LES,0}}} - u_{d_{\mathrm{ADM}}}/u_{d_{\mathrm{ADM,0}}} \rangle$", fontsize=16)
ax1.tick_params(labelsize=14)
ax1.axhline(0, linestyle="--", linewidth=2, color="k", alpha = 0.5)
ax1.grid(True)
ax1.autoscale()

# ---- Subplot 2: UTurb vs CT error colored by normalized acceleration ----

for name, group in df.groupby(group_cols):
    group = group.sort_values("Phase")
    uturb = np.append(group["UTurb_norm"].values, group["UTurb_norm"].values[0])
    ct_err = np.append(group["Diff_CT_mean"].values, group["Diff_CT_mean"].values[0])
    xturb = np.append(group["XTurb_mean"].values, group["XTurb_mean"].values[0])
    points = np.array([xturb, ct_err]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(uturb)
    lc.set_linewidth(2.5)
    ax2.add_collection(lc)

ax2.set_xlabel("Turbine Position $x$", fontsize=16)
ax2.set_ylabel(r"$\langle C_{T_{\mathrm{LES}}}/C_{T_{\mathrm{LES,0}}} - C_{T_{\mathrm{ADM}}}/C_{T_{\mathrm{ADM,0}}} \rangle$", fontsize=16)
ax2.tick_params(labelsize=14)
ax2.axhline(0, linestyle="--", linewidth=2, color="k", alpha = 0.5)
ax2.grid(True)
ax2.autoscale()

# Colorbar
# Colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2)
cbar.set_label("Turbine Velocity $\\frac{\dot{x}(t)}{A_v}$", fontsize=14)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig("f_05_ud_ct_errs.png", dpi = 600, bbox_inches='tight')

# %%
# -----------------------------
# Filter and normalize data
# -----------------------------
# Axis 1: varying amplitude
df1 = phase_stats[(phase_stats["Frequency"] == 0.5) & 
                  (phase_stats["CT_prime"] == 2)].copy()
df1 = df1[(df1["Surge_Amplitude"] > 0.05) & (df1["Surge_Amplitude"] < 0.6)]
group_cols1 = ["CT_prime", "Surge_Amplitude"]

for name, group in df1.groupby(group_cols1):
    idx = group.index
    df1.loc[idx, "XTurb_norm"] = group["XTurb_mean"] / group["XTurb_mean"].abs().max()
    df1.loc[idx, "UTurb_norm"] = group["UTurb_mean"] / group["UTurb_mean"].abs().max()
    df1.loc[idx, "ATurb_norm"] = group["ATurb_mean"] / group["ATurb_mean"].abs().max()

# Axis 2: varying frequency
df2 = phase_stats[(phase_stats["Surge_Amplitude"] == 0.3) & 
                  (phase_stats["CT_prime"] == 2)].copy()
df2 = df2[(df2["Frequency"] > 0.1) & (df2["Frequency"] < 2)]
group_cols2 = ["CT_prime", "Frequency"]

for name, group in df2.groupby(group_cols2):
    idx = group.index
    df2.loc[idx, "XTurb_norm"] = group["XTurb_mean"] / group["XTurb_mean"].abs().max()
    df2.loc[idx, "UTurb_norm"] = group["UTurb_mean"] / group["UTurb_mean"].abs().max()
    df2.loc[idx, "ATurb_norm"] = group["ATurb_mean"] / group["ATurb_mean"].abs().max()

# -----------------------------
# Figure setup
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), sharey=True)

# -----------------------------
# Subplot 1
# -----------------------------
cmap1 = plt.get_cmap("viridis_r")
norm1 = mpl.colors.Normalize(vmin=df1["UTurb_norm"].min(), vmax=df1["UTurb_norm"].max())

for name, group in df1.groupby(group_cols1):
    group = group.sort_values("Phase")
    uturb = np.append(group["UTurb_norm"].values, group["UTurb_norm"].values[0])
    ct_err = np.append(group["Diff_CT_mean"].values, group["Diff_CT_mean"].values[0])
    xturb = np.append(group["XTurb_norm"].values, group["XTurb_norm"].values[0])
    points = np.array([xturb, ct_err]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap1, norm=norm1)
    lc.set_array(uturb)
    lc.set_linewidth(3)
    ax1.add_collection(lc)

ax1.set_xlabel("$x(t)/A_x$", fontsize=22)
ax1.set_ylabel(ylabel_diff_phase, fontsize=22)
ax1.tick_params(labelsize=20)
ax1.axhline(0, linestyle="--", linewidth=2, color="k", alpha=0.5)
ax1.grid(True)
ax1.set_title(r"$f^* = 0.5$", fontsize=22)
# After adding the lines, before arrows/colorbars
ax1.set_xlim(-1.05, 1.05)  # or whatever normalized range your XTurb_norm uses
ax2.set_xlim(-1.05, 1.05)
# Optionally also set y-limits so arrows and lines are centered
ax1.set_ylim(-0.3, 0.3)  # or whatever normalized range your XTurb_norm uses
ax2.set_ylim(-0.3, 0.3)
# Arrow
ax1.annotate('', xy=(0, 0.3), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', linewidth=3, color='k'))
ax1.text(0.05, 0.155, 'Increasing\n$A_v$', fontsize=21, rotation=90, va='center', multialignment='center')

# -----------------------------
# Subplot 2
# -----------------------------
cmap2 = plt.get_cmap("viridis_r")
norm2 = mpl.colors.Normalize(vmin=df2["UTurb_norm"].min(), vmax=df2["UTurb_norm"].max())

for name, group in df2.groupby(group_cols2):
    group = group.sort_values("Phase")
    uturb = np.append(group["UTurb_norm"].values, group["UTurb_norm"].values[0])
    ct_err = np.append(group["Diff_CT_mean"].values, group["Diff_CT_mean"].values[0])
    xturb = np.append(group["XTurb_norm"].values, group["XTurb_norm"].values[0])
    points = np.array([xturb, ct_err]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap2, norm=norm2)
    lc.set_array(uturb)
    lc.set_linewidth(3)
    ax2.add_collection(lc)

ax2.set_xlabel("$x(t)/A_x$", fontsize=22)
ax2.tick_params(labelsize=20)
ax2.axhline(0, linestyle="--", linewidth=2, color="k", alpha=0.5)
ax2.grid(True)
ax2.set_title(r"$A_v = 0.3$", fontsize=22)

# Arrow
ax2.annotate('', xy=(0, 0.3), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', linewidth=3, color='k'))
ax2.text(0.05, 0.155, 'Increasing\n$f^*$', fontsize=21, rotation=90, va='center', multialignment='center')

# Colorbar without resizing axes
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.1)
sm2 = mpl.cm.ScalarMappable(cmap=cmap2, norm=norm2)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, cax=cax2)
cbar2.set_label("$\\dot{x}(t)/A_v$", fontsize=22)
cbar2.ax.tick_params(labelsize=20)

xticks = np.linspace(-1, 1, 5)  # or whatever you want

ax1.set_xticks(xticks)
ax2.set_xticks(xticks)
fig.suptitle("Hysteresis Behavior in LES–UMM " + ylabel_diff_phase + " for $C_T' = 2$", fontsize=22, y=1)

plt.tight_layout()
plt.savefig("f_05_v03_ct_arrows_equalwidth.png", dpi=600, bbox_inches='tight')
plt.show()


# %% [markdown]
# # Phase Offsets

# %%
def compute_phase_offset_fft(g):
    g = g.sort_values(["Phase"])
    ut = g["UTurb_mean"] - np.mean(g["UTurb_mean"])
    ur = g["URel_mean"] - np.mean(g["URel_mean"])
    xt = g["XTurb_mean"] - np.mean(g["XTurb_mean"])
    les_ud = g["LES_UDisk_mean"] - np.mean(g["LES_UDisk_mean"])
    umm_ud = g["UMM_UDisk_mean"] - np.mean(g["UMM_UDisk_mean"])

    ut_fft = np.fft.fft(ut)
    ur_fft = np.fft.fft(ur)
    xt_fft = np.fft.fft(xt)
    les_ud_fft = np.fft.fft(les_ud)
    umm_ud_fft = np.fft.fft(umm_ud)

    phase_ut = np.angle(ut_fft[1])
    phase_ur = np.angle(ur_fft[1])
    phase_xt = np.angle(xt_fft[1])
    phase_les_ud = np.angle(les_ud_fft[1])
    phase_umm_ud = np.angle(umm_ud_fft[1])

    les_phase_offset = phase_les_ud - phase_ur
    umm_phase_offset = phase_umm_ud - phase_ur
    les_phase_offset = (les_phase_offset + np.pi) % (2*np.pi) - np.pi
    umm_phase_offset = (umm_phase_offset + np.pi) % (2*np.pi) - np.pi
    les_phase_deg = np.degrees(les_phase_offset)
    umm_phase_deg = np.degrees(umm_phase_offset)
    return les_phase_deg, umm_phase_deg

# Columns to compute phase offset for
period_group_cols = ["CT_prime", "Surge_Amplitude", "Frequency"]

results = []
for group_vals, df_group in phase_stats.groupby(period_group_cols):
    row = dict(zip(period_group_cols, group_vals))
    les_offset, umm_offset = compute_phase_offset_fft(df_group)
    row["LES_UDisk_Offset"] = les_offset
    row["UMM_UDisk_Offset"] = umm_offset
    results.append(row)
phase_offsets_df = pd.DataFrame(results)

# %%
g = sns.relplot(
    data=phase_offsets_df,
    x="Frequency",
    y="LES_UDisk_Offset",
    hue="Surge_Amplitude",
    col="CT_prime",
    kind="scatter",
    palette=sns.color_palette("viridis", 8),
    height=4,
    aspect=1,
    s=80,
    edgecolor="k",
)

# %% [markdown]
# # Mean, Min, and Max for each $C_T'$, $f^*$, and $A_S^*$

# %%
period_group_cols = ["CT_prime", "Surge_Amplitude", "Frequency"]

mean_cols = [
    "LES_CP_mean", "LES_CT_mean", "LES_an_T_mean", "LES_an_G_mean", "LES_UDisk_mean", "LES_Power_mean",
    "UMM_CP_mean", "UMM_CT_mean", "UMM_an_T_mean", "UMM_an_G_mean", "UMM_UDisk_mean", "UMM_Power_mean",
]
overall_stats = phase_stats.groupby(period_group_cols)[mean_cols].agg(["min", "max", "mean"]).reset_index()

def clean_mean_column_name(col):
    if isinstance(col, tuple):
        # take only the first part up to _mean/_max/_min
        base = col[0]
        # remove trailing _mean if present
        if base.endswith("_mean"):
            base = base[:-5]  # remove '_mean'
        # add the second part if not empty
        if col[1]:
            base = f"{base}_{col[1]}"
        return base
    else:
        return col
overall_stats.columns = [clean_mean_column_name(col) for col in overall_stats.columns]

for col in mean_cols:
    if col.startswith("LES_"):
        col = col[:-5]  # remove '_mean'
        LES_col = col
        UMM_col = LES_col.replace("LES","UMM")
        base_name = LES_col.replace("LES_", "")
        # get extreems
        for extreme in ["_min", "_max"]:
            LES_ext_col = LES_col + extreme
            UMM_ext_col = UMM_col + extreme
            diff_col = f"Diff_{base_name}{extreme}"
            overall_stats[diff_col] = overall_stats[LES_ext_col] - overall_stats[UMM_ext_col]
        # get means
        diff_mean_col = f"Diff_{base_name}_mean"
        overall_stats[diff_mean_col] = overall_stats[LES_col + "_mean"] - overall_stats[UMM_col + "_mean"]

# Continuous palette for hue (Frequency)
palette = sns.color_palette("viridis_r", n_colors=overall_stats["Frequency"].nunique())

# Map Frequency to color
freq_colors = dict(zip(sorted(overall_stats["Frequency"].unique()), palette))


# %%
umm_les_contour_plot(
    overall_stats,
    x_key="Surge_Amplitude",
    y_key="Frequency",
    z_key="Diff_CT_max",   # adjust as needed
    z_label=ylabel_max,
    show_scatter = True
)
plt.savefig("max_ct_contour.png", dpi = 600,  bbox_inches='tight')

# %%
umm_les_contour_plot(
    overall_stats,
    x_key="Surge_Amplitude",
    y_key="Frequency",
    z_key="Diff_CT_min",   # adjust as needed
    z_label=ylabel_min,
    show_scatter = True
)
plt.savefig("min_ct_contour.png", dpi = 600, bbox_inches='tight')

# %%
sns.set_style("ticks")
sns.set_context("notebook")

# --- Color mapping ---
palette = sns.color_palette("viridis_r", n_colors=overall_stats["Frequency"].nunique())
freq_values = sorted(overall_stats["Frequency"].unique())
freq_colors = dict(zip(freq_values, palette))

# --- Select CT' values (2 columns) ---
CT_vals = sorted(overall_stats["CT_prime"].unique())  # make sure this is length 2

# --- Setup figure ---
fig, axes = plt.subplots(3, len(CT_vals), figsize=(14, 8), sharex=True)

# --- Labels ---
metrics = ["Diff_CT_max", "Diff_CT_min", "Diff_CT_mean"]

ylabels = [ylabel_max, ylabel_min, ylabel_mean]

# --- Plotting loop ---
for row, metric in enumerate(metrics):
    for col, ct in enumerate(CT_vals):
        ax = axes[row, col]
        
        df = overall_stats[(overall_stats["CT_prime"] == ct) & (overall_stats["Surge_Amplitude"] > 0)]
        
        sns.lineplot(
            data=df,
            x="Surge_Amplitude",
            y=metric,
            hue="Frequency",
            palette=freq_colors,
            linewidth=2.5,
            legend=False,
            ax=ax
        )

        sns.scatterplot(
            data=df,
            x="Surge_Amplitude",
            y=metric,
            hue="Frequency",
            palette=freq_colors,
            s=80,
            edgecolor="k",
            linewidth=1.2,
            legend=False,
            ax=ax
        )
        
        # Column titles (top row only)
        if row == 0:
            ax.set_title(rf"$C_T' = {ct}$", fontsize=22)
        
        # Row labels (left column only)
        if col == 0:
            ax.set_ylabel(ylabels[row], fontsize=22)
        else:
            ax.set_ylabel("")
        
        # X labels (bottom row only)
        if row == 2:
            ax.set_xlabel(r"$A_v$", fontsize=24)
        else:
            ax.set_xlabel("")
        
        ax.tick_params(labelsize=20)
        ax.grid(True, linestyle='--', alpha=0.3)


# --- Global legend ---
handles = [
    plt.Line2D(
        [], [],
        color=freq_colors[f],
        linewidth=4  # match your plot lines
    )
    for f in freq_values[1:]
]

labels = [str(f) for f in freq_values[1:]]

fig.legend(
    handles, labels,
    title=r"$f^*$",
    title_fontsize=24,
    fontsize=22,
    loc='upper center',
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.0)
)

y_limits = [
    (-0.1, 0.5),   # max row
    (-0.5, 0.1),    # min row
    (-0.3, 0.3),   # mean row
]
for row in range(3):
    for col in range(len(CT_vals)):
        axes[row, col].set_ylim(*y_limits[row])
        axes[row, col].axhline(0, color='gray', alpha=0.75, zorder=0, linestyle = "--", lw = 2.5)
        # axes[row, col].text(0.515, 0.04, r'$\overline{\Delta C^*_T} = 0$', color='0.3', fontsize=15, va='center')

fig.suptitle("LES-UMM Differences in Thrust Coefficient Statistics", fontsize=24, y = 1.03)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("combined_ct_mean_max_min.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
# --- Color mapping ---
palette = sns.color_palette("viridis_r", n_colors=overall_stats["Frequency"].nunique())
freq_values = sorted(overall_stats["Frequency"].unique())
freq_colors = dict(zip(freq_values, palette))

# --- Select CT' values (2 columns) ---
CT_vals = sorted(overall_stats["CT_prime"].unique())  # make sure this is length 2

# --- Setup figure ---
fig, axes = plt.subplots(3, len(CT_vals), figsize=(14, 8), sharex=True)

# --- Labels ---
metrics = ["Diff_CP_max", "Diff_CP_min", "Diff_CP_mean"]

ylabels = np.strings.replace([ylabel_max, ylabel_min, ylabel_mean], 'C_T', 'C_P')

# --- Plotting loop ---
for row, metric in enumerate(metrics):
    for col, ct in enumerate(CT_vals):
        ax = axes[row, col]
        
        df = overall_stats[(overall_stats["CT_prime"] == ct) & (overall_stats["Surge_Amplitude"] > 0)]
        
        sns.lineplot(
            data=df,
            x="Surge_Amplitude",
            y=metric,
            hue="Frequency",
            palette=freq_colors,
            linewidth=2.5,
            legend=False,
            ax=ax
        )

        sns.scatterplot(
            data=df,
            x="Surge_Amplitude",
            y=metric,
            hue="Frequency",
            palette=freq_colors,
            s=80,
            edgecolor="k",
            linewidth=1.2,
            legend=False,
            ax=ax
        )
        
        # Column titles (top row only)
        if row == 0:
            ax.set_title(rf"$C_T' = {ct}$", fontsize=22)
        
        # Row labels (left column only)
        if col == 0:
            ax.set_ylabel(ylabels[row], fontsize=22)
        else:
            ax.set_ylabel("")
        
        # X labels (bottom row only)
        if row == 2:
            ax.set_xlabel(r"$A_v$", fontsize=24)
        else:
            ax.set_xlabel("")
        
        ax.tick_params(labelsize=20)
        ax.grid(True, linestyle='--', alpha=0.3)


# --- Global legend ---
handles = [
    plt.Line2D(
        [], [],
        color=freq_colors[f],
        linewidth=4  # match your plot lines
    )
    for f in freq_values[1:]
]

labels = [str(f) for f in freq_values[1:]]

fig.legend(
    handles, labels,
    title=r"$f^*$",
    title_fontsize=24,
    fontsize=22,
    loc='upper center',
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.0)
)

y_limits = [
    (-0.05, 0.75),   # max row
    (-0.5, 0.3),    # min row
    (-0.25, 0.55),   # mean row
]
for row in range(3):
    for col in range(len(CT_vals)):
        axes[row, col].set_ylim(*y_limits[row])
        axes[row, col].axhline(0, color='gray', alpha=0.75, zorder=0, linestyle = "--", lw = 2.5)
        # axes[row, col].text(0.515, 0.04, r'$\overline{\Delta C^*_T} = 0$', color='0.3', fontsize=15, va='center')

fig.suptitle("LES-UMM Differences in Thrust Coefficient Statistics", fontsize=24, y = 1.03)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("combined_ct_mean_max_min.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
overall_stats.keys()
overall_stats["Diff_CT_max_percent"] = overall_stats["Diff_CT_max"] / overall_stats["LES_CT_max"] * 100
overall_stats["Diff_CT_min_percent"] = overall_stats["Diff_CT_min"] / overall_stats["LES_CT_min"] * 100
overall_stats["Diff_CT_mean_percent"] = overall_stats["Diff_CT_mean"] / overall_stats["LES_CT_mean"] * 100

overall_stats["Diff_CP_max_percent"] = overall_stats["Diff_CP_max"] / overall_stats["LES_CP_max"] * 100
overall_stats["Diff_CP_min_percent"] = overall_stats["Diff_CP_min"] / overall_stats["LES_CP_min"] * 100
overall_stats["Diff_CP_mean_percent"] = overall_stats["Diff_CP_mean"] / overall_stats["LES_CP_mean"] * 100

# %%
# --- Color mapping (MATCHED) ---
palette = sns.color_palette("viridis_r", n_colors=overall_stats["Frequency"].nunique())
freq_values = sorted(overall_stats["Frequency"].unique())
freq_colors = dict(zip(freq_values, palette))

# --- Plot ---
g = sns.relplot(
    data=overall_stats[overall_stats["Surge_Amplitude"] > 0],
    x="Surge_Amplitude",
    y="Diff_CT_max_percent",
    hue="Frequency",
    col="CT_prime",
    kind="line",
    palette=freq_colors,   
    lw=3,
    # height=4,
    aspect=1.3,
    legend=False,
    zorder = 0
)

# --- Scatter + axis styling ---
for ax, ct in zip(g.axes.flat, sorted(overall_stats["CT_prime"].unique())):
    sub = overall_stats[
        (overall_stats["CT_prime"] == ct) &
        (overall_stats["Surge_Amplitude"] > 0)
    ]

    sns.scatterplot(
        data=sub,
        x="Surge_Amplitude",
        y="Diff_CT_max_percent",
        hue="Frequency",
        palette=freq_colors,
        legend=False,
        ax=ax,
        s=90,                  # match size
        edgecolor="k",         # match outline
    )

    # --- Titles ---
    ax.set_title(rf"$C_T' = {ct}$", fontsize=22)

    # --- Labels ---
    ax.set_xlabel(r"$A_v$", fontsize=24)
    ax.set_ylabel(ylabel_max + " [%]", fontsize=22)

    # --- Ticks / grid ---
    ax.tick_params(labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- Reference lines ---
    ax.axvline(x=0.25, linestyle="--", color="gray", alpha=0.7, linewidth=2)
    ax.set_ylim(0, 17.5)


g.fig.legend(
    handles, labels,
    title=r"$f^*$",
    title_fontsize=24,
    fontsize=22,
    loc='upper center',
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.05)
)

# --- Title / layout ---
g.fig.suptitle(
    "Percent Difference in " + ylabel_max + " (LES–UMM)",
    fontsize=24,
    y=1.12
)

g.fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("max_ct_percent_faceted.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
# --- Color mapping (MATCHED) ---
palette = sns.color_palette("viridis_r", n_colors=overall_stats["Frequency"].nunique())
freq_values = sorted(overall_stats["Frequency"].unique())
freq_colors = dict(zip(freq_values, palette))

# --- Plot ---
g = sns.relplot(
    data=overall_stats[overall_stats["Surge_Amplitude"] > 0],
    x="Surge_Amplitude",
    y="Diff_CP_max_percent",
    hue="Frequency",
    col="CT_prime",
    kind="line",
    palette=freq_colors,   
    lw=3,
    # height=4,
    aspect=1.3,
    legend=False,
    zorder = 0
)

# --- Scatter + axis styling ---
for ax, ct in zip(g.axes.flat, sorted(overall_stats["CT_prime"].unique())):
    sub = overall_stats[
        (overall_stats["CT_prime"] == ct) &
        (overall_stats["Surge_Amplitude"] > 0)
    ]

    sns.scatterplot(
        data=sub,
        x="Surge_Amplitude",
        y="Diff_CP_max_percent",
        hue="Frequency",
        palette=freq_colors,
        legend=False,
        ax=ax,
        s=90,                  # match size
        edgecolor="k",         # match outline
    )

    # --- Titles ---
    ax.set_title(rf"$C_T' = {ct}$", fontsize=22)

    # --- Labels ---
    ax.set_xlabel(r"$A_v$", fontsize=24)
    ax.set_ylabel(np.strings.replace(ylabel_max, "C_T", "C_P") + " [%]", fontsize=22)

    # --- Ticks / grid ---
    ax.tick_params(labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- Reference lines ---
    ax.axvline(x=0.25, linestyle="--", color="gray", alpha=0.7, linewidth=2)
    ax.set_ylim(0, 25)


g.fig.legend(
    handles, labels,
    title=r"$f^*$",
    title_fontsize=24,
    fontsize=22,
    loc='upper center',
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.05)
)

# --- Title / layout ---
g.fig.suptitle(
    r"Percent Difference in " + np.strings.replace(ylabel_max, "C_T", "C_P") + " (LES–UMM)",
    fontsize=24,
    y=1.12
)

g.fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("max_cp_percent_faceted.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
ct_prime_target = 2

df = overall_stats[
    (overall_stats["Surge_Amplitude"] > 0) &
    (overall_stats["CT_prime"] == ct_prime_target)
].copy()

df_long = df.melt(
    id_vars=["Surge_Amplitude", "Frequency", "CT_prime"],
    value_vars=["Diff_CT_max_percent", "Diff_CP_max_percent"],
    var_name="Metric",
    value_name="Value"
)

# --- clean mapping ---
metric_map = {
    "Diff_CT_max_percent": "C_T",
    "Diff_CP_max_percent": "C_P"
}

ylabel_map = {
    "C_T": ylabel_max,
    "C_P": ylabel_max.replace("C_T", "C_P")
}

df_long["Metric"] = df_long["Metric"].map(metric_map)

g = sns.relplot(
    data=df_long,
    x="Surge_Amplitude",
    y="Value",
    hue="Frequency",
    col="Metric",
    kind="line",
    palette=freq_colors,
    lw=3,
    aspect=1.4,
    legend=False,
    zorder=0,
    facet_kws={"sharey": False, "sharex": True},
)
for ax, metric in zip(g.axes.flat, ["C_T", "C_P"]):

    sub = df_long[df_long["Metric"] == metric]

    sns.scatterplot(
        data=sub,
        x="Surge_Amplitude",
        y="Value",
        hue="Frequency",
        palette=freq_colors,
        legend=False,
        ax=ax,
        s=90,
        edgecolor="k",
    )

    ax.set_title(rf"${metric}$", fontsize=22)
    ax.set_xlabel(r"$A_v$", fontsize=24)
    ax.set_ylabel(ylabel_map[metric] + " [%]", fontsize=22)

    ax.tick_params(labelsize=20)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.axvline(x=0.25, linestyle="--", color="gray", alpha=0.7, linewidth=2)

    ax.set_ylim(0, 30)

g.fig.legend(
    handles, labels,
    title=r"$f^*$",
    title_fontsize=24,
    fontsize=22,
    loc='upper center',
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.05)
)

g.fig.suptitle(
    r"Percent Error in Maximum $C_T$ and $C_P$ (LES–UMM) for $C_T' = 2$",
    fontsize=24,
    y=1.12
)

g.fig.tight_layout(rect=[0, 0, 1, 0.95])

# %%
df_plot = overall_stats.copy()

df_plot = df_plot[df_plot["Surge_Amplitude"] > 0]

df_plot = df_plot.melt(
    id_vars=["Surge_Amplitude", "Frequency", "CT_prime"],
    value_vars=["Diff_CT_mean_percent", "Diff_CT_max_percent"],
    var_name="Metric",
    value_name="Percent_Diff"
)

# Clean labels
df_plot["Metric"] = df_plot["Metric"].map({
    "Diff_CT_mean_percent": "Mean",
    "Diff_CT_max_percent": "Max"
})

g = sns.relplot(
    data=df_plot,
    x="Surge_Amplitude",
    y="Percent_Diff",
    hue="Frequency",
    col="CT_prime",
    row="Metric",              # <-- KEY ADDITION
    kind="line",
    palette=freq_colors,
    lw=3,
    aspect=1.5,
    legend=False,
    zorder=0,
    facet_kws={"sharey": True, "sharex": True},
)

for (row_val, col_val), ax in g.axes_dict.items():

    sub = df_plot[
        (df_plot["Metric"] == row_val) &
        (df_plot["CT_prime"] == col_val)
    ]

    sns.scatterplot(
        data=sub,
        x="Surge_Amplitude",
        y="Percent_Diff",
        hue="Frequency",
        palette=freq_colors,
        legend=False,
        ax=ax,
        s=90,
        edgecolor="k",
    )

    # Titles (only top row)
    if row_val == "Mean":
        ax.set_title(rf"$C_T' = {col_val}$", fontsize=22)
    else:
        ax.set_title("")

    # Labels
    ax.set_xlabel(r"$A_v$", fontsize=24)

    if row_val == "Mean":
        ax.set_ylabel(ylabel_mean + " [%]", fontsize=22)
    else:
        ax.set_ylabel(ylabel_max + " [%]", fontsize=22)

    # Styling
    ax.tick_params(labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(x=0.25, linestyle="--", color="gray", alpha=0.7, linewidth=2)

    g.fig.legend(
    handles, labels,
    title=r"$f^*$",
    title_fontsize=24,
    fontsize=22,
    loc='upper center',
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.03)
)

g.fig.suptitle(
    "Percent Difference in $C_T$ Metrics (LES–UMM)",
    fontsize=26,
    y=1.08
)

g.fig.tight_layout(rect=[0, 0, 1, 0.96])

# %%
np.max(np.abs(df_plot[(df_plot["Metric"] == "Mean") & (df_plot["Surge_Amplitude"] < 0.7)]["Percent_Diff"]))

# %%
df_plot = overall_stats.copy()

df_plot = df_plot[df_plot["Surge_Amplitude"] > 0]

df_plot = df_plot.melt(
    id_vars=["Surge_Amplitude", "Frequency", "CT_prime"],
    value_vars=["Diff_CP_mean_percent", "Diff_CP_max_percent"],
    var_name="Metric",
    value_name="Percent_Diff"
)

# Clean labels
df_plot["Metric"] = df_plot["Metric"].map({
    "Diff_CP_mean_percent": "Mean",
    "Diff_CP_max_percent": "Max"
})

g = sns.relplot(
    data=df_plot,
    x="Surge_Amplitude",
    y="Percent_Diff",
    hue="Frequency",
    col="CT_prime",
    row="Metric",              # <-- KEY ADDITION
    kind="line",
    palette=freq_colors,
    lw=3,
    aspect=1.5,
    legend=False,
    zorder=0,
    facet_kws={"sharey": True, "sharex": True},
)

for (row_val, col_val), ax in g.axes_dict.items():

    sub = df_plot[
        (df_plot["Metric"] == row_val) &
        (df_plot["CT_prime"] == col_val)
    ]

    sns.scatterplot(
        data=sub,
        x="Surge_Amplitude",
        y="Percent_Diff",
        hue="Frequency",
        palette=freq_colors,
        legend=False,
        ax=ax,
        s=90,
        edgecolor="k",
    )

    # Titles (only top row)
    if row_val == "Mean":
        ax.set_title(rf"$C_T' = {col_val}$", fontsize=22)
    else:
        ax.set_title("")

    # Labels
    ax.set_xlabel(r"$A_v$", fontsize=24)

    if row_val == "Mean":
        ax.set_ylabel(np.strings.replace(ylabel_mean, "C_T", "C_P") + " [%]", fontsize=22)
    else:
        ax.set_ylabel(np.strings.replace(ylabel_max, "C_T", "C_P") + " [%]", fontsize=22)

    # Styling
    ax.tick_params(labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(x=0.25, linestyle="--", color="gray", alpha=0.7, linewidth=2)

    g.fig.legend(
    handles, labels,
    title=r"$f^*$",
    title_fontsize=24,
    fontsize=22,
    loc='upper center',
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, 1.03)
)

g.fig.suptitle(
    "Percent Difference in $C_P$ Metrics (LES–UMM)",
    fontsize=26,
    y=1.08
)

g.fig.tight_layout(rect=[0, 0, 1, 0.96])

# %%
np.max(np.abs(df_plot[(df_plot["Metric"] == "Mean") & (df_plot["Surge_Amplitude"] < 0.7)]["Percent_Diff"]))

# %%
