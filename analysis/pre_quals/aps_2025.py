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
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# %%
from UnifiedMomentumModel.Momentum import UnifiedMomentum

# %%
# %matplotlib inline
plt.rcParams["text.usetex"] = True

# %%
rho, uinf, D = 1, 1, 1
dt = 0.05


# %%
def calc_an(df, ud_key, uinf_key):
    return 1 - (df[ud_key] / df[uinf_key])

def calc_ct(df, ud_key, uinf_key):
    return np.sign(df[ud_key]) * df["Local Thrust Coefficient"] * ((df[ud_key])**2 / (df[uinf_key])**2)

def calc_cp(df, uinf_key = "UInf_Ground"):  # power in PadeOps and UMM is calculated in the turbine frame of reference!
    return df.Power / (0.5 * rho * math.pi * (D/2)**2 * (df[uinf_key])**3)


# %% [markdown]
# # UMM Data

# %%
def get_tilt_vals(Ap, f, dt):
    T = 1 / f
    omega = 2 * np.pi * f
    phi = np.array([Ap * np.sin(omega * t) for t in np.arange(0, 2*T, dt)])
    phases = np.linspace(0, 1, num = int(len(phi) / 2))
    return np.deg2rad(phi), [phases[i % len(phases)] for i in range(len(phi))]

def get_uturb_vals(As, f, dt):
    T = 1 / f
    omega = 2 * np.pi * f
    uturb = np.array([As * np.cos(omega * t) for t in np.arange(0, 2*T, dt)])
    dx = np.array([(As / omega) * np.sin(omega * t) for t in np.arange(0, 2*T, dt)])
    return uturb, dx


# %%
Ap_vals = np.array([i * 2 for i in range(1, 10)])
As_vals = np.array([i * 0.1 for i in range(1, 10)])
f_vals = np.array([i * 0.2 for i in range(1, 5)])

# %%
Ax_vals = []
for a in As_vals:
    for f in f_vals:
        omega = (2 * np.pi * f)
        if omega != 0:
            Ax_vals.append((a, f, a / omega))
Ax_vals = np.array(Ax_vals)

# %%
as_vals_unzipped, f_vals_unzipped, Ax_vals_unzipped = zip(*Ax_vals)

# %%
sns.scatterplot(x = as_vals_unzipped, y = Ax_vals_unzipped, hue = f_vals_unzipped, palette="tab20", legend="full")
plt.axhline(0.4, color='black', linestyle='--', linewidth=2)

# %%
model = UnifiedMomentum()

# %%
pitch_umm_vals = []
for ct in [1.33, 2]:
    umm_stationary = model(ct, yaw = 0.0, tilt = 0.0)
    umm_an_stationary = umm_stationary.an
    umm_Cp_stationary = umm_stationary.Cp
    umm_Ct_stationary = umm_stationary.Ct
    for a in Ap_vals:
        for f in f_vals:
            tilts, phases = get_tilt_vals(a, f, dt)
            for (i, tilt) in enumerate(tilts):
                p = phases[i]
                tilted_vals = model(ct, yaw = 0.0, tilt = tilt)
                vals = (ct, a, a, tilt, f, 0, tilted_vals.an, tilted_vals.an / umm_an_stationary, tilted_vals.Ct, tilted_vals.Ct / umm_Ct_stationary, tilted_vals.Cp, tilted_vals.Cp / umm_Cp_stationary, p)
                pitch_umm_vals.append(vals)
                
pitch_umm_vals = np.array(pitch_umm_vals)

# %%
df_umm_pitch = pd.DataFrame(pitch_umm_vals, columns=["Local Thrust Coefficient", "Amplitude", "Distance", "Tilt", "Frequency", "UTurb ", "an_Turb", "an_Turb_normalized", "Ct_Turb", "Ct_Turb_normalized", "Cp_Turb", "Cp_Turb_normalized", "Phase_Rounded"])

# %%
surge_umm_vals = []
for ct in [1.33, 2]:
    umm_stationary = model(ct, yaw = 0.0, tilt = 0.0)
    umm_an_stationary = umm_stationary.an
    umm_Cp_stationary = umm_stationary.Cp
    umm_Ct_stationary = umm_stationary.Ct
    for a in As_vals:
        a = np.round(a, decimals= 1)
        for f in f_vals:
            uturb, dx = get_uturb_vals(a, f, dt)
            for (i, ut) in enumerate(uturb):
                x = dx[i]
                p = np.arctan2(2 * np.pi * f * x, ut) 
                p = np.mod(p, 2*np.pi) / (2*np.pi)
                rp = np.round(p, decimals = 2)
                uinf_t = 1 - ut
                Cp = umm_Cp_stationary * (uinf_t**3)
                Ct = umm_Ct_stationary * (uinf_t**2)
                surge_umm_vals.append((ct, a, a / (2 * np.pi * f), 0, f, ut, umm_an_stationary, umm_an_stationary / umm_an_stationary, Ct, Ct / umm_Ct_stationary, Cp, Cp / umm_Cp_stationary, p))
surge_umm_vals = np.array(surge_umm_vals)

# %%
df_umm_surge = pd.DataFrame(surge_umm_vals, columns=["Local Thrust Coefficient", "Amplitude", "Distance", "Tilt", "Frequency", "UTurb", "an_Turb", "an_Turb_normalized", "Ct_Turb", "Ct_Turb_normalized", "Cp_Turb", "Cp_Turb_normalized", "Phase_Rounded"])

# %%
df_umm_surge["Movement"] = "Surge"
df_umm_pitch["Movement"] = "Pitch"
df_umm = pd.concat([df_umm_surge, df_umm_pitch], ignore_index=True)

# %%
df_umm["UInf_Turb"] = (uinf - df_umm["UTurb"]) * np.cos(df_umm["Tilt"])
df_umm["UDisk_Turb"] = (1 - df_umm["an_Turb"]) * df_umm["UInf_Turb"]
df_umm["UDisk_Ground"] = df_umm["UDisk_Turb"] + df_umm["UTurb"]

# %%
df_avg_umm = df_umm.groupby(["Frequency", "Amplitude", "Movement", "Local Thrust Coefficient"], as_index=False).agg({
    "an_Turb": "mean",
    "an_Turb_normalized": "mean",
    "Ct_Turb": "mean",
    "Ct_Turb_normalized": "mean",
    "Cp_Turb": "mean",
    "Cp_Turb_normalized": "mean",
})
df_avg_umm["stat"] = "mean"

# %%
df_std_umm = df_umm.groupby(["Frequency", "Amplitude", "Movement", "Local Thrust Coefficient"], as_index=False).agg({
    "an_Turb": "std",
    "an_Turb_normalized": "std",
    "Ct_Turb": "std",
    "Ct_Turb_normalized": "std",
    "Cp_Turb": "std",
    "Cp_Turb_normalized": "std",
})
df_std_umm["stat"] = "std"

# %%
df_stat_umm = pd.concat([df_avg_umm, df_std_umm], ignore_index=True)
df_stat_umm

# %%
label_size = 22
title_size = 24

freqs = sorted(df_avg_umm["Frequency"].unique())
freqs = np.round(freqs, decimals=2)
palette = sns.color_palette("GnBu_d", n_colors=len(freqs))

# 2) Create the FacetGrid / relplot for the UMM lines (sharing y only)
g = sns.relplot(
    data=df_stat_umm,
    x="Amplitude",
    y="Cp_Turb",
    hue="Frequency",
    style = "Local Thrust Coefficient",
    col = "stat",
    row="Movement",
    kind="scatter",
    palette=palette,
    height=4,
    aspect=1.2,
    facet_kws={"sharey": True, "sharex": False},
    zorder = 10,
    s = 120
)

# %% [markdown]
# # LES Data

# %%
df_les = pd.read_csv("/Users/sky/src/HowlandLab/data/sim_16_all_runs_data_points_11_25.csv")
df_les = df_les.dropna()
df_les = df_les[((df_les["Thrust Coefficient"] != 1.66) & ((df_les["Frequency"] > 0.1) | (df_les["Frequency"] == 0.0)) & (df_les["Frequency"] < 1))]
df_les = df_les[(((df_les["Movement"] == "Pitch") & (df_les["Amplitude"] < 20)) | ((df_les["Movement"] == "Surge") & (df_les["Amplitude"] < 1)) | (df_les["Amplitude"] == 0.0))]

# %%
df_les = df_les.rename(columns={'UDisk': 'UDisk_Turb', 'Thrust Coefficient': 'Local Thrust Coefficient'}) # disk velocity in the turbine frame of reference
df_les["UDisk_Ground"] = df_les["UDisk_Turb"] + df_les["UTurb"] # disk velocity in the ground frame of reference

# %%
df_les["UInf_Turb"] = (uinf - df_les["UTurb"]) * np.cos(df_les["Tilt"])
df_les["UInf_Ground"] = uinf * np.cos(df_les["Tilt"])

# %%
df_les["an_Turb"] = calc_an(df_les, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb")
df_les["Ct_Turb"] = calc_ct(df_les, ud_key = "UDisk_Turb", uinf_key = "UInf_Ground")
df_les["Cp_Turb"] = calc_cp(df_les, uinf_key = "UInf_Ground")

# %%
df_les["Phase"] = np.arctan2(2 * np.pi * df_les["Frequency"] * df_les["DeltaX"], df_les["UTurb"]) 
df_les["Phase"] = np.mod(df_les["Phase"], 2*np.pi) / (2*np.pi)
df_les["Phase_Shifted"] = (df_les["Phase"] + 0.5) % 1
df_les["Phase_Rounded"] = np.round(df_les["Phase"], decimals = 2)

# %%
df_avg_les = df_les.groupby(["Frequency", "Amplitude", "Movement", "Local Thrust Coefficient"], as_index=False).agg({
    "an_Turb": "mean",
    "Ct_Turb": "mean",
    "Cp_Turb": "mean",
    "UDisk_Turb" : "mean",
    "UDisk_Ground": "mean",
})
df_avg_les["an_Turb_normalized"] = np.nan
df_avg_les["Ct_Turb_normalized"] = np.nan
df_avg_les["Cp_Turb_normalized"] = np.nan
les_an_stationary_ct_133 = df_avg_les[((df_avg_les["Movement"] == "Stationary") & (df_avg_les["Local Thrust Coefficient"] == 1.33))]["an_Turb"]
les_an_stationary_ct_200 = df_avg_les[((df_avg_les["Movement"] == "Stationary") & (df_avg_les["Local Thrust Coefficient"] == 2.00))]["an_Turb"]

les_ct_stationary_ct_133 = df_avg_les[((df_avg_les["Movement"] == "Stationary") & (df_avg_les["Local Thrust Coefficient"] == 1.33))]["Ct_Turb"]
les_ct_stationary_ct_200 = df_avg_les[((df_avg_les["Movement"] == "Stationary") & (df_avg_les["Local Thrust Coefficient"] == 2.00))]["Ct_Turb"]

les_cp_stationary_ct_133 = df_avg_les[((df_avg_les["Movement"] == "Stationary") & (df_avg_les["Local Thrust Coefficient"] == 1.33))]["Cp_Turb"]
les_cp_stationary_ct_200 = df_avg_les[((df_avg_les["Movement"] == "Stationary") & (df_avg_les["Local Thrust Coefficient"] == 2.00))]["Cp_Turb"]
df_les = df_les[(df_les["Frequency"] != 0)]

# %%
df_phase_avg_les = df_les.groupby(["Frequency", "Amplitude", "Movement", "Local Thrust Coefficient", "Phase_Rounded"], as_index=False).agg({
    "an_Turb": "mean",
    "Phase_Shifted": "mean",
    "UTurb" : "mean",
    "UDisk_Turb" : "mean",
    "UDisk_Ground": "mean",
    "UInf_Turb" : "mean",
})

# %%
df_les["an_Turb_normalized"] = np.nan
df_les["Ct_Turb_normalized"] = np.nan
df_les["Cp_Turb_normalized"] = np.nan
df_les.loc[df_les["Local Thrust Coefficient"] == 1.33, "an_Turb_normalized"] = \
    df_les.loc[df_les["Local Thrust Coefficient"] == 1.33, "an_Turb"] / float(les_an_stationary_ct_133)

# Normalize where Ct = 2.00
df_les.loc[df_les["Local Thrust Coefficient"] == 2.00, "an_Turb_normalized"] = \
    df_les.loc[df_les["Local Thrust Coefficient"] == 2.00, "an_Turb"] / float(les_an_stationary_ct_200)

df_les.loc[df_les["Local Thrust Coefficient"] == 1.33, "Ct_Turb_normalized"] = \
    df_les.loc[df_les["Local Thrust Coefficient"] == 1.33, "Ct_Turb"] / float(les_ct_stationary_ct_133)

# Normalize where Ct = 2.00
df_les.loc[df_les["Local Thrust Coefficient"] == 2.00, "Ct_Turb_normalized"] = \
    df_les.loc[df_les["Local Thrust Coefficient"] == 2.00, "Ct_Turb"] / float(les_ct_stationary_ct_200)

df_les.loc[df_les["Local Thrust Coefficient"] == 1.33, "Cp_Turb_normalized"] = \
    df_les.loc[df_les["Local Thrust Coefficient"] == 1.33, "Cp_Turb"] / float(les_cp_stationary_ct_133)

# Normalize where Ct = 2.00
df_les.loc[df_les["Local Thrust Coefficient"] == 2.00, "Cp_Turb_normalized"] = \
    df_les.loc[df_les["Local Thrust Coefficient"] == 2.00, "Cp_Turb"] / float(les_cp_stationary_ct_200)


# %%
df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 1.33, "an_Turb_normalized"] = \
    df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 1.33, "an_Turb"] / float(les_an_stationary_ct_133)

# Normalize where Ct = 2.00
df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 2.00, "an_Turb_normalized"] = \
    df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 2.00, "an_Turb"] / float(les_an_stationary_ct_200)

df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 1.33, "Ct_Turb_normalized"] = \
    df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 1.33, "Ct_Turb"] / float(les_ct_stationary_ct_133)

# Normalize where Ct = 2.00
df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 2.00, "Ct_Turb_normalized"] = \
    df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 2.00, "Ct_Turb"] / float(les_ct_stationary_ct_200)

df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 1.33, "Cp_Turb_normalized"] = \
    df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 1.33, "Cp_Turb"] / float(les_cp_stationary_ct_133)

# Normalize where Ct = 2.00
df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 2.00, "Cp_Turb_normalized"] = \
    df_avg_les.loc[df_avg_les["Local Thrust Coefficient"] == 2.00, "Cp_Turb"] / float(les_cp_stationary_ct_200)

df_avg_les = df_avg_les[(df_avg_les["Frequency"] != 0)]

# %%
df_std_les = df_les.groupby(["Frequency", "Amplitude", "Movement", "Local Thrust Coefficient"], as_index=False).agg({
    "an_Turb": "std",
    "Ct_Turb": "std",
    "Cp_Turb": "std",
    "an_Turb_normalized": "std",
    "Ct_Turb_normalized": "std",
    "Cp_Turb_normalized": "std",
})

# %%
df_avg_les["Distance"] = 0.0
df_avg_les.loc[df_avg_les["Movement"] == "Surge", "Distance"] = \
    df_avg_les.loc[df_avg_les["Movement"] == "Surge", "Amplitude"] / (2 * np.pi * df_avg_les.loc[df_avg_les["Movement"] == "Surge", "Frequency"])
df_avg_les.loc[df_avg_les["Movement"] == "Pitch", "Distance"] = df_avg_les.loc[df_avg_les["Movement"] == "Pitch", "Amplitude"]

# %%
df_pitch_avg_les = df_avg_les[(df_avg_les["Movement"] == "Pitch")]
df_surge_avg_les = df_avg_les[(df_avg_les["Movement"] == "Surge")]

# %%
df_avg_les["stat"] = "mean"
df_std_les["stat"] = "std"
df_stat_les = pd.concat([df_avg_les, df_std_les], ignore_index=True)

# %%
label_size = 22
title_size = 24

freqs = sorted(df_avg_umm["Frequency"].unique())
freqs = np.round(freqs, decimals=2)
palette = sns.color_palette("GnBu_d", n_colors=len(freqs))

# %%
g = sns.relplot(
    data=df_stat_les[(df_stat_les["Local Thrust Coefficient"] == 1.33)].sort_values(["Movement", "Frequency"], ascending=False),
    x="Amplitude",
    y="Cp_Turb_normalized",
    hue="Frequency",
    col = "stat",
    row="Movement",
    kind="scatter",
    palette=palette,
    height=4,
    aspect=1.2,
    facet_kws={"sharey": False, "sharex": False},
    zorder = 10,
    s = 110,
    edgecolor = "k",
)

for (movement, stat), ax in g.axes_dict.items():
    sns.lineplot(
        data=df_stat_umm[
            (df_stat_umm["Local Thrust Coefficient"] == 1.33) &
            (df_stat_umm["Movement"] == movement) &
            (df_stat_umm["stat"] == stat)
        ].sort_values(["Movement", "Frequency"], ascending=False),
        x="Amplitude",
        y="Cp_Turb_normalized",
        hue="Frequency",
        palette=palette,
        ax=ax,
        legend=False,   # avoid duplicate legend entries
        linewidth = 3
    )

for ax in g.axes[0, :]:   # row 0, all columns
    ax.set_ylim(0, 2.5)

for ax in g.axes[1, :]:   # row 1, all columns
    ax.set_ylim(-0.1, 1.1)

# Correct GridSpec object
rt = 2.5 + 1.2
gs = g.fig.axes[0].get_subplotspec().get_gridspec()
gs.set_height_ratios([3, 1.5])

for ax in g.axes.flatten():
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

g.fig.set_size_inches(14, 6)
g.fig.set_dpi(300)

g._legend.remove()

# ====== Create custom legend entries ======

# LES subheaders: Amplitude and Frequency
space_header = Line2D([], [], linestyle="none", marker="", label=" ", color="none")

# UMM line legend header
umm_header = Line2D([], [], linestyle="none", marker="", label="\\textbf{UMM}", color="none")

# Your line entries (UMM)
umm_handles = [
    Line2D([], [], color=palette[i], linewidth=2.5, label=f"$f = {freqs[i]}$")
    for i in range(len(freqs))
]

# LES scatter legend header
les_header = Line2D([], [], linestyle="none", marker="", label="\\textbf{LES}", color="none")

# Your scatter entries (LES)
les_handles = [
    Line2D([], [], color=palette[i], marker='o', linestyle="none",
           markersize=10, markeredgecolor="black", label=f"$f = {freqs[i]}$")
    for i in range(len(freqs))
]
all_handles = [umm_header] + umm_handles + [space_header, les_header] + les_handles
g.fig.subplots_adjust(right=0.8)
leg = g.fig.legend(
    handles=all_handles,
    loc="center right",
    bbox_to_anchor=(1.1, 0.5),
    bbox_transform=g.fig.transFigure,
    frameon = False
)
# leg.get_title().set_fontweight('bold')
leg.get_title().set_fontsize(label_size)
leg.get_title().set_fontweight("bold")
for text in leg.get_texts():
    text.set_fontsize(label_size )

# adjusting labels and such
for (i, ax) in enumerate(g.axes.flatten()):
    if i  < 2:
        ax.set_xlabel("$A_s = U_t / U_\infty$", fontsize = label_size + 2)
    else:
        ax.set_xlabel("$A_p = \phi$", fontsize = label_size + 2)
    
    if i == 0:
        ax.set_ylabel(
            "\\textbf{Surge}\n\n$\\overline{C_P} / C_{P_{\\textrm{\\fontsize{16}{16}\\selectfont fb}}}$",
            fontsize=label_size + 2
        )
    elif i == 2:
        ax.set_ylabel(
            "\\textbf{Pitch}\n\n$\\overline{C_P}/ C_{P_{\\textrm{\\fontsize{16}{16}\\selectfont fb}}} $",
            fontsize=label_size + 2
        )
    else:
        ax.set_ylabel(
            "$\\sigma (C_P/ C_{P_{\\textrm{\\fontsize{16}{16}\\selectfont fb}}})$",
            fontsize=label_size
        )

    # Tick labels
    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)

g.set_titles(col_template=" ", row_template= " ", size=0)

plt.tight_layout()
plt.show()

# %%
# 2) Create the FacetGrid / relplot for the UMM lines (sharing y only)
g = sns.relplot(
    data=df_avg_umm[df_avg_umm["Local Thrust Coefficient"] == 1.33],
    x="Amplitude",
    y="an_Turb_normalized",
    hue="Frequency",
    style = "Frequency",
    col="Movement",
    kind="line",
    palette=palette,
    height=4,
    aspect=1.2,
    linewidth = 2.5,
    facet_kws={"sharey": True, "sharex": False},
    zorder = 10
)
g.fig.set_size_inches(14, 5)
g.fig.set_dpi(300)
g._legend.remove()
# 3) Overlay LES scatterpoints onto the corresponding axes
# IMPORTANT: do NOT pass sharex/sharey to sns.scatterplot
for (movement), ax in g.axes_dict.items():
    sns.scatterplot(
        data=df_avg_les[
            (df_avg_les["Local Thrust Coefficient"] == 1.33) &
            (df_avg_les["Movement"] == movement)
        ],
        x="Amplitude",
        y="an_Turb_normalized",
        hue="Frequency",
        palette=palette,
        ax=ax,
        legend=False,   # avoid duplicate legend entries
        s=90,
        edgecolor = "k"
    )

# 4) Final cosmetics (one legend, titles, y-limits etc.)
g.fig.subplots_adjust(top=0.8)
# g.fig.suptitle("Normalized Induction ($C_T' = 1.33$)", size = title_size + 2)

for (i, ax) in enumerate(g.axes.flatten()):
    if i == 0:
        ax.set_xlabel("Surge Amplitude, $U_t / U_\infty$", fontsize = label_size)
    else:
        ax.set_xlabel("Pitch Amplitude, $\phi$", fontsize = label_size)
    ax.set_ylabel(
        "$\\overline{a_n} / \\overline{a_{n,\\textrm{\\fontsize{16}{16}\\selectfont fixed}}}$",
        fontsize=label_size + 4
    )

    # Tick labels
    ax.tick_params(axis='x', labelsize=label_size - 2)
    ax.tick_params(axis='y', labelsize=label_size - 2)

# ====== Create custom legend entries ======

# LES subheaders: Amplitude and Frequency
space_header = Line2D([], [], linestyle="none", marker="", label=" ", color="none")

# UMM line legend header
umm_header = Line2D([], [], linestyle="none", marker="", label="\\textbf{UMM}", color="none")

# Your line entries (UMM)
umm_handles = [
    Line2D([], [], color=palette[i], linewidth=2.5, label=f"$f = {freqs[i]}$")
    for i in range(len(freqs))
]

# LES scatter legend header
les_header = Line2D([], [], linestyle="none", marker="", label="\\textbf{LES}", color="none")

# Your scatter entries (LES)
les_handles = [
    Line2D([], [], color=palette[i], marker='o', linestyle="none",
           markersize=8, markeredgecolor="black", label=f"$f = {freqs[i]}$")
    for i in range(len(freqs))
]
all_handles = [umm_header] + umm_handles + [space_header, les_header] + les_handles
g.fig.subplots_adjust(right=0.8)
leg = g.fig.legend(
    handles=all_handles,
    loc="center right",
    bbox_to_anchor=(0.9, 0.5),
    bbox_transform=g.fig.transFigure,
    frameon = False
)
# leg.get_title().set_fontweight('bold')
leg.get_title().set_fontsize(label_size)
leg.get_title().set_fontweight("bold")
for text in leg.get_texts():
    text.set_fontsize(label_size - 4)

g.set_titles(col_template=" ", size=0)

# %%
label_size = 22
title_size = 24

freqs = sorted(df_avg_umm["Frequency"].unique())
freqs = np.round(freqs, decimals=2)
palette = sns.color_palette("GnBu_d", n_colors=len(freqs))

# 2) Create the FacetGrid / relplot for the UMM lines (sharing y only)
g = sns.relplot(
    data=df_avg_umm[df_avg_umm["Local Thrust Coefficient"] == 1.33],
    x="Amplitude",
    y="Cp_Turb_normalized",
    hue="Frequency",
    style = "Frequency",
    col="Movement",
    kind="line",
    palette=palette,
    height=4,
    aspect=1.2,
    linewidth = 2.5,
    facet_kws={"sharey": True, "sharex": False},
    zorder = 10
)
g.fig.set_size_inches(14, 5)
g.fig.set_dpi(300)
g._legend.remove()
# 3) Overlay LES scatterpoints onto the corresponding axes
# IMPORTANT: do NOT pass sharex/sharey to sns.scatterplot
for (movement), ax in g.axes_dict.items():
    sns.scatterplot(
        data=df_avg_les[
            (df_avg_les["Local Thrust Coefficient"] == 1.33) &
            (df_avg_les["Movement"] == movement)
        ],
        x="Amplitude",
        y="Cp_Turb_normalized",
        hue="Frequency",
        palette=palette,
        ax=ax,
        legend=False,   # avoid duplicate legend entries
        s=90,
        edgecolor = "k",
        zorder = 0
    )

# 4) Final cosmetics (one legend, titles, y-limits etc.)
g.fig.subplots_adjust(top=0.8)
# g.fig.suptitle("Normalized Induction ($C_T' = 1.33$)", size = title_size + 2)

for (i, ax) in enumerate(g.axes.flatten()):
    if i == 0:
        ax.set_xlabel("Surge Amplitude, $U_t / U_\infty$", fontsize = label_size)
    else:
        ax.set_xlabel("Pitch Amplitude, $\phi$", fontsize = label_size)
    ax.set_ylabel(
        "$C_T / C_{T,\\textrm{\\fontsize{16}{16}\\selectfont fixed}}$",
        fontsize=label_size + 2
    )

    # Tick labels
    ax.tick_params(axis='x', labelsize=label_size - 2)
    ax.tick_params(axis='y', labelsize=label_size - 2)

# ====== Create custom legend entries ======

# LES subheaders: Amplitude and Frequency
space_header = Line2D([], [], linestyle="none", marker="", label=" ", color="none")

# UMM line legend header
umm_header = Line2D([], [], linestyle="none", marker="", label="\\textbf{UMM}", color="none")

# Your line entries (UMM)
umm_handles = [
    Line2D([], [], color=palette[i], linewidth=2.5, label=f"$f = {freqs[i]}$")
    for i in range(len(freqs))
]

# LES scatter legend header
les_header = Line2D([], [], linestyle="none", marker="", label="\\textbf{LES}", color="none")

# Your scatter entries (LES)
les_handles = [
    Line2D([], [], color=palette[i], marker='o', linestyle="none",
           markersize=8, markeredgecolor="black", label=f"$f = {freqs[i]}$")
    for i in range(len(freqs))
]
all_handles = [umm_header] + umm_handles + [space_header, les_header] + les_handles
g.fig.subplots_adjust(right=0.8)
leg = g.fig.legend(
    handles=all_handles,
    loc="center right",
    bbox_to_anchor=(0.9, 0.5),
    bbox_transform=g.fig.transFigure,
    frameon = False
)
# leg.get_title().set_fontweight('bold')
leg.get_title().set_fontsize(label_size)
leg.get_title().set_fontweight("bold")
for text in leg.get_texts():
    text.set_fontsize(label_size - 4)

g.set_titles(col_template=" ", size=0)

# %%
pitch_les = df_avg_les[df_avg_les["Movement"] == "Pitch"]

# %%
pitch_les = df_avg_les[(df_avg_les["Movement"] == "Pitch") & (df_avg_les["Local Thrust Coefficient"] == 1.33)]
sns.scatterplot(x = pitch_les["Distance"], y = pitch_les["Cp_Turb"], hue = pitch_les["Amplitude"], style = pitch_les["Frequency"])

# %%
pitch_les = df_les[(df_les["Movement"] == "Pitch") & (df_les["Local Thrust Coefficient"] == 1.33)]
g = sns.relplot(
    data=pitch_les.sort_values(by = ["Tilt"]),
    x="Tilt",
    y="Cp_Turb_normalized",
    hue="Amplitude",
    palette=palette,
    edgecolors = "k",
    s = 60,
    linewidth=0.5,
    kind="scatter",
    col="Frequency",
    height=4,
    aspect=1.2,
    facet_kws={"sharey": True, "sharex": True},
)

# %%
df_umm_box = df_umm[(df_umm["Local Thrust Coefficient"] == 1.33)]
df_umm_box["Model"] = "UMM"

df_les_box = df_les[(df_les["Local Thrust Coefficient"] == 1.33)]
df_les_box["Model"] = "LES"

# %%
df_box = pd.concat([df_umm_box, df_les_box], ignore_index=True)
df_box["Model_Frequency"] = df_box['Model'].astype(str) + ': f = ' + df_box['Frequency'].astype(str)
df_box["Amplitude"] = np.round(df_box["Amplitude"], decimals=1)
df_box = df_box[df_box["Amplitude"] % 0.2 == 0]
df_box

# %%
g = sns.catplot(
    data=df_box,
    x="Amplitude",
    y="Cp_Turb",
    hue="Model_Frequency",
    col="Movement",
    kind="box",
    palette="tab10",
    sharey = True,
    sharex=False,
    showmeans=True,
    meanprops={
        "marker": "x",        # Diamond shape
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "markersize": 4
    }
)

# g.figure.suptitle('Pitching Turbine: $a_n$ vs Amplitude', y = 1.05)

# g.set_titles(col_template="$C_T'$ = {col_name}")
# g.set_axis_labels("Amplitude", "$a_n$")

# %%
df_phase_avg_les_surge = df_phase_avg_les[df_phase_avg_les["Movement"] == "Surge"]
df_total_avg_umm_surge_ct_133 = df_avg_umm[(df_avg_umm["Local Thrust Coefficient"] == 1.33) & (df_avg_umm["Movement"] == "Surge")]
df_total_avg_umm_surge_ct_200 = df_avg_umm[(df_avg_umm["Local Thrust Coefficient"] == 2.00) & (df_avg_umm["Movement"] == "Surge")]

# %%
# Create the relplot with columns based on "Local Thrust Velocity"
g = sns.relplot(
    data=df_phase_avg_les_surge[df_phase_avg_les_surge["Local Thrust Coefficient"] == 1.33],
    x="Phase_Rounded",
    y="an_Turb",
    hue="Amplitude",
    style="Frequency",
    palette=palette,
    linewidth=2.5,
    kind="line",
    col="Local Thrust Coefficient",
    height=4,
    aspect=1.2,
    facet_kws={"sharey": True, "sharex": True},
)


g.fig.set_size_inches(5, 5)
g.fig.set_dpi(300)
g._legend.remove()

# Apply custom line styles for frequencies
freq_styles = ['-', '-.', '--', ':']
freqs = sorted(df_phase_avg_les_surge["Frequency"].unique())
for ax in g.axes.flatten():
    lines = ax.get_lines()
    for i, line in enumerate(lines):
        freq_idx = i % len(freqs)  # match line index to frequency
        line.set_linestyle(freq_styles[freq_idx])

# Add the UMM horizontal line to each axis
for (i, ax) in enumerate(g.axes.flatten()):
    # ax.set_box_aspect(0.5)  
    if i == 0:
        vals = df_total_avg_umm_surge_ct_133
    else:
        vals = df_total_avg_umm_surge_ct_200
    umm_mean = np.mean(vals["an_Turb"])
    ax.axhline(y=umm_mean, c="darkorange", linewidth=2.5)
    ax.axvline(0.25, color='darkgrey', linestyle='--', linewidth=1)
    ax.axvline(0.75, color='darkgrey', linestyle='--', linewidth=1)
    # shaded vertical region between x1 and x2
    shade_label = '$U_{\infty, t} > U_{\infty, g}$ ($u_{t, g} < 0$)'
    ax.axvspan(0.25, 0.75, color='lightgrey', alpha=0.3, label=shade_label)

# ====== Custom legend ======
# LES main header (bold)
les_header = Line2D([], [], linestyle="none", marker="", label=r"\textbf{LES}", color="none")

# LES subheaders: Amplitude and Frequency
space_header = Line2D([], [], linestyle="none", marker="", label=" ", color="none")

# LES handles (colors for amplitudes)
amps = sorted(df_phase_avg_les_surge["Amplitude"].unique())
amp_handles = [Line2D([], [], color=palette[i], lw=2, label=f"$A_S = {amps[i]}$") for i in range(len(amps))]

# LES handles (line styles for frequencies)
freqs = sorted(df_phase_avg_les_surge["Frequency"].unique())
freq_handles = [Line2D([], [], color='grey', linestyle=freq_styles[i % len(freq_styles)], lw=2, label=f"$f = {freqs[i]}$") for i in range(len(freqs))]

# UMM line handle
umm_header = Line2D([], [], color="darkorange", linewidth=2.5, label=r"\textbf{UMM}")

# Shaded region handle
shade_handle = Patch(
    facecolor='grey',
    edgecolor='none',
    alpha=0.4,
    label='$\\mathbf{U_{t} < 0}$'
)

# Combine all handles
all_handles = [les_header] + amp_handles + [space_header] + freq_handles + [space_header, umm_header, shade_handle]

# Place the legend outside the plot (right)
g.fig.subplots_adjust(right=0.8)
leg = g.fig.legend(
    handles=all_handles,
    # title="Frequency",
    loc="center right",
    bbox_to_anchor=(1.15, 0.5),
    bbox_transform=g.fig.transFigure,
    frameon = False
)

for text in leg.get_texts():
    text.set_fontsize(label_size - 4)

# Set labels and title for the whole figure
g.set_titles(" ", size=label_size)
for ax in g.axes.flatten():
    ax.set_xlabel("$t/T$", fontsize=label_size)
    ax.set_ylabel("$a_n$", fontsize=label_size + 2)
    ax.tick_params(axis="both", labelsize=label_size - 2)
    ax.tick_params(axis="y", labelsize=label_size - 2)

# g.fig.suptitle("Surging Phase-Averaged Induction", fontsize=title_size)
# g.fig.tight_layout()
g.fig.subplots_adjust(top=0.9)  # leave space for suptitle
g.fig.set_dpi(300)


# %%
palette = sns.color_palette("viridis_r", n_colors=len(freqs))
palette = [palette[0], palette[2], palette[3], palette[1]]
palette

# %%
# ===================== CREATE RELPLOT =====================
import itertools
g = sns.relplot(
    data=df_phase_avg_les_surge[df_phase_avg_les_surge["Local Thrust Coefficient"] == 1.33],
    x="Phase_Rounded",
    y="an_Turb",
    hue="Amplitude",
    style="Frequency",
    palette=palette,
    linewidth=2.5,
    kind="line",
    col="Local Thrust Coefficient",
    height=4,
    aspect=1.2,
    facet_kws={"sharey": True, "sharex": True}
)

g.fig.set_size_inches(5, 5)
g.fig.set_dpi(300)

# Remove Seaborn's auto-legend
g._legend.remove()


# ===================== APPLY ALPHA + SOLID LINES =====================
# Frequency order and alpha values
freqs = sorted(df_phase_avg_les_surge["Frequency"].unique())
alpha_vals = ([1.0, 0.75, 0.5, 0.25])
alpha_vals.reverse()

# Number of amplitudes
amps_sorted = sorted(df_phase_avg_les_surge["Amplitude"].unique())
n_amp = len(amps_sorted)

# Apply alpha and force solid linestyle
for ax in g.axes.flatten():
    lines = ax.get_lines()

    # Repeat alpha for each amplitude per frequency
    alphas = alpha_vals * 4

    for line, alpha in zip(lines, alphas):
        line.set_alpha(alpha)
        line.set_linestyle("-")
# ====================== ADD UMM + SHADING ======================
for i, ax in enumerate(g.axes.flatten()):
    vals = df_total_avg_umm_surge_ct_133 if i == 0 else df_total_avg_umm_surge_ct_200
    umm_mean = np.mean(vals["an_Turb"])

    ax.axhline(y=umm_mean, c="k", linewidth=3, linestyle = "--")
    ax.axvline(0.25, color='darkgrey', linestyle='--', linewidth=1)
    ax.axvline(0.75, color='darkgrey', linestyle='--', linewidth=1)

    ax.axvspan(0.25, 0.75, color='lightgrey', alpha=0.3,
               label='$U_{\\infty,t} > U_{\\infty,g}$ ($u_{t,g} < 0$)')


# ======================= CUSTOM LEGEND =========================
les_header = Line2D([], [], linestyle="none", marker="", 
                    label=r"\textbf{LES}", color="none")

space_header = Line2D([], [], linestyle="none", marker="", label=" ", color="none")

# Amplitude handles
amp_handles = [
    Line2D([], [], color=palette[i], lw=2.5, label=f"$A_S = {amps_sorted[i]}$")
    for i in range(len(amps_sorted))
]

# Frequency handles (solid lines, alpha matches plot)
freq_handles = [
    Line2D([], [], color="grey", lw=2.5, linestyle="-",
           alpha=alpha_vals[i], label=f"$f = {freqs[i]}$")
    for i in range(len(freqs))
]

# UMM handle
umm_header = Line2D([], [], color="k", linewidth=2.5, linestyle="--",
                    label=r"\textbf{UMM}")

# Shaded region handle
shade_handle = Patch(facecolor='grey', edgecolor='none', alpha=0.4,
                     label=r"$\mathbf{U_{t} < 0}$")

# Combine legend handles
all_handles = ([les_header] + amp_handles +
               [space_header] + freq_handles +
               [space_header, umm_header, shade_handle])

# Place legend outside plot
g.fig.subplots_adjust(right=0.8)
leg = g.fig.legend(handles=all_handles, loc="center right",
                   bbox_to_anchor=(1.15, 0.5),
                   bbox_transform=g.fig.transFigure,
                   frameon=False)

# Style legend
for text in leg.get_texts():
    text.set_fontsize(label_size - 4)


# ======================= LABELS & TITLES =======================
g.set_titles(" ", size=label_size)

for ax in g.axes.flatten():
    ax.set_xlabel("$t/T$", fontsize=label_size)
    ax.set_ylabel("$a_n$", fontsize=label_size + 2)
    ax.tick_params(axis="both", labelsize=label_size - 2)

g.fig.subplots_adjust(top=0.9)
g.fig.set_dpi(300)


# %%
df_phase_avg_les_surge

# %%
les_udisk_stationary_ct_133 = (1 - les_an_stationary_ct_133[0])
umm_udisk_stationary_ct_133 = (1 - model(1.33, yaw = 0.0, tilt = 0.0).an)

# %%
sub_df_phase_avg_les_surge = df_phase_avg_les_surge[((df_phase_avg_les_surge["Amplitude"] == 0.6) | (df_phase_avg_les_surge["Amplitude"] == 0.8))]
sub_df_phase_avg_les_surge = sub_df_phase_avg_les_surge[(sub_df_phase_avg_les_surge["Local Thrust Coefficient"] == 1.33)]
sub_df_phase_avg_les_surge["UDisk_Ground_Normalized"] = sub_df_phase_avg_les_surge["UDisk_Ground"] / les_udisk_stationary_ct_133
sub_df_phase_avg_les_surge["UDisk_Turb_Normalized"] = sub_df_phase_avg_les_surge["UDisk_Turb"] / les_udisk_stationary_ct_133
sub_df_phase_avg_les_surge.sort_values(by = ["Phase_Rounded"])

# %%
sub_df_umm_surge = df_umm[(df_umm["Local Thrust Coefficient"] == 1.33) & (df_umm["Movement"] == "Surge")]
sub_df_umm_surge["UDisk_Ground_Normalized"] = sub_df_umm_surge["UDisk_Ground"] / umm_udisk_stationary_ct_133
sub_df_umm_surge["UDisk_Turb_Normalized"] = sub_df_umm_surge["UDisk_Turb"] / umm_udisk_stationary_ct_133
sub_df_umm_surge.sort_values(by = ["Phase_Rounded"])

# %%
# Create the relplot with columns based on "Local Thrust Velocity"
# Create the relplot
palette = sns.color_palette("GnBu_d", n_colors=len(freqs))

g = sns.relplot(
    data=sub_df_phase_avg_les_surge[sub_df_phase_avg_les_surge["Amplitude"] == 0.6],
    x="UTurb",
    y="UDisk_Ground_Normalized",
    hue="Frequency",
    palette=palette,
    linewidth=2.5,
    kind="line",
    col="Amplitude",
    height=4,
    aspect=1.2,
    facet_kws={"sharey": True, "sharex": True},
    sort=False,
)

# === Add ARROWS for Phase = 0.25 and Phase = 0.75 ===
for (amp, ax) in g.axes_dict.items():

    # rows for this amplitude
    dfA = sub_df_phase_avg_les_surge[
        (sub_df_phase_avg_les_surge["Amplitude"] == amp)
    ]

    # phase = 0 and 0.5 rows
    df0  = dfA[dfA["Phase_Rounded"] == 0]
    df1  = dfA[dfA["Phase_Rounded"] == 0.16]
    df2 = dfA[dfA["Phase_Rounded"] == 0.56]

    # Loop through frequency groups so we can color arrows by frequency
    for freq, df_f in df0.groupby("Frequency"):
        color = palette[list(sorted(dfA["Frequency"].unique())).index(freq)]

        ax.scatter(
            df_f["UTurb"],
            df_f["UDisk_Ground_Normalized"],
            marker="*",
            s=200,
            facecolors=color,
            edgecolors="k",
            linewidth=2,
            zorder=25
        )

    for freq, df_f in df1.groupby("Frequency"):
        color = palette[list(sorted(dfA["Frequency"].unique())).index(freq)]

        ax.scatter(
            df_f["UTurb"],
            df_f["UDisk_Ground_Normalized"],
            marker="<",
            s=120,
            facecolors=color, 
            edgecolors="k",
            linewidth=2,
            zorder=25
        )

    for freq, df_f in df2.groupby("Frequency"):
        color = palette[list(sorted(dfA["Frequency"].unique())).index(freq)]

        ax.scatter(
            df_f["UTurb"],
            df_f["UDisk_Ground_Normalized"],
            marker=">",
            s=120,
            facecolors=color,
            edgecolors="k",
            linewidth=2,
            zorder=25
        )
g.fig.set_size_inches(5, 5)
g.fig.set_dpi(300)
g._legend.remove()

# # Add the UMM horizontal line to each axis
for (i, ax) in enumerate(g.axes.flatten()):
    if i == 0:
        vals = sub_df_umm_surge[(sub_df_umm_surge["Amplitude"] == 0.6)]
    else:
        vals = sub_df_umm_surge[(sub_df_umm_surge["Amplitude"] == 0.8)]
    vals = vals.sort_values(by = ["Phase_Rounded"])
    ax.plot(vals["UTurb"], vals["UDisk_Ground_Normalized"], c="darkorange", linewidth=2.5)
    

# ====== Custom legend ======
# # LES subheaders: Amplitude and Frequency
les_header = Line2D([], [], linestyle="none", marker="", label="\\textbf{LES}", color="none")

# # LES handles (line styles for frequencies)
freqs = sorted(df_phase_avg_les_surge["Frequency"].unique())
freq_handles = [Line2D([], [], color=palette[i], lw=2, label=f"$f = {freqs[i]}$") for i in range(len(freqs))]

# # UMM line handle
umm_header = Line2D([], [], color="darkorange", linewidth=2.5, label="\\textbf{UMM}")

# Star marker for t/T = 0
star_handle = Line2D(
    [], [], 
    marker="*", 
    markersize=20,
    markerfacecolor="white",
    markeredgecolor="k",
    linewidth=0,
    markeredgewidth=2,
    label="$\mathbf{t/T = 0}$"
)

# # Combine all handles
all_handles = [les_header] + freq_handles + [space_header, umm_header, star_handle]

# # Place the legend outside the plot (right)
leg = g.fig.legend(
    handles=all_handles,
    # title="Frequency",
    loc="center right",
    bbox_to_anchor=(1.2, 0.5),
    bbox_transform=g.fig.transFigure,
    frameon = False
)

# # Style legend
leg.get_title().set_fontsize(label_size)
leg.get_title().set_fontweight("bold")
for text in leg.get_texts():
    text.set_fontsize(label_size - 4)

# # Set labels and title for the whole figure
g.set_titles(" ", size=0)
for ax in g.axes.flatten():
    ax.set_xlabel("$U_t$", fontsize=label_size)
    ax.set_ylabel("$u_{d,\\textrm{\\fontsize{14}{14}\\selectfont ground}} /  u_{d,\\textrm{\\fontsize{14}{14}\\selectfont fixed}}$", fontsize=label_size + 2)
    ax.tick_params(axis="both", labelsize=label_size - 2)
    ax.tick_params(axis="y", labelsize=label_size - 2)

# g.fig.suptitle("Normalized Disk Velocity in the Ground Frame", fontsize=title_size)
# g.fig.tight_layout()
# g.fig.subplots_adjust(top=0.9)  # leave space for suptitle


# %%
f = 0.4
uturb, dx = get_uturb_vals(0.2, f, 0.05)
uinf_vals = np.ones_like(uturb)
uinf_t = 1 - uturb
model_vals = model(1.33, yaw = 0.0, tilt = 0.0)
xvals = np.arange(0, 1/f , dt)

# %%
fig, ax = plt.subplots(figsize=(6,4))

xvals = range(0, len(uturb))

sns.lineplot(ax=ax, x=xvals, y=model_vals.Ct * uinf_t**2, label="$C_T$")
sns.lineplot(ax=ax, x=xvals, y=uturb, label="$U_t$")
sns.lineplot(ax=ax, x=xvals, y=uinf, label="$U_{\infty}$", color="tab:green")
sns.lineplot(ax=ax, x=xvals, y=uinf_t, label="$U_{\infty, t}$", color="tab:green", linestyle="--")

# Axis labels
ax.set_xlabel("Timestep", fontsize=label_size)

# Tick sizes
ax.tick_params(axis="both", labelsize=18)

# Legend
leg = ax.legend(frameon=False, fontsize=16)
for text in leg.get_texts():
    text.set_fontsize(16)

fig.set_size_inches(5, 5)
fig.set_dpi(300)


# %%
