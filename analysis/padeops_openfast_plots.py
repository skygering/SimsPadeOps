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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.interpolate import RegularGridInterpolator
from matplotlib.lines import Line2D

# %%
# =========================================================
# CONFIG
# =========================================================
LES_FILE = "saved_stats/overall_stats_summary.csv"
OF_MOTION_FILE = "saved_stats/openfast_condition_summary_stats_all_cases.csv"
OF_CT_FILE = "saved_stats/openfast_ct_condition_summary_stats_all_cases.csv"
ERROR_COL = "Diff_CT_max"  # or Diff_CP_mean
MERGE_KEYS = ["HWindSpeed", "WaveHs", "WaveTp", "n_seeds"]

SHOW_WORST_CASE_POINTS = True
WC_MARKER = "^"
WC_SIZE = 70
MEAN_COLOR = "crimson"
WC_COLOR = "k"
WC_LABEL = "OpenFAST worst-case amplitude"

# choose worst-case amplitude/frequency columns
# (edit if your file uses different names)
WC_A_COL_CANDIDATES = ["v_A_wc_ens_mean", "x_A_wc_ens_mean"]
WC_F_COL_CANDIDATES = ["v_f_ens_mean", "x_f_ens_mean"]  # often same freq basis

# %%
# --- OpenFAST filtering ---
FILTER_OPENFAST = True
REMOVE_WIND_SPEEDS = [3.0]  # [] to keep all

# --- LES limiting (set to None to disable each) ---
LES_CT_RANGE = None          # e.g. (0.2, 0.8)
LES_A_RANGE = (0.0, 0.3)           # e.g. (0.0, 1.2)
LES_F_RANGE = None           # e.g. (0.0, 2.0)

# %%
# --- Axis controls ---
PAD_FRAC = 0.05
FORCE_FREQ_MAX = None         # set None to disable
FORCE_FREQ_MIN = None        # e.g. 0.0
FORCE_CT_MIN = None
FORCE_CT_MAX = None
FORCE_A_MIN = None
FORCE_A_MAX = None

# %%
COLOR_OPENFAST_POINTS_BY_ERROR = False
DPI = 300


# %%
# =========================================================
# HELPERS
# =========================================================
def apply_range(df, col, lim):
    if lim is None:
        return df
    lo, hi = lim
    return df[(df[col] >= lo) & (df[col] <= hi)]

def padded_bounds(a, b, pad_frac=0.05):
    x = np.concatenate([np.asarray(a).ravel(), np.asarray(b).ravel()])
    x = x[np.isfinite(x)]
    mn, mx = np.min(x), np.max(x)
    if mx == mn:
        d = 0.5 if mn == 0 else 0.05 * abs(mn)
        return mn - d, mx + d
    pad = pad_frac * (mx - mn)
    return mn - pad, mx + pad

def override_limits(lim, forced_min=None, forced_max=None):
    lo, hi = lim
    if forced_min is not None:
        lo = forced_min
    if forced_max is not None:
        hi = forced_max
    return (lo, hi)

def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# %%
# =========================================================
# LOAD DATA
# =========================================================
of_motion = pd.read_csv(OF_MOTION_FILE)
of_ct = pd.read_csv(OF_CT_FILE)
of = pd.merge(of_motion, of_ct, on=MERGE_KEYS, how="inner")

# map requested OF vars
D = 240.0 
of["CT_prime"] = of["CTp_ens_mean"]
U = of["HWindSpeed"].to_numpy(dtype=float)
U_safe = np.where(np.abs(U) > 1e-12, U, np.nan)
of["Frequency"] = of["v_f_ens_mean"].to_numpy(dtype=float) * D / U_safe
of["Surge_Amplitude"] = of["v_A_ens_mean"].to_numpy(dtype=float) / U_safe

# worst-case mapping
wc_a_col = first_existing_col(of, WC_A_COL_CANDIDATES)
wc_f_col = first_existing_col(of, WC_F_COL_CANDIDATES)

if SHOW_WORST_CASE_POINTS:
    if wc_a_col is None or wc_f_col is None:
        print("[warn] Worst-case columns not found; skipping WC points.")
        SHOW_WORST_CASE_POINTS = False
    else:
        # numeric coercion
        of[wc_a_col] = pd.to_numeric(of[wc_a_col], errors="coerce")
        of[wc_f_col] = pd.to_numeric(of[wc_f_col], errors="coerce")

        # if freq is dimensional, nondim it same way: f* = f D / U
        # if already nondim, replace with of[wc_f_col] directly
        U = pd.to_numeric(of["HWindSpeed"], errors="coerce").to_numpy()
        U_safe = np.where(np.abs(U) > 1e-12, U, np.nan)

        of["Frequency_wc"] = of[wc_f_col].to_numpy(dtype=float) * D / U_safe
        # amplitude worst-case nondim: A* = A/U (if dimensional)
        # if already nondim, replace with of[wc_a_col]
        of["Surge_Amplitude_wc"] = of[wc_a_col].to_numpy(dtype=float) / U_safe

        # keep CT same mapping
        of["CT_prime_wc"] = of["CT_prime"]

# OF filter: remove wind speeds
if FILTER_OPENFAST and len(REMOVE_WIND_SPEEDS) > 0:
    n0 = len(of)
    of = of[~of["HWindSpeed"].isin(REMOVE_WIND_SPEEDS)].copy()
    print(f"Removed {n0-len(of)} OF rows by HWindSpeed filter.")


# %%
def extend_les_plateau_frequency(
    les_df,
    freq_col="Frequency",
    group_cols=("CT_prime", "Surge_Amplitude"),
    value_cols=("Diff_CP_min","Diff_CP_max","Diff_CP_mean","Diff_CT_min","Diff_CT_max","Diff_CT_mean"),
    f_plateau_start=2.0,
    f_max=8.0,
    f_step=None,                 # None -> reuse LES freq spacing if possible
    use_nearest_if_missing=True
):
    les = les_df.copy()

    # Determine frequency points to add
    f_unique = np.sort(les[freq_col].dropna().unique())

    if f_step is None:
        if len(f_unique) >= 2:
            # use smallest positive spacing from existing LES
            diffs = np.diff(f_unique)
            diffs = diffs[diffs > 0]
            f_step = diffs.min() if len(diffs) else 0.25
        else:
            f_step = 0.25

    new_freqs = np.arange(f_plateau_start + f_step, f_max + 0.5*f_step, f_step)
    new_freqs = new_freqs[new_freqs <= f_max + 1e-12]

    rows_to_add = []

    # build per (CT',A) plateau rows
    for keys, g in les.groupby(list(group_cols), dropna=False):
        g = g.sort_values(freq_col)

        # choose baseline row at f=2 (or nearest)
        exact = g[np.isclose(g[freq_col], f_plateau_start)]
        if len(exact) > 0:
            base = exact.iloc[0]
        else:
            if not use_nearest_if_missing:
                continue
            idx = (g[freq_col] - f_plateau_start).abs().idxmin()
            base = g.loc[idx]

        # add rows for new frequencies if not already present
        existing_f = set(np.round(g[freq_col].to_numpy(dtype=float), 12))
        for fnew in new_freqs:
            if np.round(fnew, 12) in existing_f:
                continue

            row = base.copy()
            row[freq_col] = float(fnew)
            rows_to_add.append(row)

    if rows_to_add:
        add_df = pd.DataFrame(rows_to_add)
        les = pd.concat([les, add_df], ignore_index=True)

    les = les.sort_values(list(group_cols) + [freq_col]).reset_index(drop=True)
    return les


# %%
les = pd.read_csv(LES_FILE)

# LES limiting
les = apply_range(les, "CT_prime", LES_CT_RANGE)
les = apply_range(les, "Surge_Amplitude", LES_A_RANGE)
les = apply_range(les, "Frequency", LES_F_RANGE)

les = extend_les_plateau_frequency(
    les_df=les,
    freq_col="Frequency",
    group_cols=("CT_prime", "Surge_Amplitude"),
    value_cols=("Diff_CP_min","Diff_CP_max","Diff_CP_mean","Diff_CT_min","Diff_CT_max","Diff_CT_mean"),
    f_plateau_start=2.0,
    f_max=8.0,
    f_step=0.25,   # pick your preferred resolution
    use_nearest_if_missing=True
)

if les.empty:
    raise ValueError("LES data is empty after LES_*_RANGE filters.")

# %%
# =========================================================
# LES GRID (SMOOTH STYLE via griddata on fine (A,f) mesh)
# =========================================================
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator  # keep for OF blend
import numpy as np
import pandas as pd
from matplotlib import cm, colors

# Smoothing controls (style like your contour function)
GRID_RES = 120                 # smoothness
SMOOTH_METHOD = "cubic"        # "cubic" | "linear" | "nearest"
SURFACE_ALPHA = 0.90
SHOW_LES_SCATTER = False       # optional raw LES points on each slice
DZ_SNAP = 0.05                 # snap colorbar bounds to 0.05 bins

ct_vals = np.sort(les["CT_prime"].unique())
a_vals = np.sort(les["Surge_Amplitude"].unique())
f_vals = np.sort(les["Frequency"].unique())

if len(ct_vals) < 2:
    print("[warn] Fewer than 2 CT slices in LES after filtering.")

# Keep structured grid for OF blend interpolation
grid_index = pd.MultiIndex.from_product(
    [ct_vals, a_vals, f_vals],
    names=["CT_prime", "Surge_Amplitude", "Frequency"]
)
les_grid = (
    les.set_index(["CT_prime", "Surge_Amplitude", "Frequency"])[ERROR_COL]
       .reindex(grid_index)
       .values
       .reshape(len(ct_vals), len(a_vals), len(f_vals))
)

interp_by_ct = {}
for i, ct in enumerate(ct_vals):
    interp_by_ct[ct] = RegularGridInterpolator(
        (a_vals, f_vals), les_grid[i, :, :],
        bounds_error=False, fill_value=np.nan
    )

pts_af = of[["Surge_Amplitude", "Frequency"]].to_numpy()

if len(ct_vals) >= 2:
    ct0, ct1 = ct_vals[0], ct_vals[-1]
    v0 = interp_by_ct[ct0](pts_af)
    v1 = interp_by_ct[ct1](pts_af)
    w = (of["CT_prime"] - ct0) / (ct1 - ct0 + 1e-15)
    w = np.clip(w, 0.0, 1.0)
    of[f"{ERROR_COL}_blend"] = (1 - w) * v0 + w * v1
else:
    of[f"{ERROR_COL}_blend"] = interp_by_ct[ct_vals[0]](pts_af)

# %%
# =========================================================
# PRECOMPUTE SMOOTH SURFACES FOR EACH CT SLICE
# =========================================================
smooth_surfaces = {}  # ct -> dict(Xf, Ya, C)
global_cmin, global_cmax = np.inf, -np.inf

for ct in ct_vals:
    sub = les[les["CT_prime"] == ct].dropna(
        subset=["Surge_Amplitude", "Frequency", ERROR_COL]
    ).copy()

    x = sub["Frequency"].to_numpy()         # f
    y = sub["Surge_Amplitude"].to_numpy()   # A
    z = sub[ERROR_COL].to_numpy()           # color value

    xi = np.linspace(np.nanmin(x), np.nanmax(x), GRID_RES)
    yi = np.linspace(np.nanmin(y), np.nanmax(y), GRID_RES)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, y), z, (Xi, Yi), method=SMOOTH_METHOD)

    # Fill holes for stability (especially cubic near convex hull edges)
    if np.isnan(Zi).any():
        Zi_lin = griddata((x, y), z, (Xi, Yi), method="linear")
        Zi_near = griddata((x, y), z, (Xi, Yi), method="nearest")
        Zi = np.where(np.isnan(Zi), Zi_lin, Zi)
        Zi = np.where(np.isnan(Zi), Zi_near, Zi)

    smooth_surfaces[ct] = dict(Xf=Xi, Ya=Yi, C=Zi, raw_f=x, raw_a=y)

    zmin = np.nanmin(Zi)
    zmax = np.nanmax(Zi)
    if np.isfinite(zmin): global_cmin = min(global_cmin, zmin)
    if np.isfinite(zmax): global_cmax = max(global_cmax, zmax)

# include OF blended values in color range
all_blend = of[f"{ERROR_COL}_blend"].to_numpy()
all_blend = all_blend[np.isfinite(all_blend)]
if all_blend.size:
    global_cmin = min(global_cmin, np.nanmin(all_blend))
    global_cmax = max(global_cmax, np.nanmax(all_blend))

# snap to dz bins (style like your contour function)
if np.isfinite(global_cmin) and np.isfinite(global_cmax):
    vmin = DZ_SNAP * np.floor(global_cmin / DZ_SNAP)
    vmax = DZ_SNAP * np.ceil(global_cmax / DZ_SNAP)
else:
    vmin, vmax = 0.0, 1.0

norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.viridis

# %%
# =========================================================
# AXIS LIMITS: COMBINED LES + OF (then optional force)
# =========================================================
f_lim = padded_bounds(f_vals, of["Frequency"], PAD_FRAC)
a_lim = padded_bounds(a_vals, of["Surge_Amplitude"], PAD_FRAC)
ct_lim = padded_bounds(ct_vals, of["CT_prime"], PAD_FRAC)

f_lim = override_limits(f_lim, FORCE_FREQ_MIN, FORCE_FREQ_MAX)
a_lim = override_limits(a_lim, FORCE_A_MIN, FORCE_A_MAX)
ct_lim = override_limits(ct_lim, FORCE_CT_MIN, FORCE_CT_MAX)

print("Axis limits used:")
print("  Frequency:", f_lim)
print("  Amplitude:", a_lim)
print("  CT':      ", ct_lim)


# %%
# =========================================================
# DRAW FUNCTION (SMOOTH SURFACES)
# =========================================================
def draw_panel(ax, layout, elev, azim):
    """
    Draw one 3D panel with:
      - smoothed LES surfaces (colored by ERROR_COL)
      - OpenFAST mean points
      - OpenFAST worst-case points (optional)
    """

    import numpy as np

    layout_cfg = {
        "f_A_CT": {
            "labels": (
                r"Non-dimensional frequency, $fD/U_\infty$ [-]",
                r"Non-dimensional amplitude, $A_v/U_\infty$ [-]",
                r"$CT'$ [-]",
            ),
            "lims": (f_lim, a_lim, ct_lim),
            "of_cols": ("Frequency", "Surge_Amplitude", "CT_prime"),
            "wc_cols": ("Frequency_wc", "Surge_Amplitude_wc", "CT_prime_wc"),
            "surface_xyz": lambda Xi, Yi, ct: (Xi, Yi, np.full_like(Xi, ct, dtype=float)),
            "les_xyz": lambda rf, ra, ct: (rf, ra, np.full_like(rf, ct, dtype=float)),
        },
        "CT_f_A": {
            "labels": (
                r"$CT'$ [-]",
                r"Non-dimensional frequency, $fD/U_\infty$ [-]",
                r"Non-dimensional amplitude, $A_v/U_\infty$ [-]",
            ),
            "lims": (ct_lim, f_lim, a_lim),
            "of_cols": ("CT_prime", "Frequency", "Surge_Amplitude"),
            "wc_cols": ("CT_prime_wc", "Frequency_wc", "Surge_Amplitude_wc"),
            "surface_xyz": lambda Xi, Yi, ct: (np.full_like(Xi, ct, dtype=float), Xi, Yi),
            "les_xyz": lambda rf, ra, ct: (np.full_like(rf, ct, dtype=float), rf, ra),
        },
        "f_CT_A": {
            "labels": (
                r"Non-dimensional frequency, $fD/U_\infty$ [-]",
                r"$CT'$ [-]",
                r"Non-dimensional amplitude, $A_v/U_\infty$ [-]",
            ),
            "lims": (f_lim, ct_lim, a_lim),
            "of_cols": ("Frequency", "CT_prime", "Surge_Amplitude"),
            "wc_cols": ("Frequency_wc", "CT_prime_wc", "Surge_Amplitude_wc"),
            "surface_xyz": lambda Xi, Yi, ct: (Xi, np.full_like(Xi, ct, dtype=float), Yi),
            "les_xyz": lambda rf, ra, ct: (rf, np.full_like(rf, ct, dtype=float), ra),
        },
    }

    if layout not in layout_cfg:
        raise ValueError(f"Unknown layout: {layout}")
    cfg = layout_cfg[layout]

    # axes
    (xlabel, ylabel, zlabel) = cfg["labels"]
    (xlim, ylim, zlim) = cfg["lims"]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # LES surfaces
    for ct in ct_vals:
        Xi = smooth_surfaces[ct]["Xf"]
        Yi = smooth_surfaces[ct]["Ya"]
        Ci = smooth_surfaces[ct]["C"]

        X, Y, Z = cfg["surface_xyz"](Xi, Yi, ct)
        ax.plot_surface(
            X, Y, Z,
            facecolors=cmap(norm(Ci)),
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=SURFACE_ALPHA,
        )

        if SHOW_LES_SCATTER:
            rf = smooth_surfaces[ct]["raw_f"]
            ra = smooth_surfaces[ct]["raw_a"]
            sx, sy, sz = cfg["les_xyz"](rf, ra, ct)
            ax.scatter(sx, sy, sz, marker="+", c="k", s=20, alpha=0.8)

    def plot_points(cols, marker, size, label, fallback_color="k", lw=0.35, alpha=0.95):
        xcol, ycol, zcol = cols
        if not all(c in of.columns for c in (xcol, ycol, zcol)):
            return

        mask = (
            np.isfinite(of[xcol].to_numpy()) &
            np.isfinite(of[ycol].to_numpy()) &
            np.isfinite(of[zcol].to_numpy())
        )
        if not np.any(mask):
            return

        xs = of.loc[mask, xcol].to_numpy()
        ys = of.loc[mask, ycol].to_numpy()
        zs = of.loc[mask, zcol].to_numpy()

        use_error_color = COLOR_OPENFAST_POINTS_BY_ERROR and (f"{ERROR_COL}_blend" in of.columns)

        if use_error_color:
            cvals = of.loc[mask, f"{ERROR_COL}_blend"].to_numpy()
            ax.scatter(
                xs, ys, zs,
                c=cvals, cmap=cmap, norm=norm,
                marker=marker, s=size,
                edgecolor="k", linewidth=lw,
                depthshade=False, alpha=alpha,
                label=label,
            )
        else:
            ax.scatter(
                xs, ys, zs,
                c=fallback_color,
                marker=marker, s=size,
                edgecolor="k", linewidth=lw,
                depthshade=False, alpha=alpha,
                label=label,
            )

    # worst-case (follows same color mode as mean)
    if SHOW_WORST_CASE_POINTS:
        plot_points(
            cols=cfg["wc_cols"],
            marker=WC_MARKER,
            size=WC_SIZE,
            label=WC_LABEL,
            fallback_color=WC_COLOR,
            lw=0.4,
            alpha=0.98 if COLOR_OPENFAST_POINTS_BY_ERROR else 0.95,
        )

    # mean
    plot_points(
        cols=cfg["of_cols"],
        marker="o",
        size=34 if COLOR_OPENFAST_POINTS_BY_ERROR else 30,
        label="OpenFAST mean points",
        fallback_color=MEAN_COLOR,
        lw=0.35,
        alpha=0.98 if COLOR_OPENFAST_POINTS_BY_ERROR else 0.95,
    )


    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)

# %%
# --- Plot controls ---
PLOT_LAYOUTS = [
    "f_A_CT",
    # "CT_f_A",
    "f_CT_A",
]  # includes requested CT on y

VIEWS = {
    "f_A_CT": [(24, -128), (20, -38)],   # x=f, y=A, z=CT'
    # "CT_f_A": [(20, -128), (18, -38)],     # x=CT', y=f, z=A
    "f_CT_A": [(20, -140), (18, -25)],   # x=f, y=CT', z=A
}

# %%
# =========================================================
# PLOT FIGURES (same layout style: close panels + right-shifted cbar)
# =========================================================
for layout in PLOT_LAYOUTS:
    fig = plt.figure(figsize=(22, 7), dpi=DPI)

    # more left margin; keep tight gap between panels
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        left=0.425, right=0.84,   # <-- increased left, slightly reduced right
        bottom=0.10, top=0.92,
        # wspace=0.1
    )
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")

    (e1, a1), (e2, a2) = VIEWS[layout]
    draw_panel(ax1, layout, e1, a1)
    draw_panel(ax2, layout, e2, a2)

    ax1.set_title("View A")
    ax2.set_title("View B")

    # shared colorbar farther right
    cax = fig.add_axes([0.88, 0.16, 0.018, 0.68])
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(ERROR_COL)

    # ---- one shared legend for points ----
    if COLOR_OPENFAST_POINTS_BY_ERROR:
        mean_face = "lightgray"   # proxy only; true colors come from colormap
        wc_face = "lightgray"
    else:
        mean_face = MEAN_COLOR
        wc_face = WC_COLOR

    handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=mean_face, markeredgecolor='k',
               markersize=7, label="OpenFAST mean points")
    ]
    if SHOW_WORST_CASE_POINTS:
        handles.append(
            Line2D([0], [0], marker=WC_MARKER, color='none',
                   markerfacecolor=wc_face, markeredgecolor='k',
                   markersize=8, label=WC_LABEL)
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.6, 0.95),   # centered over the two subplots region
        ncol=len(handles),
        frameon=True
    )

    fig.suptitle(f"Smoothed LES surfaces + OpenFAST points | layout={layout}", y=0.995)

    outname = f"les_openfast_2panel_smoothed_{layout}.png"
    plt.savefig(outname, bbox_inches="tight", pad_inches=0.08)
    plt.show()
    print(f"Saved: {outname}")

# %%
