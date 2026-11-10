# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: simspadeops-YbNxeWqo-py3.11
#     language: python
#     name: python3
# ---

# %%
import os
import lib.analysis_utils as au
import lib.quick_metadata_plots as mplts
import padeopsIO as pio
import matplotlib.pyplot as plt
import numpy as np
import streamtube
import streamtube
import pandas as pd
from pathlib import Path

from padeopsIO.budget_addons import Wave_x
from padeopsIO.utils import fluids_utils as fluids

from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from scipy import ndimage

from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from scipy.interpolate import interp1d, RegularGridInterpolator
from streamtube.interpolate import trilinear_interpolation, linear_interp
from streamtube.integrate import rk4_step

from UnifiedMomentumModel.Momentum import UnifiedMomentum

# %%
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams['figure.dpi'] = 100

# %%
Ct_prime = 2
f = 1.0
Av = 0.3
D = 1
R = D / 2
rho = 1
UInf = 1

# %%
model = UnifiedMomentum()
umm_sol = model(Ct_prime, yaw=0.0, tilt=0.0)
ss_an = umm_sol.an
ss_CT = umm_sol.Ct
ss_u4 = umm_sol.u4

# %%
Lx, Ly, Lz = 25, 8.5, 8.5
n = 256
dx, dy, dz = Lx/n, Ly/n, Lz/n
filterwidth = np.sqrt(dx**2 + dy**2 + dz**2)
M_inv = np.array(1 + (Ct_prime * filterwidth) / (2 * np.sqrt(3 * np.pi)))

# %%
lims = dict(xlim=(-3, 8), ylim=(-2.5, 2.5), zlim=(-2.5, 2.5))

# %%
run_id = 0

# %%
sim_folder =  os.path.join(au.DATA_PATH, "F_0030_SU_Files/")
run_folder = au.get_run_folder(sim_folder, run_id)
sim = pio.BudgetIO(run_folder, padeops=True, runid=run_id, normalize_origin="turbine")
data = pd.DataFrame(au.load_simulation_timeseries(run_folder, "000" + str(run_id), f, spinup_time=75))

# %%
sim.existing_terms()

# %%
steady_id = 2
steady_run_folder = au.get_run_folder(sim_folder, steady_id)
steady_sim = pio.BudgetIO(steady_run_folder, padeops=True, runid=steady_id, normalize_origin="turbine")
steady_data = pd.DataFrame(au.load_simulation_timeseries(steady_run_folder, "000" + str(steady_id), 0, spinup_time=75))

# %%
data['UDisk_sqr'] = data["UDisk"]**2
data["ATurb"] = - (2*np.pi*f)**2 * data["DeltaX"]
motion = data.groupby("PhaseBin", sort=True).agg(
    xturb=("DeltaX", "mean"),
    uturb=("UTurb", "mean"),
    aturb=("ATurb", "mean"),
    udisk = ("UDisk", "mean"),
    udisk_sqr = ("UDisk_sqr", "mean"),
)
phases = motion.index.to_numpy(dtype=float)

# %%
# u_plt_xy_folder = mplts.plot_instantaneous_field(sim_folder, run_id, tidx = "all", field = "u", zlim=0, vmin = 0, vmax = 2, dpi = 300)
# mplts.film_instantaneous_field(u_plt_xy_folder, fps = 10, video_name = "u_xy_stability.mp4")

# %%
# u_plt_xz_folder = mplts.plot_instantaneous_field(sim_folder, run_id, tidx = "all", field = "u", xlim=lims["xlim"], ylim=0, zlim=lims["zlim"], vmin = 0, vmax = 2, dpi = 300)
# mplts.film_instantaneous_field(u_plt_xz_folder, fps = 10, video_name = "u_xz_stability.mp4")

# %%
# v_plt_xy_folder = mplts.plot_instantaneous_field(sim_folder, run_id, tidx = "all", field = "v", xlim=lims["xlim"], ylim=lims["ylim"], zlim=0, vmin = -0.001, vmax = 0.001, dpi = 300)
# mplts.film_instantaneous_field(v_plt_xy_folder, fps = 10, video_name = "v_xy_stability.mp4")

# %%
vals = sim.slice(budget_terms=["ubar", "vbar", "wbar", "pbar", "uStor", "vStor", "wStor"], phase=0.5, xlim=lims["xlim"], ylim=lims["ylim"], zlim=lims["zlim"])
x = vals["x"].values
y = vals["y"].values
z = vals["z"].values

# %%
key = "pbar"
target = 0

plot_vals = vals[key]
iy = np.argmin(np.abs(y - target))
plot_vals = plot_vals[:, iy, :] # (x, y) slice
extent=[x.min(), x.max(), z.min(), z.max()]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    plot_vals.T,
    extent = extent,
    origin="lower",
    aspect="equal",
    interpolation="nearest",
    cmap = "bwr",
    vmin = -0.3, vmax = 0.3, 
)
cbar = fig.colorbar(im, ax=ax)


# %%
def get_streamtube(sim, phase=None, motion=None, ymax=3.0, zmax=3.0):
    # get data slices
    vals = sim.slice(budget_terms=["ubar", "vbar", "wbar", "pbar"], phase=phase,
                     xlim=lims["xlim"], ylim=lims["ylim"], zlim=lims["zlim"])
    # use turbine frame of reference
    x, u = vals["x"], vals["ubar"]
    if phase is not None:
        x = x - motion.loc[phase, "xturb"]
        u = u - motion.loc[phase, "uturb"]
    # get streamtube
    stream = streamtube.Streamtube(x, vals["y"], vals["z"],
                                   u, vals["vbar"], vals["wbar"])
    # get mask
    mask = stream.compute_mask(return_mask=True, R=R - filterwidth)
    closed_mask = ndimage.binary_closing(mask, iterations=2, border_value=0)
    filled_mask = ndimage.binary_fill_holes(closed_mask)

    # Assumes the mask axes are (x, y, z), so y is axis 1 and z is axis 2.
    y = np.asarray(vals["y"]).ravel()
    z = np.asarray(vals["z"]).ravel()
    assert y.size == filled_mask.shape[1] and z.size == filled_mask.shape[2], \
        (y.shape, z.shape, filled_mask.shape)

    Y = y[None, :, None]   # shape (1, ny, 1)
    Z = z[None, None, :]   # shape (1, 1, nz)
    outside = (np.abs(Y) > ymax) | (np.abs(Z) > zmax)   # broadcasts to (1, ny, nz)

    filled_mask &= ~outside   # applies to every x slice

    return vals, filled_mask, stream


# %%
vals, mask, stream = get_streamtube(steady_sim, phase=None, motion=None)
steady_streamtube = {
    "vals": vals,
    "mask": mask,
    "stream": stream,
    "trajectories": stream.trajectories,
}

# %%
phase_streamtubes = {}
for phase in phases:
    vals, mask, stream = get_streamtube(sim, phase=phase, motion = motion)
    phase_streamtubes[phase] = {
        "vals": vals,
        "mask": mask,
        "stream": stream,
        "trajectories": stream.trajectories,
    }


# %%
def plot_steamlines(trajectories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for traj in trajectories[::10]:
        xyz = np.asarray(traj, dtype=float).T

        # If trajectories are in simulation coordinates rather than
        # turbine-relative coordinates, uncomment this shift:
        # xyz[:, 0] -= motion.loc[phase, "xturb"]

        axes[0].plot(xyz[:, 0], xyz[:, 1], alpha=0.6)
        axes[1].plot(xyz[:, 0], xyz[:, 2], alpha=0.6)

    axes[0].set_xlabel(r"$x/D$")
    axes[0].set_ylabel(r"$y/D$")
    axes[0].set_title(f"Phase {phase}: horizontal deflection")

    axes[1].set_xlabel(r"$x/D$")
    axes[1].set_ylabel(r"$z/D$")
    axes[1].set_title(f"Phase {phase}: vertical deflection")

    plt.tight_layout()


# %%
plot_steamlines(steady_streamtube["trajectories"])

# %%
phase = phases[5]
trajectories = phase_streamtubes[phase]["trajectories"]
plot_steamlines(trajectories)


# %% [markdown]
# ## Control Volume Analysis

# %%
def get_near_wake_profiles(
    vals,
    mask,
    x_turb=0.0,
    u_turb = 0.0,
    xlim=(-3.0, 2.0),
):
    """
    Extract streamtube-averaged profiles relative to the turbine disk, like in 
    Liew et al Figure S10.

    Parameters
    ----------
    vals : dict
        Phase-specific fields: x, ubar, vbar, pbar.
    mask : ndarray of bool
        Streamtube mask, shaped (x, y, z).
    x_turb : float
        Turbine streamwise location in the same coordinates as vals["x"].
    xlim : tuple
        Plotting/extraction limits in x/D relative to the disk.
    """
    x = np.asarray(vals["x"], dtype=float)
    y = np.asarray(vals["y"], dtype=float)
    z = np.asarray(vals["z"], dtype=float)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    r = np.hypot(Y, Z)

    mask = np.asarray(mask, dtype=bool)

    u = np.asarray(vals["ubar"], dtype=float)
    v = np.asarray(vals["vbar"], dtype=float)
    w = np.asarray(vals["wbar"], dtype=float)
    v_r = np.divide(Y * v + Z * w, r, out=np.full_like(v, np.nan), where=r > 0)
    p = np.asarray(vals["pbar"], dtype=float)

    # Switch into turbine frame of reference if non-zero values provided
    x_D = x - x_turb
    u_D = (u - u_turb) #/ (1 - u_turb)

    idx = np.flatnonzero(
        (x_D >= xlim[0]) & (x_D <= xlim[1])
    )

    u_profile = np.full(len(idx), np.nan)
    u_def_profile = np.full(len(idx), np.nan)
    vr_profile = np.full(len(idx), np.nan)
    p_profile = np.full(len(idx), np.nan)
    c_profile = np.full(len(idx), np.nan)

    for j, i in enumerate(idx):
        # Use the same cross-sectional points for all three quantities.
        valid = (
            mask[i]
            & np.isfinite(u_D[i])
            & np.isfinite(v_r[i])
            & np.isfinite(v[i])
            & np.isfinite(p[i])
        )

        if not np.any(valid):
            continue

        ui = u_D[i][valid]
        vi = v[i][valid]
        wi = w[i][valid]
        vri = v_r[i][valid]
        pi = p[i][valid]


        # (a) Streamwise velocity deficit; U_infinity = 1.
        u_profile[j] = np.mean(ui)
        u_def_profile[j] = ((1 - u_turb) - np.mean(ui)) #/ (1 - u_turb)
        # (b) Streamtube-averaged pressure.
        p_profile[j] = np.mean(pi)
        # (c) Lateral + Vertical RMS
        c_profile[j] = np.sqrt(np.mean(vi**2 + wi**2))
        # (d) Radial Lateral + Vertical Velocity
        vr_profile[j] = np.mean(vri)

    


    return {
        "x_D": x_D[idx],
        "u": u_profile,
        "u_deficit": u_def_profile,
        "p": p_profile,
        "vr_velocity": vr_profile,
        "rms_velocity": c_profile,
    }


# %%
def plot_phase_profiles(
    phases,
    motion,
    phase_streamtubes,
    xlim=(-3.0, 2.0),
    ground_frame = True,
):
    """
    Plot streamtube profiles for all phases, colored by phase.
    Similar to Liew et al Figure S10.

    Parameters
    ----------
    phase_vals : dict
        {phase: vals_for_that_phase}
    phase_masks : dict
        {phase: mask_for_that_phase}
    data : pandas.DataFrame
        Time-resolved turbine data containing Phase and DeltaX.
    x_turb_ref : float
        Reference turbine location in the LES grid coordinates.
    """
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Mean turbine displacement at each phase.
    # phase_displacement = (data.groupby("PhaseBin", sort=True)["DeltaX"].mean())

    cmap = plt.get_cmap("vanimo")
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 16), sharex=True, layout="constrained"
    )
    dp_arr = np.zeros_like(phases)
    ud_arr = np.zeros_like(phases)
    for (i, phase) in enumerate(phases):
        # if phase not in [0.0, 0.2, 0.4, 0.6, 0.8]:
        #     continue
        xturb = motion.loc[phase, "xturb"]
        if ground_frame:
            uturb = 0
        else:
            uturb = motion.loc[phase, "uturb"]

        urel = (1 - uturb)
        u4 = urel * ss_u4
        u4_deficit = (urel - u4) #/ urel

        vals = phase_streamtubes[phase]["vals"]
        mask = phase_streamtubes[phase]["mask"]

        profiles = get_near_wake_profiles(
            vals=vals,
            mask=mask,
            x_turb=xturb,
            u_turb = uturb,
            xlim=xlim,
        )

        color = cmap(norm(phase))
        x_D = profiles["x_D"]

        axes[0].axhline(y = u4_deficit, color=color, lw = 2, linestyle = "--")
        axes[0].plot(x_D, profiles["u_deficit"], color=color, lw = 2)
        axes[1].plot(x_D, profiles["p"], color=color, lw = 2)
        axes[2].plot(x_D, profiles["rms_velocity"], color=color, lw = 2)
        axes[3].plot(x_D, profiles["vr_velocity"], color=color, lw = 2)

        # grab key values from profiles
        # Ensure x_D is sorted for interpolation
        sort_idx = np.argsort(x_D)
        x_sorted = np.asarray(x_D)[sort_idx]
        u_sorted = np.asarray(profiles["u"])[sort_idx]

        # Interpolate velocity at the disk plane, x/D = 0
        ud_arr[i] = np.interp(0.0, x_sorted, u_sorted) * (1 / M_inv)

        # Interpolate pressure onto a uniform grid near the disk
        near_disk = np.abs(x_D) < 0.25
        dp_arr[i] = (
            np.max(profiles["p"][near_disk])
            - np.min(profiles["p"][near_disk])
        )

    axes[0].set_ylabel(
        r"$1-\langle\overline{u}\rangle_A$"
    )
    axes[0].set_ylim(-0.25, 1.25)
    axes[1].set_ylabel(
        r"$\langle\overline{p}\rangle_A$"
    )
    axes[1].set_ylim(-0.4, 0.4)
    axes[1].axhline(y = 0, c = "k")
    axes[2].set_ylabel(
            r"$\sqrt{\langle\overline{v}^2"
            r"+\overline{w}^{\,2}\rangle_A}$"
    )
    # axes[2].set_ylim(0, 0.2)
    axes[3].set_ylabel(
            r"$\langle\overline{v_r}\rangle_A$"
    )
    # axes[3].set_ylim(-0.2, 0.2)

    for ax in axes:
        ax.set_xlabel(r"$x/D$")
        ax.set_xlim(xlim)
        ax.axvline(0, color="black", lw=0.8, alpha=0.5)
        ax.tick_params(direction="in", top=True, right=True)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, ax=axes, location="right", shrink=0.9, aspect=30, pad=0.02
    )
    cbar.set_label("Phase")

    return fig, axes, dp_arr, ud_arr


# %%
1 - (motion["udisk"] + motion["uturb"])

# %%
fig, axes, les_dp_arr, les_ud_arr = plot_phase_profiles(phases, motion, phase_streamtubes, xlim=(-1.5, 3), ground_frame = False)

# %%
umm_dp_arr = np.array([0.5 * rho * ss_CT * (UInf - ut)**2 for ut in motion["uturb"]])
umm_ud_arr = np.array([(1 - ss_an) * (UInf - ut) for ut in motion["uturb"]])

# %%
les_dp_udisk = (
    0.5 * Ct_prime * (les_ud_arr / UInf)**2
)

# %%
plt.scatter(phases, les_ud_arr, label = "LES Fields")
plt.scatter(phases, motion["udisk"], label = "LES UDisk")
plt.scatter(phases, umm_ud_arr, label = "UMM")
plt.title("Disk Velocity")
plt.legend()

# %%
plt.scatter(phases, les_dp_arr, label = "LES Fields")
plt.scatter(phases, les_dp_udisk, label = "LES UDisk")
plt.scatter(phases, umm_dp_arr, label = "UMM")
plt.title("Pressure Drop")
plt.legend()


# %%
def get_near_wake_quantities(phase_streamtubes, motion, D = 1, ground_frame = True):
    """Calculate near-wake quantities from masked 3D LES fields."""

    vals = phase_streamtubes[phase]["vals"]
    mask = phase_streamtubes[phase]["mask"]

    xturb = motion.loc[phase, "xturb"]
    if ground_frame:
        uturb = 0
    else:
        uturb = motion.loc[phase, "uturb"]

    fields = {
        name: np.asarray(vals[name], dtype=float)
        for name in ("ubar", "vbar", "wbar", "pbar")
    }
    # move into turbine frame
    x = np.asarray(vals["x"], dtype=float) - xturb
    fields["ubar"] -= uturb

    # Streamwise locations satisfying 0 < x/D < 5.
    idx = np.flatnonzero((x / D > 0) & (x / D < 5))

    # Streamtube-averaged profiles at each streamwise location.
    mask = np.asarray(mask, dtype=bool)
    def streamtube_profile(field):
        profile = []
        for i in idx:
            valid = mask[i] & np.isfinite(field[i])
            profile.append(
                np.mean(field[i][valid]) if np.any(valid) else np.nan
            )
        return np.asarray(profile)

    profiles = {
        name: streamtube_profile(field)
        for name, field in fields.items()
    }

    # u4: minimum streamwise velocity; x0 is its streamwise location.
    ju = np.nanargmin(profiles["ubar"])
    iu = idx[ju]

    # Restrict v4 and w4 searches to x < x0.
    upstream = x[idx] < x[iu]
    idx_upstream = idx[upstream]

    # v4: maximum lateral velocity upstream of x0.
    jv = np.nanargmax(profiles["vbar"][upstream])
    iv = idx_upstream[jv]

    # w4: maximum vertical velocity upstream of x0.
    jw = np.nanargmax(profiles["wbar"][upstream])
    iw = idx_upstream[jw]

    return {
        "u4": profiles["ubar"][ju],
        "v4": profiles["vbar"][jv],
        "w4": profiles["wbar"][jw],
        "p4": profiles["pbar"][ju],
        "x0": x[iu],
        "x_v4": x[iv],
        "x_w4": x[iw],
    }


# %%
results = []

for phase in phases:
    # Extract the near-wake quantities.
    quantities = get_near_wake_quantities(phase_streamtubes, motion)

    results.append({
        "phase": phase,
        **quantities,
    })

# One row per phase.
results = pd.DataFrame(results).set_index("phase")
results


# %%
def plot_phase_streamwise(
    sim,
    phases,
    motion,
    *,
    lims,
    ground_frame = True,
    z_value = 0,
    D=1,
    U_h=1,
    color_mode="deficit",
    color_limits=(-0.5, 0.5),
    cmap="RdBu_r",
    nlevels=31,
    quiver_stride=8,
    quiver_scale=0.05,
    quiver_width=0.003,
    vector_min=0.01,
    figsize=(10, 14),
):
    """
    Plot phase-averaged y-z cross-sections at a specified streamwise location.

    Parameters
    ----------
    sim : object
        Simulation object supporting sim.slice(...).
    x_value : float
        Requested streamwise coordinate, in the same units as vals["x"].
    D : float
        Rotor diameter, in the same units as x, y, and z.
    U_h : float
        Reference hub-height/free-stream velocity.
    lims : dict
        Dictionary containing "xlim", "ylim", and "zlim".
    color_mode : {"deficit", "normalized", "raw"}
        "deficit":   (ubar - U_h) / U_h
        "normalized": ubar / U_h
        "raw":        ubar
    color_limits : tuple
        Shared color limits for all panels.
    cmap : str
        Matplotlib colormap.
    nlevels : int
        Number of contour levels.
    quiver_stride : int
        Plot every Nth grid point in each cross-plane direction.
    quiver_scale : float
        Smaller values produce longer arrows.
    quiver_width : float
        Arrow shaft width as a fraction of the axes width.
    vector_min : float
        Suppress vectors with cross-plane speed below this fraction
        of U_h. For example, 0.01 suppresses speeds below 1% of U_h.
    figsize : tuple
        Overall figure size.

    Returns
    -------
    fig, axes
        Matplotlib figure and 5x2 axes array.
    """
    fig, axes = plt.subplots(
        5, 2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes_flat = axes.ravel()
    levels = np.linspace(color_limits[0], color_limits[1], nlevels)
    contour_handles = []

    stride = max(1, int(quiver_stride))
    for i, phase in enumerate(phases):
        ax = axes_flat[i]

        xturb = motion.loc[phase, "xturb"]
        if ground_frame:
            uturb = 0
        else:
            uturb = motion.loc[phase, "uturb"]

        # Retrieve phase-averaged fields.
        vals = sim.slice(
            budget_terms=["ubar", "vbar", "wbar"],
            phase=phase,
            xlim=lims["xlim"],
            ylim=lims["ylim"],
            zlim=lims["zlim"],
        )

        x = np.asarray(vals["x"]) - xturb
        y = np.asarray(vals["y"])
        z = np.asarray(vals["z"])

        u = np.asarray(vals["ubar"]) - uturb
        v = np.asarray(vals["vbar"])
        w = np.asarray(vals["wbar"])

        # Select the nearest available z plane.
        iz = np.argmin(np.abs(z - z_value))
        z_plane = z[iz]

        # Extract the y-z cross-section.
        # Normalize coordinates so the plotted axes are y/D and z/D.
        X, Y = np.meshgrid(x, y, indexing="ij")

        u_p = u[:, :, iz]
        v_p = v[:, :, iz]
        w_p = w[:, :, iz]

        # Streamwise velocity is the background contour field.
        if color_mode == "deficit":
            c_p = (u_p - U_h) / U_h
        elif color_mode == "normalized":
            c_p = u_p / U_h
        elif color_mode == "raw":
            c_p = u_p
        else:
            raise ValueError(
                "color_mode must be 'deficit', 'normalized', or 'raw'."
            )

        cf = ax.contourf(
            X,
            Y,
            c_p,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        contour_handles.append(cf)

        # ----------------------------------------------------------
        # Quiver: thin arrows, with negligible vectors suppressed.
        # ----------------------------------------------------------
        Xs = X[::stride, ::stride]
        Ys = Y[::stride, ::stride]

        # Dimensionless in-plane velocity components.
        Vs = v_p[::stride, ::stride] / U_h
        Ws = w_p[::stride, ::stride] / U_h

        # Cross-plane velocity magnitude.
        # speed = np.hypot(Vs, Ws)

        # Remove nonfinite and negligible vectors BEFORE plotting.
        # This prevents near-zero vectors from appearing as dots.
        valid = (
            np.isfinite(Xs)
            & np.isfinite(Ys)
            & np.isfinite(Vs)
            & np.isfinite(Ws)
            & (np.abs(Vs) >= vector_min)
        )

        if np.any(valid):
            ax.quiver(
                Xs[valid],
                Ys[valid],
                np.zeros_like(Xs[valid]),
                Vs[valid],
                color="k",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                pivot="mid",
                units="width",
                width=quiver_width,
                headwidth=3.5,
                headlength=4.0,
                headaxislength=3.5,
                minlength=0,
                minshaft=1.0,
            )

        ax.set_title(rf"$\phi = {float(phase):.1f}$")
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlim(
            lims["xlim"][0] / D,
            lims["xlim"][1] / D,
        )
        ax.set_ylim(
            lims["ylim"][0] / D,
            lims["ylim"][1] / D,
        )

        if i // 2 == 4:
            ax.set_xlabel(r"$x/D$ [-]")

        if i % 2 == 0:
            ax.set_ylabel(r"$y/D$ [-]")

    fig.suptitle(
        rf"Phase-averaged cross-sections at $z/D = {z_plane / D:.3f}$",
        fontsize=14,
    )

    # One shared colorbar.
    cbar = fig.colorbar(
        contour_handles[0],
        ax=axes_flat.tolist(),
        orientation="vertical",
        pad=0.02,
        shrink=0.9,
    )

    if color_mode == "deficit":
        cbar.set_label(r"$(\bar{u}-U_h)/U_h$ [-]")
    elif color_mode == "normalized":
        cbar.set_label(r"$\bar{u}/U_h$ [-]")
    else:
        cbar.set_label(r"$\bar{u}$")

    return fig, axes


# %%
def plot_phase_cross_sections(
    sim,
    phases,
    motion,
    x_value,
    *,
    lims,
    ground_frame = True,
    D=1,
    U_h=1,
    color_mode="deficit",
    color_limits=(-0.5, 0.5),
    cmap="RdBu_r",
    nlevels=31,
    quiver_stride=8,
    quiver_scale=0.05,
    quiver_width=0.003,
    vector_min=0.01,
    figsize=(10, 14),
):
    """
    Plot phase-averaged y-z cross-sections at a specified streamwise location.

    Parameters
    ----------
    sim : object
        Simulation object supporting sim.slice(...).
    x_value : float
        Requested streamwise coordinate, in the same units as vals["x"].
    D : float
        Rotor diameter, in the same units as x, y, and z.
    U_h : float
        Reference hub-height/free-stream velocity.
    lims : dict
        Dictionary containing "xlim", "ylim", and "zlim".
    color_mode : {"deficit", "normalized", "raw"}
        "deficit":   (ubar - U_h) / U_h
        "normalized": ubar / U_h
        "raw":        ubar
    color_limits : tuple
        Shared color limits for all panels.
    cmap : str
        Matplotlib colormap.
    nlevels : int
        Number of contour levels.
    quiver_stride : int
        Plot every Nth grid point in each cross-plane direction.
    quiver_scale : float
        Smaller values produce longer arrows.
    quiver_width : float
        Arrow shaft width as a fraction of the axes width.
    vector_min : float
        Suppress vectors with cross-plane speed below this fraction
        of U_h. For example, 0.01 suppresses speeds below 1% of U_h.
    figsize : tuple
        Overall figure size.

    Returns
    -------
    fig, axes
        Matplotlib figure and 5x2 axes array.
    """
    fig, axes = plt.subplots(
        5, 2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes_flat = axes.ravel()
    levels = np.linspace(color_limits[0], color_limits[1], nlevels)
    contour_handles = []

    stride = max(1, int(quiver_stride))
    for i, phase in enumerate(phases):
        ax = axes_flat[i]

        xturb = motion.loc[phase, "xturb"]
        if ground_frame:
            uturb = 0
        else:
            uturb = motion.loc[phase, "uturb"]
        

        # Retrieve phase-averaged fields.
        vals = sim.slice(
            budget_terms=["ubar", "vbar", "wbar"],
            phase=phase,
            xlim=lims["xlim"],
            ylim=lims["ylim"],
            zlim=lims["zlim"],
        )

        x = np.asarray(vals["x"]) - xturb
        y = np.asarray(vals["y"])
        z = np.asarray(vals["z"])

        u = np.asarray(vals["ubar"]) - uturb
        v = np.asarray(vals["vbar"])
        w = np.asarray(vals["wbar"])

        # Select the nearest available streamwise plane.
        ix = np.argmin(np.abs(x - x_value))
        x_plane = x[ix]

        # Extract the y-z cross-section.
        # Normalize coordinates so the plotted axes are y/D and z/D.
        Y, Z = np.meshgrid(y / D, z / D, indexing="ij")

        u_p = u[ix, :, :]
        v_p = v[ix, :, :]
        w_p = w[ix, :, :]

        # Streamwise velocity is the background contour field.
        if color_mode == "deficit":
            c_p = (u_p - U_h) / U_h
        elif color_mode == "normalized":
            c_p = u_p / U_h
        elif color_mode == "raw":
            c_p = u_p
        else:
            raise ValueError(
                "color_mode must be 'deficit', 'normalized', or 'raw'."
            )

        cf = ax.contourf(
            Y,
            Z,
            c_p,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        contour_handles.append(cf)

        # ----------------------------------------------------------
        # Quiver: thin arrows, with negligible vectors suppressed.
        # ----------------------------------------------------------
        Ys = Y[::stride, ::stride]
        Zs = Z[::stride, ::stride]

        # Dimensionless in-plane velocity components.
        Vs = v_p[::stride, ::stride] / U_h
        Ws = w_p[::stride, ::stride] / U_h

        # Cross-plane velocity magnitude.
        speed = np.hypot(Vs, Ws)

        # Remove nonfinite and negligible vectors BEFORE plotting.
        # This prevents near-zero vectors from appearing as dots.
        valid = (
            np.isfinite(Ys)
            & np.isfinite(Zs)
            & np.isfinite(Vs)
            & np.isfinite(Ws)
            & (speed >= vector_min)
        )

        if np.any(valid):
            ax.quiver(
                Ys[valid],
                Zs[valid],
                Vs[valid],
                Ws[valid],
                color="k",
                angles="xy",
                scale_units="xy",
                scale=quiver_scale,
                pivot="mid",
                units="width",
                width=quiver_width,
                headwidth=3.5,
                headlength=4.0,
                headaxislength=3.5,
                minlength=0,
                minshaft=1.0,
            )

        ax.set_title(rf"$\phi = {float(phase):.1f}$")
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlim(
            lims["ylim"][0] / D,
            lims["ylim"][1] / D,
        )
        ax.set_ylim(
            lims["zlim"][0] / D,
            lims["zlim"][1] / D,
        )

        if i // 2 == 4:
            ax.set_xlabel(r"$y/D$ [-]")

        if i % 2 == 0:
            ax.set_ylabel(r"$z/D$ [-]")

    fig.suptitle(
        rf"Phase-averaged cross-sections at $x/D = {x_plane / D:.3f}$",
        fontsize=14,
    )

    # One shared colorbar.
    cbar = fig.colorbar(
        contour_handles[0],
        ax=axes_flat.tolist(),
        orientation="vertical",
        pad=0.02,
        shrink=0.9,
    )

    if color_mode == "deficit":
        cbar.set_label(r"$(\bar{u}-U_h)/U_h$ [-]")
    elif color_mode == "normalized":
        cbar.set_label(r"$\bar{u}/U_h$ [-]")
    else:
        cbar.set_label(r"$\bar{u}$")

    return fig, axes

# %%
fig, axes = plot_phase_streamwise(
    sim=sim, phases = phases, motion = motion,
    lims=dict(xlim=(-1, 3), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5)),
    color_mode="deficit",
    color_limits=(-0.3, 0.3),
    quiver_stride=3,
    quiver_scale=1,
    quiver_width=0.2,
    vector_min=0.075,
)

plt.show()

# %%
fig, axes = plot_phase_cross_sections(
    sim=sim, phases = phases, motion = motion,
    x_value=0,
    lims=dict(xlim=(-2, 8), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5)),
    color_mode="deficit",
    color_limits=(-0.5, 0.5),
    quiver_stride=8,
    quiver_scale=0.3,
    quiver_width=0.01,
    vector_min=0.05
)

plt.show()


# %%
def compute_steady_bernoulli_budget(
    phases, phase_streamtubes, motion,
    *,
    Ct_prime,
    ground_frame = False,
    U_inf=1.0,
    rho=1.0,
    p_ref=0.0,
    normalize_by_term_IV = True
):
    """
    Evaluate the steady Bernoulli equation at every phase and x-plane.

    Terms 
    -----
    term_I   = 0.5 <u^2 + v^2 + w^2>_A
    term_II  = -0.5 U_inf^2
    term_III = <p - p_ref>_A / rho
    term_IV  = 0.5 Ct_prime * U_disk_relative^2

    The disk pressure-drop term is estimated from the phase-averaged
    rotor-plane velocity relative to the translating turbine.

    Coordinates are assumed to be turbine-centered, with the disk at
    x = 0 and the rotor center at y = z = 0.
    """
    rows = []

    for phase in phases:
        xt = motion.loc[phase, "xturb"]
        ud_sqr = motion.loc[phase, "udisk_sqr"]
        if ground_frame:
            ut = 0
        else:
            ut = motion.loc[phase, "uturb"]

        vals = phase_streamtubes[phase]["vals"]
        mask = phase_streamtubes[phase]["mask"]

        x = np.asarray(vals["x"], dtype=float) - xt
        u = np.asarray(vals["ubar"], dtype=float) - ut
        v = np.asarray(vals["vbar"], dtype=float)
        w = np.asarray(vals["wbar"], dtype=float)
        p = np.asarray(vals["pbar"], dtype=float)

        # Indices of planes in the window, in full-grid numbering
        ix_sel = np.where((x >= 0.5) & (x <= 2.5))[0]

        term_I   = np.full(ix_sel.size, np.nan)
        term_III = np.full(ix_sel.size, np.nan)

        for j, ix in enumerate(ix_sel):        # j: output slot, ix: field slice
            valid = (
                mask[ix]
                & np.isfinite(u[ix]) & np.isfinite(v[ix])
                & np.isfinite(w[ix]) & np.isfinite(p[ix])
            )
            if not valid.any():
                continue

            speed_sq = (u[ix][valid]**2
                        + v[ix][valid]**2
                        + w[ix][valid]**2)

            term_I[j]   = 0.5 * speed_sq.mean()
            term_III[j] = (p[ix][valid] - p_ref).mean() / rho

        # Everything that goes in the DataFrame uses the selected x
        x_sel = x[ix_sel]
        term_II = np.full(ix_sel.size, -0.5 * (U_inf - ut)**2)
        term_IV = np.full(ix_sel.size, 0.5 * Ct_prime * ud_sqr)

        residual = (
            term_I
            + term_II
            + term_III
            + term_IV
        )

        if normalize_by_term_IV:
            residual = residual / term_IV
            term_I = term_I / term_IV
            term_II = term_II / term_IV
            term_III = term_III / term_IV
            term_IV = term_IV / term_IV

        rows.append(
            pd.DataFrame({
                "phase": float(phase),
                "xD": x_sel,
                "term_I": term_I,
                "term_II": term_II,
                "term_III": term_III,
                "term_IV": term_IV,
                "residual": residual,
            })
        )

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["phase", "xD"])
        .reset_index(drop=True)
    )


# %%
normalize_by_term_IV = True

# %%
steady_budgets = compute_steady_bernoulli_budget(
    phases, phase_streamtubes, motion,
    Ct_prime = Ct_prime,
    normalize_by_term_IV = normalize_by_term_IV,
    ground_frame = False
)


# %%
def plot_steady_budget_phase(
    budget,
    phases,
    *,
    U_inf=1.0,
    xlim=(0.5, 2.5),
    figsize=(10, 14),
    normalize_by_term_IV = True,
):

    terms = [
        ("term_I",   r"Term I: kinetic energy"),
        ("term_II",  r"Term II: freestream reference"),
        ("term_III", r"Term III: pressure"),
        ("term_IV",  r"Term IV: disk pressure drop"),
        ("residual", r"Residual"),
    ]

    fig, axes = plt.subplots(
        5, 2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for i, phase in enumerate(phases):
        ax = axes_flat[i]

        sub = budget[
            np.isclose(budget["phase"], phase)
            & budget["xD"].between(*xlim)
        ].sort_values("xD")

        for key, label in terms:
            style = "-." if key == "residual" else "-"
            width = 2.4 if key == "residual" else 1.7

            ax.plot(
                sub["xD"],
                sub[key] / U_inf**2,
                ls=style,
                lw=width,
                label=label,
            )

        ax.axhline(0.0, color="black", lw=0.8)   # no label -> not in legend
        ax.set_title(rf"Phase {phase:.1f}")
        ax.tick_params(direction="in", top=True, right=True)

    # Hide any unused axes (if fewer than 10 phases)
    for ax in axes_flat[len(phases):]:
        ax.set_visible(False)

    # Only label outer axes (shared x/y)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x/D$")
    for ax in axes[:, 0]:
        if normalize_by_term_IV:
            ax.set_ylabel(r"Energy term / $0.5 C_T' (u_d)^2$")
        else:
            ax.set_ylabel(r"Energy term")

    # One shared legend, outside the subplot grid
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="outside lower center",   # or "outside right center"
        ncol=3,
        frameon=False,
    )

    fig.suptitle("Steady Bernoulli budget by phase")

    # NOTE: no fig.tight_layout() — it conflicts with constrained_layout
    return fig, axes


fig, axes = plot_steady_budget_phase(
    steady_budgets,
    phases,
    U_inf=UInf,
    normalize_by_term_IV = normalize_by_term_IV,
)
plt.show()


# %%
def get_accelerations(
    phases,
    f,
    motion,
    phase_streamtubes,
    save_dir,
):
    """
    Calculate turbine-frame acceleration fields for every phase
    and save each field as a separate .npy file.

    Returns
    -------
    dict
        x_common, y, z, dt, and a dictionary mapping phases to
        their saved acceleration-field paths.
    """
    phases = list(phases)
    nphase = len(phases)
    dt = 1.0 / (f * nphase)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Construct the common turbine-relative grid.
    # ------------------------------------------------------------
    x_rel = {
        phase: (
            np.asarray(
                phase_streamtubes[phase]["vals"]["x"],
                dtype=float,
            )
            - motion.loc[phase, "xturb"]
        )
        for phase in phases
    }

    xlo = max(xp[0] for xp in x_rel.values())
    xhi = min(xp[-1] for xp in x_rel.values())

    x_common = x_rel[phases[0]]
    x_common = x_common[
        (x_common >= xlo) & (x_common <= xhi)
    ]

    vals0 = phase_streamtubes[phases[0]]["vals"]
    y = np.asarray(vals0["y"], dtype=float)
    z = np.asarray(vals0["z"], dtype=float)

    # Save the coordinate grid and phase metadata once.
    np.savez(
        save_dir / "acceleration_grid.npz",
        x_common=x_common,
        y=y,
        z=z,
        phases=np.asarray(phases),
        dt=dt,
    )

    # ------------------------------------------------------------
    # 2. Register one phase's velocity onto the common grid.
    # ------------------------------------------------------------
    def register_velocity(phase):
        vals = phase_streamtubes[phase]["vals"]

        xt = motion.loc[phase, "xturb"]
        ut = motion.loc[phase, "uturb"]

        x = np.asarray(vals["x"], dtype=float)
        x_target = x_common + xt

        u = linear_interp(
            x_target,
            x,
            np.asarray(vals["ubar"]),
            axis=0,
        ) - ut

        v = linear_interp(
            x_target,
            x,
            np.asarray(vals["vbar"]),
            axis=0,
        )

        w = linear_interp(
            x_target,
            x,
            np.asarray(vals["wbar"]),
            axis=0,
        )

        return u, v, w

    # ------------------------------------------------------------
    # 3. Use periodic central differences and save each phase.
    # ------------------------------------------------------------
    paths = {}

    # Start with the last, first, and second phases.
    u_prev, v_prev, w_prev = register_velocity(phases[-1])
    u_curr, v_curr, w_curr = register_velocity(phases[0])
    u_next, v_next, w_next = register_velocity(phases[1])

    for k, phase in enumerate(phases):
        print(f"Phase {k + 1}/{nphase}: {phase}")

        # Allocate a vector-valued acceleration field.
        accel = np.empty(
            u_curr.shape + (3,),
            dtype=float,
        )

        # du/dt at fixed turbine-relative coordinates, plus
        # the fictitious-body-force contribution.
        accel[..., 0] = u_next - u_prev
        accel[..., 0] /= 2.0 * dt
        accel[..., 0] += motion.loc[phase, "aturb"]

        accel[..., 1] = v_next - v_prev
        accel[..., 1] /= 2.0 * dt

        accel[..., 2] = w_next - w_prev
        accel[..., 2] /= 2.0 * dt

        # Save the field without compression.
        path = save_dir / f"accel_phase_{k:03d}.npy"
        np.save(path, accel)
        paths[phase] = path

        # Rotate the three-phase window.
        if k < nphase - 1:
            u_prev, v_prev, w_prev = u_curr, v_curr, w_curr
            u_curr, v_curr, w_curr = u_next, v_next, w_next

            u_next, v_next, w_next = register_velocity(
                phases[(k + 2) % nphase]
            )

    return {
        "x_common": x_common,
        "y": y,
        "z": z,
        "dt": dt,
        "paths": paths,
        "save_dir": save_dir,
    }


# %%
accel_data = get_accelerations(
    phases=phases,
    f=f,
    motion=motion,
    phase_streamtubes=phase_streamtubes,
    save_dir="acceleration_fields",
)


# %%
def plot_acceleration_fields(
    accel_data,
    phase,
    *,
    x_slice=0.0,
    y_slice=0.0,
    z_slice=0.0,
    save_path=None,
):
    """
    Plot all three acceleration components on two 2D slices.

    Top row:    x-z plane at y ~= y_slice.
    Bottom row: y-z plane at x ~= x_slice.

    Parameters
    ----------
    accel_data : dict
        Output from get_accelerations().
    phase : phase value
        Phase to plot.
    x_slice : float
        Streamwise location for the y-z slice.
    y_slice : float
        Cross-stream location for the x-z slice.
    save_path : str or Path, optional
        Save the figure if provided.

    Returns
    -------
    fig, axes
    """
    # Load the coordinate grid and selected phase.
    x = np.asarray(accel_data["x_common"])
    y = np.asarray(accel_data["y"])
    z = np.asarray(accel_data["z"])

    accel = np.load(
        accel_data["paths"][phase],
        mmap_mode="r",
    )

    # Select the nearest available grid coordinates.
    ix = np.argmin(np.abs(x - x_slice))
    iy = np.argmin(np.abs(y - y_slice))
    iz = np.argmin(np.abs(z - z_slice))

    components = [r"$a_x$", r"$a_y$", r"$a_z$"]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 8),
        constrained_layout=True,
    )

    for j, component in enumerate(components):
        # x-z slice at y ~= y_slice.
        field_xz = np.asarray(accel[:, iy, :, j]).T

        # x-y slice at z ~= z_slice.
        field_xy = np.asarray(accel[:, :, iz, j]).T

        # y-z slice at x ~= x_slice.
        field_yz = np.asarray(accel[ix, :, :, j]).T

        # Use symmetric color limits, robust to isolated extremes.
        values = np.concatenate((
            field_xz.ravel(),
            field_yz.ravel(),
        ))
        values = values[np.isfinite(values)]

        if values.size == 0:
            vmax = 1.0
        else:
            vmax = np.nanpercentile(np.abs(values), 99)
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0

        # Top row: streamwise-vertical plane.
        if j != 1:
            im1 = axes[0, j].pcolormesh(
                x, z, field_xz,
                shading="auto",
                cmap="RdBu_r",
                vmin=-0.5,
                vmax=0.5,
            )
            axes[0, j].set_title(
                f"{component}, y = {y[iy]:.3f}"
            )
            axes[0, j].set_ylabel(r"$z/D$")
        else:
            im1 = axes[0, j].pcolormesh(
                x, y, field_xy,
                shading="auto",
                cmap="RdBu_r",
                vmin=-0.5,
                vmax=0.5,
            )
            axes[0, j].set_title(
                f"{component}, z = {z[iz]:.3f}"
            )
            axes[0, j].set_ylabel(r"$y/D$")

        axes[0, j].set_xlabel(r"$x/D$")
        fig.colorbar(im1, ax=axes[0, j], label=component)

        # Bottom row: cross-stream plane.
        im2 = axes[1, j].pcolormesh(
            y, z, field_yz,
            shading="auto",
            cmap="RdBu_r",
            vmin=-0.5,
            vmax=0.5,
        )

        axes[1, j].set_title(
            f"{component}, x = {x[ix]:.3f}"
        )
        axes[1, j].set_xlabel(r"$y/D$")
        axes[1, j].set_ylabel(r"$z/D$")
        fig.colorbar(im2, ax=axes[1, j], label=component)

    fig.suptitle(f"Acceleration fields — phase {phase}", fontsize=15)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes


# %%
fig, axes = plot_acceleration_fields(
    accel_data,
    phase=phases[8],
    x_slice=0.0,
    y_slice=0.0,
)

plt.show()

# %%
grid = np.load("acceleration_fields/acceleration_grid.npz")

phase = phases[0]
accel = np.load(
    accel_data["paths"][phase],
    mmap_mode="r",
)

accel_interp = RegularGridInterpolator(
    (
        grid["x_common"],
        grid["y"],
        grid["z"],
    ),
    accel,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)


# %%
def integrate_Bu_gauss2(xyz, accel_interp, max_ds):
    """
    Integrate a_B · dx along a piecewise-linear streamline.

    Uses vectorized two-point Gauss-Legendre quadrature,
    subdividing each original segment to respect max_ds.

    Parameters
    ----------
    xyz : ndarray, shape (N, 3)
        Streamline coordinates.
    accel_interp : RegularGridInterpolator
        Interpolator returning [ax, ay, az].
    max_ds : float
        Maximum substep length.

    Returns
    -------
    Bu : ndarray, shape (N,)
        Cumulative line integral at each original streamline point.
    """
    xyz = np.asarray(xyz, dtype=float)

    if len(xyz) < 2:
        return np.zeros(len(xyz), dtype=float)

    delta = np.diff(xyz, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    nseg = len(delta)

    # Retain the original integration resolution.
    nsteps = np.maximum(
        1, np.ceil(lengths / max_ds).astype(int)
    )

    # Identify every substep belonging to each original segment.
    segment_ids = np.repeat(np.arange(nseg), nsteps)
    segment_starts = np.repeat(
        np.cumsum(nsteps) - nsteps, nsteps
    )
    local_step = np.arange(len(segment_ids)) - segment_starts

    steps_per_segment = nsteps[segment_ids]
    fraction = local_step / steps_per_segment

    # Start and displacement vector of every substep.
    segment_delta = delta[segment_ids]
    sub_delta = segment_delta / steps_per_segment[:, None]

    p_start = (
        xyz[segment_ids]
        + fraction[:, None] * segment_delta
    )

    # Two Gauss-Legendre points on [0, 1].
    g1 = 0.5 - np.sqrt(3.0) / 6.0
    g2 = 0.5 + np.sqrt(3.0) / 6.0

    q1 = p_start + g1 * sub_delta
    q2 = p_start + g2 * sub_delta

    # Interpolate all three acceleration components in batches.
    nsub = len(sub_delta)
    points = np.concatenate((q1, q2), axis=0)
    a = accel_interp(points)

    a1 = a[:nsub]
    a2 = a[nsub:]

    # Integral over each substep: ∫ a(x) · dx.
    sub_integrals = 0.5 * np.einsum(
        "ij,ij->i",
        a1 + a2,
        sub_delta,
    )

    # Sum the substeps belonging to each original segment.
    segment_integrals = np.bincount(
        segment_ids,
        weights=sub_integrals,
        minlength=nseg,
    )

    Bu = np.empty(len(xyz), dtype=float)
    Bu[0] = 0.0
    Bu[1:] = np.cumsum(segment_integrals)

    return Bu


# %%
def integrate_Bu_rk4(
    xyz,
    grid,
    ax,
    ay,
    az,
    max_ds=None,
):
    """
    Integrate a_B · dx along an existing streamline using RK4.

    Parameters
    ----------
    xyz : ndarray, shape (N, 3)
        Streamline coordinates.
    grid : tuple
        (x, y, z) coordinates of the acceleration fields.
    ax, ay, az : ndarray
        Bernoulli acceleration components on the grid.
    max_ds : float, optional
        Maximum RK4 step in arc length.

    Returns
    -------
    Bu : ndarray, shape (N,)
        Cumulative line integral at each streamline point.
    """
    xyz = np.asarray(xyz, dtype=float)

    if max_ds is None:
        max_ds = 0.25 * min(
            np.min(np.abs(np.diff(g))) for g in grid
        )

    dxyz = np.diff(xyz, axis=0)
    lengths = np.linalg.norm(dxyz, axis=1)

    Bu = np.zeros(len(xyz), dtype=float)

    for i, (delta, length) in enumerate(zip(dxyz, lengths)):
        if length == 0:
            Bu[i + 1] = Bu[i]
            continue

        tangent = delta / length
        p0 = xyz[i]

        def dBu_ds(s, Bu_current):
            # Position along the straight segment.
            point = (
                p0 + (s / length) * delta
            )[None, :]

            # Interpolate acceleration at this position.
            aB_point = np.array([
                trilinear_interpolation(grid, ax, point)[0],
                trilinear_interpolation(grid, ay, point)[0],
                trilinear_interpolation(grid, az, point)[0],
            ])

            # dBu/ds = a_B · (dx/ds).
            return np.dot(aB_point, tangent)

        # Subdivide the segment for RK4 integration.
        nsteps = max(1, int(np.ceil(length / max_ds)))
        ds = length / nsteps

        B_current = Bu[i]
        s = 0.0

        for _ in range(nsteps):
            B_current = rk4_step(
                s, B_current, dBu_ds, ds
            )
            s += ds

        Bu[i + 1] = B_current

    return Bu


# %%
budget = steady_budgets

# %%
nphase = len(phases)
period = 1.0 / f
dt = period / nphase

x_rel = {
    phase: (
        np.asarray(
            phase_streamtubes[phase]["vals"]["x"],
            dtype=float,
        )
        - motion.loc[phase, "xturb"]
    )
    for phase in phases
}

xlo = max(xp[0] for xp in x_rel.values())
xhi = min(xp[-1] for xp in x_rel.values())

x_common = x_rel[phases[0]]
x_common = x_common[
    (x_common >= xlo) & (x_common <= xhi)
]

vals0 = phase_streamtubes[phases[0]]["vals"]
y = np.asarray(vals0["y"], dtype=float)
z = np.asarray(vals0["z"], dtype=float)

max_ds = 0.25 * min(
    np.min(np.abs(np.diff(g)))
    for g in (x_common, y, z)
)

# %%
u_rel, v_rel, w_rel = [], [], []
for phase in phases:
    vals = phase_streamtubes[phase]["vals"]

    xt = motion.loc[phase, "xturb"]
    ut = motion.loc[phase, "uturb"]

    x = np.asarray(vals["x"], dtype=float)
    x_target = x_common + xt

    u_rel.append(
        linear_interp(
            x_target,
            x,
            np.asarray(vals["ubar"]),
            axis=0,
        ) - ut
    )

    v_rel.append(
        linear_interp(
            x_target,
            x,
            np.asarray(vals["vbar"]),
            axis=0,
        )
    )

    w_rel.append(
        linear_interp(
            x_target,
            x,
            np.asarray(vals["wbar"]),
            axis=0,
        )
    )

u_rel = np.asarray(u_rel)
v_rel = np.asarray(v_rel)
w_rel = np.asarray(w_rel)

# %%
k, phase = 0, 0.0
budget = budget.copy()
budget["term_unsteady"] = np.nan
phase_rows = budget["phase"] == float(phase)

# %%
print(k, phase)
phase_rows = budget["phase"] == float(phase)

x_targets = budget.loc[
    phase_rows, "xD"
].to_numpy(dtype=float)

trajectories = phase_streamtubes[phase]["trajectories"]

# Periodic central differences at fixed turbine-relative x.
k_prev = (k - 1) % nphase
k_next = (k + 1) % nphase

# Construct a single vector-valued acceleration field.
accel_grid = np.empty(
    u_rel[k].shape + (3,),
    dtype=float,
)

accel_grid[..., 0] = (
    (u_rel[k_next] - u_rel[k_prev]) / (2.0 * dt)
    + motion.loc[phase, "aturb"]
)

accel_grid[..., 1] = (
    v_rel[k_next] - v_rel[k_prev]
) / (2.0 * dt)

accel_grid[..., 2] = (
    w_rel[k_next] - w_rel[k_prev]
) / (2.0 * dt)

# Build the interpolator once per phase.
accel_interp = RegularGridInterpolator(
    (x_common, y, z),
    accel_grid,
    method="linear",
    bounds_error=False,
    fill_value=None, 
)

# del accel_grid

# %%
traj = trajectories[0, :, :]
xyz = np.asarray(traj, dtype=float).T

# %%
Bu = integrate_Bu_gauss2(
    xyz,
    accel_interp,
    max_ds,
)

# %%
Bu


# %%

# %%
def add_unsteady_bernoulli_term(
    budget,
    phases,
    motion,
    phase_streamtubes,
    *,
    f,
    Ct_prime,
    D=1.0,
    x_refD=-2.0,
    normalize_by_term_IV=True,
):
    """
    Add the unsteady Bernoulli correction to an existing budget.

    Uses vectorized two-point Gauss-Legendre quadrature to integrate
    the unsteady acceleration along existing streamlines.
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    # ------------------------------------------------------------
    # Local helper: integrate acceleration along one streamline.
    # ------------------------------------------------------------
    def integrate_Bu(xyz, accel_interp, max_ds):
        xyz = np.asarray(xyz, dtype=float)

        if len(xyz) < 2:
            return np.zeros(len(xyz), dtype=float)

        delta = np.diff(xyz, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        nseg = len(delta)

        # Subdivide each streamline segment.
        nsteps = np.maximum(
            1, np.ceil(lengths / max_ds).astype(int)
        )

        segment_ids = np.repeat(
            np.arange(nseg), nsteps
        )

        segment_starts = np.repeat(
            np.cumsum(nsteps) - nsteps, nsteps
        )

        local_step = (
            np.arange(len(segment_ids)) - segment_starts
        )

        steps = nsteps[segment_ids]
        fraction = local_step / steps

        segment_delta = delta[segment_ids]
        sub_delta = segment_delta / steps[:, None]

        p_start = (
            xyz[segment_ids]
            + fraction[:, None] * segment_delta
        )

        # Two-point Gauss-Legendre quadrature on [0, 1].
        g1 = 0.5 - np.sqrt(3.0) / 6.0
        g2 = 0.5 + np.sqrt(3.0) / 6.0

        q1 = p_start + g1 * sub_delta
        q2 = p_start + g2 * sub_delta

        # Interpolate all acceleration components in two batches.
        nsub = len(sub_delta)
        accel = accel_interp(
            np.concatenate((q1, q2), axis=0)
        )

        a1 = accel[:nsub]
        a2 = accel[nsub:]

        # Integral of a_B · dx over each substep.
        sub_integrals = 0.5 * np.einsum(
            "ij,ij->i",
            a1 + a2,
            sub_delta,
        )

        # Accumulate substeps over the original segments.
        segment_integrals = np.bincount(
            segment_ids,
            weights=sub_integrals,
            minlength=nseg,
        )

        Bu = np.zeros(len(xyz), dtype=float)
        Bu[1:] = np.cumsum(segment_integrals)

        return Bu

    # ------------------------------------------------------------
    # 1. Construct a common turbine-relative x grid.
    # ------------------------------------------------------------
    nphase = len(phases)
    period = 1.0 / f
    dt = period / nphase

    x_rel = {
        phase: (
            np.asarray(
                phase_streamtubes[phase]["vals"]["x"],
                dtype=float,
            )
            - motion.loc[phase, "xturb"]
        )
        for phase in phases
    }

    xlo = max(xp[0] for xp in x_rel.values())
    xhi = min(xp[-1] for xp in x_rel.values())

    x_common = x_rel[phases[0]]
    x_common = x_common[
        (x_common >= xlo) & (x_common <= xhi)
    ]

    vals0 = phase_streamtubes[phases[0]]["vals"]
    y = np.asarray(vals0["y"], dtype=float)
    z = np.asarray(vals0["z"], dtype=float)

    # Same maximum substep length as the original RK4 method.
    max_ds = 0.25 * min(
        np.min(np.abs(np.diff(g)))
        for g in (x_common, y, z)
    )

    # ------------------------------------------------------------
    # 2. Register velocities onto the turbine-relative grid.
    # ------------------------------------------------------------
    u_rel, v_rel, w_rel = [], [], []

    for phase in phases:
        vals = phase_streamtubes[phase]["vals"]

        xt = motion.loc[phase, "xturb"]
        ut = motion.loc[phase, "uturb"]

        x = np.asarray(vals["x"], dtype=float)
        x_target = x_common + xt

        u_rel.append(
            linear_interp(
                x_target,
                x,
                np.asarray(vals["ubar"]),
                axis=0,
            ) - ut
        )

        v_rel.append(
            linear_interp(
                x_target,
                x,
                np.asarray(vals["vbar"]),
                axis=0,
            )
        )

        w_rel.append(
            linear_interp(
                x_target,
                x,
                np.asarray(vals["wbar"]),
                axis=0,
            )
        )

    u_rel = np.asarray(u_rel)
    v_rel = np.asarray(v_rel)
    w_rel = np.asarray(w_rel)

    # ------------------------------------------------------------
    # 3. Integrate the correction for each phase.
    # ------------------------------------------------------------
    budget = budget.copy()
    budget["term_unsteady"] = np.nan

    for k, phase in enumerate(phases):
        print(k, phase)
        phase_rows = budget["phase"] == float(phase)

        x_targets = budget.loc[
            phase_rows, "xD"
        ].to_numpy(dtype=float)

        trajectories = phase_streamtubes[phase]["trajectories"]

        # Periodic central differences at fixed turbine-relative x.
        k_prev = (k - 1) % nphase
        k_next = (k + 1) % nphase

        # Construct a single vector-valued acceleration field.
        accel_grid = np.empty(
            u_rel[k].shape + (3,),
            dtype=float,
        )

        accel_grid[..., 0] = (
            (u_rel[k_next] - u_rel[k_prev]) / (2.0 * dt)
            + motion.loc[phase, "aturb"]
        )

        accel_grid[..., 1] = (
            v_rel[k_next] - v_rel[k_prev]
        ) / (2.0 * dt)

        accel_grid[..., 2] = (
            w_rel[k_next] - w_rel[k_prev]
        ) / (2.0 * dt)

        # Build the interpolator once per phase.
        accel_interp = RegularGridInterpolator(
            (x_common, y, z),
            accel_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        del accel_grid

        Bu_profiles = []

        for traj in trajectories:
            xyz = np.asarray(traj, dtype=float).T

            # Fast, vectorized line integration.
            Bu = integrate_Bu(
                xyz,
                accel_interp,
                max_ds,
            )

            # Find the reference-plane crossing.
            x_offset = xyz[:, 0] - x_refD

            crossings = np.flatnonzero(
                x_offset[:-1] * x_offset[1:] <= 0
            )

            if crossings.size == 0:
                Bu_profiles.append(
                    np.full(x_targets.shape, np.nan)
                )
                continue

            i = crossings[0]

            if np.isclose(x_offset[i], 0.0):
                Bu_ref = Bu[i]

            elif np.isclose(x_offset[i + 1], 0.0):
                Bu_ref = Bu[i + 1]

            else:
                frac = (
                    (x_refD - xyz[i, 0])
                    / (xyz[i + 1, 0] - xyz[i, 0])
                )

                Bu_ref = (
                    Bu[i]
                    + frac * (Bu[i + 1] - Bu[i])
                )

            # Set Bu(x_ref) = 0.
            Bu = Bu - Bu_ref

            # Interpolate onto the budget x locations.
            order = np.argsort(xyz[:, 0])

            Bu_profile = np.interp(
                x_targets,
                xyz[order, 0],
                Bu[order],
                left=np.nan,
                right=np.nan,
            )

            Bu_profiles.append(Bu_profile)

        # Average across available streamlines.
        if Bu_profiles:
            with np.errstate(invalid="ignore"):
                term_unsteady = np.nanmean(
                    np.asarray(Bu_profiles),
                    axis=0,
                )
        else:
            term_unsteady = np.full(
                x_targets.shape, np.nan
            )

        # Normalize by Bernoulli term IV if requested.
        if normalize_by_term_IV:
            term_IV = (
                0.5
                * Ct_prime
                * motion.loc[phase, "udisk_sqr"]
            )

            term_unsteady = term_unsteady / term_IV

        budget.loc[
            phase_rows, "term_unsteady"
        ] = term_unsteady

    # ------------------------------------------------------------
    # 4. Update the residual.
    # ------------------------------------------------------------
    budget["residual_unsteady"] = (
        budget["residual"] + budget["term_unsteady"]
    )

    return budget


# %%
unsteady_budget = add_unsteady_bernoulli_term(
    steady_budgets,
    phases,
    motion,
    phase_streamtubes,
    f=f,
    Ct_prime=Ct_prime,
    D=D,
    x_refD=-2.0,
    normalize_by_term_IV=True,
)

# %%
unsteady_budget


# %%
def plot_budget_phase(
    budget,
    phases,
    *,
    U_inf=1.0,
    xlim=(0.5, 2.5),
    figsize=(10, 14),
    normalize_by_term_IV = True,
):

    terms = [
        ("term_I",   r"Term I: kinetic energy"),
        ("term_II",  r"Term II: freestream reference"),
        ("term_III", r"Term III: pressure"),
        ("term_IV",  r"Term IV: disk pressure drop"),
        ("residual", r"Residual"),
        ("term_unsteady", r"Unsteady Term"),
        ("residual_unsteady", r"Unsteady Residual"),
    ]

    fig, axes = plt.subplots(
        5, 2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.ravel()

    for i, phase in enumerate(phases):
        ax = axes_flat[i]

        sub = budget[
            np.isclose(budget["phase"], phase)
            & budget["xD"].between(*xlim)
        ].sort_values("xD")

        for key, label in terms:
            style = "-." if (key == "residual" or key == "residual_unsteady") else "-"
            width = 2.4 if key == "residual" else 1.7

            ax.plot(
                sub["xD"],
                sub[key] / U_inf**2,
                ls=style,
                lw=width,
                label=label,
            )

        ax.axhline(0.0, color="black", lw=0.8)   # no label -> not in legend
        ax.set_title(rf"Phase {phase:.1f}")
        ax.tick_params(direction="in", top=True, right=True)

    # Hide any unused axes (if fewer than 10 phases)
    for ax in axes_flat[len(phases):]:
        ax.set_visible(False)

    # Only label outer axes (shared x/y)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x/D$")
    for ax in axes[:, 0]:
        if normalize_by_term_IV:
            ax.set_ylabel(r"Energy term / $0.5 C_T' (u_d)^2$")
        else:
            ax.set_ylabel(r"Energy term")

    # One shared legend, outside the subplot grid
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="outside lower center",   # or "outside right center"
        ncol=3,
        frameon=False,
    )

    fig.suptitle("Steady Bernoulli budget by phase")

    # NOTE: no fig.tight_layout() — it conflicts with constrained_layout
    return fig, axes


fig, axes = plot_budget_phase(
    unsteady_budget,
    phases,
    U_inf=UInf,
    normalize_by_term_IV = normalize_by_term_IV,
)
plt.show()

# %% [markdown]
# ## Wave Budget

# %%
# keys = fluids.uvw_keys + fluids.rs_keys[0] + ["DuDt", "dpdx", "xSGS", "xAD", "pbar", "xCor"]
# lims = dict(xlim=(-2, 10), ylim=(-5, 5), zlim=(-5, 5))
# ds_phase, ds_time = sim.read_wave_budgets(keys, phase="all", **lims)
# # _, w = sim.read_phase_budgets(keys[:1], **lims)        # bin weights

# %%
# # Check the input dimensions
# print("ds_phase:", ds_phase.sizes)
# print("ds_time: ", ds_time.sizes)

# # Check the output of the budget function directly
# raw = fluids.compute_wave_budget(
#     ds_phase,
#     ds_time,
#     direction=0,
#     period=1.0,
# )

# print("\nRaw budget dimensions:", raw.sizes)
# for key in raw.data_vars:
#     print(f"{key:12s}: {raw[key].dims}")

# # Check the Wave_x object before and after its inherited compute workflow
# wb = Wave_x(ds_phase, ds_time, base_agg=0, period=1.0)
# wb._compute_budget()

# print("\nBase terms:", wb.base_terms.sizes)
# for key in wb.base_terms.data_vars:
#     print(f"{key:12s}: {wb.base_terms[key].dims}")

# %%
# wb = Wave_x(ds_phase, ds_time, base_agg=0, period=1.0)
# wb.compute()

# print("Final wb dimensions:", wb.sizes)
# for key in wb.data_vars:
#     print(f"{key:12s}: {wb[key].dims}")

# print("\nBase terms dimensions:", wb.base_terms.sizes)
# for key in wb.base_terms.data_vars:
#     print(f"{key:12s}: {wb.base_terms[key].dims}")

# %%
# wb = Wave_x(ds_phase, ds_time, base_agg=0, period=1.0)
# wb.compute()

# %%
wb

# %%
{v: wb[v].dims for v in wb.data_vars}

# %%
(wb["cor"].sel(phase=[0.0, 0.5], method="nearest")
         .sel(z=0.0, method="nearest")
         .sel(x=slice(0, 10), y=slice(-2, 2))
         .plot.pcolormesh(x="x", y="y", col="phase", cmap="RdBu_r",
                          robust=True, center=0, aspect=2.5, size=3))

# %%
phases = np.array(wb.phase)

# %%
phase_masks = [[] for p in phases]
for (i, phase) in enumerate(phases):
    vals = sim.slice(budget_terms=["ubar", "vbar", "wbar"], phase=phase, xlim=lims["xlim"], ylim=lims["ylim"], zlim=lims["zlim"])
    stream = streamtube.Streamtube(vals["x"], vals["y"], vals["z"], vals["ubar"], vals["vbar"], vals["wbar"])
    mask = stream.compute_mask(return_mask=True, R = 0.5 - filterwidth)
    # closed_mask = ndimage.binary_closing(mask, iterations=2, border_value=1)
    # filled_mask = ndimage.binary_fill_holes(closed_mask)
    phase_masks[i] = mask

# %%
vals = sim.slice(budget_terms=["ubar", "vbar", "wbar"], phase=None, xlim=lims["xlim"], ylim=lims["ylim"], zlim=lims["zlim"])
stream = streamtube.Streamtube(vals["x"], vals["y"], vals["z"], vals["ubar"], vals["vbar"], vals["wbar"])
mean_mask = stream.compute_mask(return_mask=True)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x = vals["x"].values
y = vals["y"].values
z = vals["z"].values

mask = np.asarray(mean_mask)      # works for numpy arrays or DataArrays
print(mask.shape, mask.dtype)     # expect (124, 256, 256) in (x, y, z) order

iy = np.argmin(np.abs(y))         # z-plane closest to 0
m_xz = mask[:, iy, :]             # (x, y) slice

# Two-color map for a boolean or 0/1 mask
cmap = ListedColormap(["white", "tab:blue"])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    m_xz.T.astype(int),           # rows = y, cols = x
    origin="lower",
    extent=[x.min(), x.max(), z.min(), z.max()],
    aspect="equal",
    cmap=cmap,
    interpolation="nearest",
    vmin=0, vmax=1,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Mask at y = {y[iy]:.3f}")
cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(["outside", "inside"])
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# 1. Close small gaps (dilate then erode), then fill enclosed holes
closed = ndimage.binary_closing(m_xz, structure=np.ones((3, 3)), iterations=2)
filled = ndimage.binary_fill_holes(closed)

# 2. Compare before and after
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, data, title in zip(axes, [m_xz, filled], ["Original", "Filled"]):
    ax.imshow(data.T, origin="lower",
              extent=[x.min(), x.max(), y.min(), y.max()],
              aspect="equal", cmap="Blues", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
plt.show()

# %%
mask = xr.DataArray(
    np.stack(phase_masks),                 # (phase, x, y, z)
    dims=("phase", "x", "y", "z"),
    coords={"phase": wb.phase, "x": wb.x, "y": wb.y, "z": wb.z},
)

tube_avg = wb.where(mask).mean(("y", "z"))   # dims: (phase, x)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

hub = dict(y=0.0, z=0.0)
D = 1.0
phase_colors = plt.cm.twilight(np.linspace(0, 1, mask.sizes["phase"], endpoint=False))

def _outline(m2d, ax, x, y, **kw):
    """Draw the tube boundary (the 0.5 contour of a boolean mask)."""
    m2d.astype(float).plot.contour(ax=ax, x=x, y=y, levels=[0.5], **kw)


# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
for c, p in zip(phase_colors, mask.phase.values):
    m = mask.sel(phase=p)
    _outline(m.sel(z=hub["z"], method="nearest"), axs[0], "x", "y", colors=[c])
    _outline(m.sel(y=hub["y"], method="nearest"), axs[1], "x", "z", colors=[c])

axs[0].set_title(f"Top view (z = {hub['z']})");  axs[1].set_title(f"Side view (y = {hub['y']})")
for ax in axs:
    ax.set_aspect("equal")
    ax.axvline(0, color="0.6", lw=0.8, ls="--")          # rotor plane
proxies = [Line2D([], [], color=c) for c in phase_colors]
axs[0].legend(proxies, [f"{p:.1f}" for p in mask.phase.values], title="phase",
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)

# %%
x_locs = [0.5, 2, 4, 6, 8, 10]
fig, axs = plt.subplots(1, len(x_locs), figsize=(3 * len(x_locs), 3.2),
                        sharey=True, constrained_layout=True)
for ax, xl in zip(axs, x_locs):
    for c, p in zip(phase_colors, mask.phase.values):
        _outline(mask.sel(phase=p).sel(x=xl, method="nearest"), ax, "y", "z", colors=[c])
    ax.add_patch(Circle((hub["y"], hub["z"]), D / 2, fill=False, ls="--", color="k"))  # rotor disk
    ax.set_aspect("equal"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.set_title(f"x = {float(mask.x.sel(x=xl, method='nearest')):.2f}")

# %%
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",     # LaTeX-like font without needing LaTeX
    "font.size": 14,
    # "text.usetex": True,        # uncomment for real LaTeX rendering (needs a LaTeX install)
})

# (variable in wb, legend label, color): edit the labels to match your term definitions
TERMS = [
    ("adv",      "Advection", "#7f7f7f"),
    ("prod",     "Production",              "#5fd3d0"),
    ("wavewave", "Wave-Wave",      "#5b2a9e"),
    ("storage", "Storage", "#e377c2"),
    ("prss",     "Pressure",                                                       "#4a7fd4"),
    ("sgs",      "SGS",                                                            "#ffb347"),
    ("cor",      "Cor",                                                            "#5aa832"),
    ("rs",       "Reynolds Stresses",                                              "#d93a3a"),
    ("adm",     "ADM",     "#8c564b"),
]

SOLID  = dict(lw=1.8)
DOTTED = dict(lw=2.8, ls=(0, (1, 1.5)), alpha=0.55)


def plot_budget(ds_a, ds_b=None, terms=TERMS, res="residual",
                xlim=None, D=1.0, ax=None):
    """ds_a -> solid lines, ds_b -> dotted lines. Each must be 1D in x."""
    if xlim is not None:
        ds_a = ds_a.sel(x=slice(*xlim))
        ds_b = ds_b.sel(x=slice(*xlim)) if ds_b is not None else None

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    handles, labels = [], []

    for v, label, c in terms:
        if v not in ds_a:
            continue
        (h1,) = ax.plot(ds_a.x / D, ds_a[v], color=c, **SOLID)
        h = [h1]
        if ds_b is not None:
            (h2,) = ax.plot(ds_b.x / D, ds_b[v], color=c, **DOTTED)
            h.append(h2)
        handles.append(tuple(h))
        labels.append(label)

    # Residual: black dash-dot (solid case), gray dashed (dotted case)
    if res is not None and res in ds_a:
        (h1,) = ax.plot(ds_a.x / D, ds_a[res], color="k", ls="-.", lw=1.8)
        h = [h1]
        if ds_b is not None:
            (h2,) = ax.plot(ds_b.x / D, ds_b[res], color="0.3", ls="--", lw=2.5)
            h.append(h2)
        handles.append(tuple(h))
        labels.append("Res")

    ax.set_xlabel(r"$x/D$")
    ax.set_ylim(-0.4, 0.4)
    ax.legend(handles, labels,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
              handlelength=4.5, frameon=False,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              labelspacing=0.9)
    fig.tight_layout()
    return fig, ax


# %%
tube_avg

# %%
fig, ax = plot_budget(tube_avg.sel(phase=0.0),
                      xlim=(2, 6))

# Or one phase vs. the phase average
# fig, ax = plot_budget(tube_avg.sel(phase=0.0), tube_avg.mean("phase"), xlim=(2.5, 15))

# A single case (solid lines only)
# fig, ax = plot_budget(tube_avg.mean("phase"), xlim=(2.5, 15))

fig.savefig("tube_budget.pdf", bbox_inches="tight")

# %%
fig, ax = plot_budget(tube_avg.sel(phase=0.5),
                      xlim=(2, 15))

# Or one phase vs. the phase average
# fig, ax = plot_budget(tube_avg.sel(phase=0.0), tube_avg.mean("phase"), xlim=(2.5, 15))

# A single case (solid lines only)
# fig, ax = plot_budget(tube_avg.mean("phase"), xlim=(2.5, 15))

fig.savefig("tube_budget.pdf", bbox_inches="tight")

# %%
