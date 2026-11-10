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
import numpy as np
import pandas as pd
import os
import lib.analysis_utils as au
import padeopsIO as pio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from UnifiedMomentumModel.Momentum import UnifiedMomentum
from matplotlib.lines import Line2D

# %%
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams['figure.dpi'] = 100


# %%
def get_cv_axis(grid, bounds):
    """Construct CV coordinates, including exact boundaries."""
    lower, upper = bounds
    interior = grid[(grid > lower) & (grid < upper)]
    return np.concatenate(([lower], interior, [upper]))


def get_bracketing_bounds(grid, bounds):
    """Return grid coordinates bracketing the requested bounds."""
    lower, upper = bounds

    i0 = max(np.searchsorted(grid, lower, side="right") - 1, 0)
    i1 = min(np.searchsorted(grid, upper, side="left"), len(grid) - 1)

    return grid[i0], grid[i1]

def get_cv_fields(
    surge_sim,
    phase,
    x_bounds,
    y_bounds,
    z_bounds,
    xturb,
    sgs_keys=("tau11", "tau12", "tau13"),
    storage_key="uStor",
):
    """
    Extract phase-conditioned ground-frame fields on the box at its
    phase-phi position (turbine-relative bounds shifted by xturb).

    Returns turbine-relative coordinates and fields shaped (x, y, z).
    """
    xlim = np.asarray(x_bounds) + xturb

    x_cv_ground = get_cv_axis(surge_sim.x, xlim)
    y_cv = get_cv_axis(surge_sim.y, y_bounds)
    z_cv = get_cv_axis(surge_sim.z, z_bounds)

    x_slice = get_bracketing_bounds(surge_sim.x, xlim)
    y_slice = get_bracketing_bounds(surge_sim.y, y_bounds)
    z_slice = get_bracketing_bounds(surge_sim.z, z_bounds)

    budget_terms = [
        "ubar", "vbar", "wbar",
        "uu", "uv", "uw",
        "pbar", "xAD",
        *sgs_keys,
    ]
    if phase is not None:
        budget_terms.append(storage_key)

    vals = surge_sim.slice(
        budget_terms=budget_terms,
        phase=phase,
        xlim=x_slice,
        ylim=y_slice,
        zlim=z_slice,
    )

    vals = vals.interp(x=x_cv_ground, y=y_cv, z=z_cv, method="linear")
    x_cv = x_cv_ground - xturb

    # Phase-mean velocities
    u = vals["ubar"].values
    v = vals["vbar"].values
    w = vals["wbar"].values

    # Resolved Reynolds stresses <u'u'_j> about the PHASE mean.
    # Passed directly -- no reconstruction of total second moments.
    uu_p = vals["uu"].values
    uv_p = vals["uv"].values
    uw_p = vals["uw"].values

    # Phase-averaged SGS stresses (model output)
    tau11 = vals[sgs_keys[0]].values
    tau12 = vals[sgs_keys[1]].values
    tau13 = vals[sgs_keys[2]].values

    # Eulerian storage <du/dt|_x>
    du_dt = (
        vals[storage_key].values if phase is not None
        else np.zeros_like(u)
    )

    p = vals["pbar"].values
    fx = vals["xAD"].values

    fields = {
        "u": u, "v": v, "w": w,
        "uu_p": uu_p, "uv_p": uv_p, "uw_p": uw_p,
        "p": p,
        "du_dt": du_dt,
        "tau11": tau11, "tau12": tau12, "tau13": tau13,
        "fx": fx,
    }
    return x_cv, y_cv, z_cv, fields


# %%
def integrate_volume(q, x, y, z):
    """Integrate a 3D field over the control volume."""
    return np.trapezoid(
        np.trapezoid(
            np.trapezoid(q, x=z, axis=2),
            x=y, axis=1
        ),
        x=x, axis=0
    )

def integrate_face(q, coord_1, coord_2):
    """Integrate a 2D field over a control-volume face."""
    return np.trapezoid(
        np.trapezoid(q, x=coord_2, axis=1),
        x=coord_1, axis=0
    )

def _signed_face_integrals(qx, qy, qz, x, y, z):
    """
    Face-by-face contributions to  oint_S q_j n_j dS  on a rectangular CV.
    qx on x faces (n = -e_x, +e_x), qy on y faces, qz on z faces.
    Pass None to skip a component.
    """
    out = {}
    if qx is not None:
        out["x_min"] = -integrate_face(qx[0, :, :], y, z)
        out["x_max"] = +integrate_face(qx[-1, :, :], y, z)
    if qy is not None:
        out["y_min"] = -integrate_face(qy[:, 0, :], x, z)
        out["y_max"] = +integrate_face(qy[:, -1, :], x, z)
    if qz is not None:
        out["z_min"] = -integrate_face(qz[:, :, 0], x, y)
        out["z_max"] = +integrate_face(qz[:, :, -1], x, y)
    return out

def _by_direction(faces):
    """Sum min/max faces for each direction."""
    return {
        d: faces.get(f"{d}_min", 0.0) + faces.get(f"{d}_max", 0.0)
        for d in ("x", "y", "z")
    }

def _scale(faces, c):
    return {k: c * q for k, q in faces.items()}


# %%
def mass_conservation_cv(x, y, z, u, v, w, u_t, rho):
    """
    Phase-averaged mass conservation for a moving control volume.

    u, v, w: phase-averaged ground-frame velocities
    u_t:     phase-conditioned turbine velocity
    rho:     constant density

    All fields have dimensions (x, y, z).
    """

    # Relative velocity: U_r = U - U_CV.
    # The control surface moves with velocity (u_t, 0, 0).
    ur = u - u_t

    # Mass flux through each face.
    # Signs correspond to the outward-pointing face normals.
    mass_flux = {
        "x_min": -integrate_face(ur[0, :, :], y, z),
        "x_max": +integrate_face(ur[-1, :, :], y, z),

        "y_min": -integrate_face(v[:, 0, :], x, z),
        "y_max": +integrate_face(v[:, -1, :], x, z),

        "z_min": -integrate_face(w[:, :, 0], x, y),
        "z_max": +integrate_face(w[:, :, -1], x, y),
    }

    # Constant density and fixed turbine-frame volume:
    # d/dt int_V rho dV = 0.
    mass_storage = 0.0

    # Net outward mass flux. Should equal zero.
    net_mass_flux = sum(mass_flux.values())

    return {
        "mass_storage": mass_storage,
        "mass_flux": mass_flux,
        "net_mass_flux": net_mass_flux,
        "residual": mass_storage + net_mass_flux,
    }


# %%
def x_momentum_conservation_cv(
    x, y, z,
    u, v, w,
    uu_p, uv_p, uw_p,
    p,
    du_dt,
    tau11, tau12, tau13,
    u_t, a_t,
    fx=None,
    frame="turbine",
    Ro=None,
    latitude=45.0,
    G_y=0.0,
    fx_other=None,
):
    """
    Phase-averaged x-momentum budget on a rectangular CV following a
    surging turbine. Inviscid LES, PadeOps SGS sign convention.

    Inputs: ground-frame phase averages on the box at its phase-phi
    position, all shaped (x, y, z):

        u, v, w           : <u_i>_g
        uu_p, uv_p, uw_p  : resolved Reynolds stresses <u'_1 u'_j>_g
                            (fluctuations about the PHASE mean)
        p                 : <p>_g (modified pressure)
        du_dt             : <du/dt |_x>_g (Eulerian storage)
        tau11..tau13      : <tau_1j>_g, phase-averaged SGS stress output
        fx                : <f_x>_g, actuator force per unit mass on fluid
        u_t, a_t          : surge velocity / acceleration at phi

    Budget (fictitious force moved to LHS):

      LHS: storage + mean advection [+ frame acceleration]
      RHS: pressure + Reynolds stress + SGS stress + actuator
           + Coriolis + other forcing

    Forms (equivalent when oint U_j n_j dS = 0):
      "ground_eulerian" (A): rho int Udot dV + rho oint U U_j n_j dS
      "ground_moving"   (B): rho d/dt int_{V(t)} U dV
                             + rho oint U (U_j - u_t delta_j1) n_j dS
      "turbine"         (C): rho d/dt int <v_1> dV
                             + rho oint <v_1><v_j> n_j dS + rho a_t V
    """
    turbine = frame == "turbine"
    if frame == "ground_moving":
        return None
    # moving = frame in ("turbine", "ground_moving")

    volume = (x[-1] - x[0]) * (y[-1] - y[0]) * (z[-1] - z[0])

    # ---------------------------------------------------------
    # I. Storage (LHS)
    # ---------------------------------------------------------
    storage_time = integrate_volume(du_dt, x, y, z)

    # Delta Q_x = int_out U dA - int_in U dA  (exact int dU/dx dV)
    dQx = integrate_face(u[-1, :, :], y, z) - integrate_face(u[0, :, :], y, z)
    storage_motion = u_t * dQx #if moving else 0.0

    storage_acceleration = -a_t * volume if turbine else 0.0
    storage = storage_time + storage_motion + storage_acceleration

    # Fictitious force: -rho a_t V on RHS -> +rho a_t V on LHS
    frame_acceleration = -a_t * volume if turbine else 0.0
    frame_acceleration_lhs = -frame_acceleration

    # ---------------------------------------------------------
    # II. Mean advection (LHS) -- phase-mean flux ONLY
    # ---------------------------------------------------------
    if turbine:
        ur = u - u_t                                  # <v_1>
        mean_x, mean_y, mean_z = ur * ur, ur * v, ur * w
    # elif frame == "ground_moving":
    #     mean_x, mean_y, mean_z = u * (u - u_t), u * v, u * w

    advection_faces =  _signed_face_integrals(mean_x, mean_y, mean_z, x, y, z)
    adv_dir = _by_direction(advection_faces)
    advection = sum(advection_faces.values())

    # ---------------------------------------------------------
    # III. Pressure force (RHS): -oint p n_x dS
    # ---------------------------------------------------------
    pressure_faces = _scale(
        _signed_face_integrals(p, None, None, x, y, z), -1.0
    )
    pressure_force = sum(pressure_faces.values())

    # ---------------------------------------------------------
    # IV. Resolved Reynolds stress force (RHS):
    #     -rho oint <u'_1 u'_j> n_j dS   (frame-independent)
    # ---------------------------------------------------------
    reynolds_faces = _scale(
        _signed_face_integrals(uu_p, uv_p, uw_p, x, y, z), -1.0
    )
    rey_dir = _by_direction(reynolds_faces)
    reynolds_force = sum(reynolds_faces.values())

    # ---------------------------------------------------------
    # V. SGS stress force (RHS): -rho oint <tau_1j> n_j dS
    # ---------------------------------------------------------
    sgs_faces = _scale(
        _signed_face_integrals(tau11, tau12, tau13, x, y, z), -1.0
    )
    sgs_dir = _by_direction(sgs_faces)
    sgs_force = sum(sgs_faces.values())

    turb_stress_force = reynolds_force + sgs_force

    # ---------------------------------------------------------
    # VI. Actuator force on fluid (RHS): rho int <f_x> dV  (= -T)
    # ---------------------------------------------------------
    actuator_force = (
        integrate_volume(fx, x, y, z) if fx is not None else 0.0
    )

    # ---------------------------------------------------------
    # VII. Coriolis + geostrophic (RHS), f-plane x-equation
    # ---------------------------------------------------------
    if Ro is not None:
        f_hat = 2.0 * np.sin(np.deg2rad(latitude)) / Ro
        coriolis_force = rho * f_hat * integrate_volume(v - G_y, x, y, z)
    else:
        coriolis_force = 0.0

    other_force = (
        rho * integrate_volume(fx_other, x, y, z)
        if fx_other is not None else 0.0
    )

    # ---------------------------------------------------------
    # VIII. Mass conservation diagnostic
    # ---------------------------------------------------------
    Q_faces = _signed_face_integrals(u, v, w, x, y, z)
    Q_net = sum(Q_faces.values())
    frame_mass_discrepancy = -1.0 * u_t * Q_net   # LHS(C) - LHS(A)

    # ---------------------------------------------------------
    # IX. Assemble
    # ---------------------------------------------------------
    lhs_terms = {
        "storage_time": storage_time,
        "storage_motion": storage_motion,
        "storage_acceleration": storage_acceleration,
        "advection_x": adv_dir["x"],
        "advection_y": adv_dir["y"],
        "advection_z": adv_dir["z"],
        "frame_acceleration": frame_acceleration_lhs,
    }
    rhs_terms = {
        "pressure_x_min": pressure_faces["x_min"],
        "pressure_x_max": pressure_faces["x_max"],
        "reynolds_x": rey_dir["x"],
        "reynolds_y": rey_dir["y"],
        "reynolds_z": rey_dir["z"],
        "sgs_x": sgs_dir["x"],
        "sgs_y": sgs_dir["y"],
        "sgs_z": sgs_dir["z"],
        "actuator_force": actuator_force,
        "coriolis_force": coriolis_force,
        "other_force": other_force,
    }

    lhs = sum(lhs_terms.values())
    rhs = sum(rhs_terms.values())

    # Thrust: force of fluid on turbine, +downstream
    thrust_direct = -actuator_force
    thrust_cv = (
        pressure_force + reynolds_force + sgs_force
        + coriolis_force + other_force - lhs
    )

    return {
        "frame": frame,
        "volume": volume,

        "storage_time": storage_time,
        "storage_motion": storage_motion,
        "storage_acceleration": storage_acceleration,
        "storage": storage,

        "advection_faces": advection_faces,
        "advection_x": adv_dir["x"],
        "advection_y": adv_dir["y"],
        "advection_z": adv_dir["z"],
        "advection": advection,

        "pressure_faces": pressure_faces,
        "pressure_force": pressure_force,

        "reynolds_faces": reynolds_faces,
        "reynolds_x": rey_dir["x"],
        "reynolds_y": rey_dir["y"],
        "reynolds_z": rey_dir["z"],
        "reynolds_force": reynolds_force,

        "sgs_faces": sgs_faces,
        "sgs_x": sgs_dir["x"],
        "sgs_y": sgs_dir["y"],
        "sgs_z": sgs_dir["z"],
        "sgs_force": sgs_force,

        "turb_stress_force": turb_stress_force,

        "actuator_force": actuator_force,
        "coriolis_force": coriolis_force,
        "other_force": other_force,

        "frame_acceleration": frame_acceleration,
        "frame_acceleration_lhs": frame_acceleration_lhs,

        "Q_faces": Q_faces,
        "Q_net": Q_net,
        "frame_mass_discrepancy": frame_mass_discrepancy,

        "lhs_terms": lhs_terms,
        "rhs_terms": rhs_terms,
        "lhs": lhs,
        "rhs": rhs,
        "residual": lhs - rhs,

        "thrust_cv": thrust_cv,
        "thrust_direct": thrust_direct,
    }


# %%
def plot_mass_budget(mass, phase = None, mass_scale=None):
    """Plot normalized mass flux through the CV faces."""

    scale = 1.0 if mass_scale is None else mass_scale

    faces = [
        "x_min", "x_max",
        "y_min", "y_max",
        "z_min", "z_max",
    ]

    labels = ["x−", "x+", "y−", "y+", "z−", "z+", "Residual"]

    values = np.array([
        *[mass["mass_flux"][face] for face in faces],
        mass["residual"],
    ]) / scale

    colors = (
        ["tab:blue"] * 2
        + ["tab:orange"] * 2
        + ["tab:green"] * 2
        + ["black"]
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)

    if mass_scale is None:
        ax.set_ylabel("Mass flux")
    else:
        ax.set_ylabel(r"Normalized mass flux, $\dot{m}_i/\dot{m}_{ref}$")

    title_text =  f"Mass conservation budget\n" + f"Residual = {mass['residual']:.3e}"
    if phase is not None:
        title_text += f", Phase = {phase}"

    ax.set_title(title_text)
    ax.grid(axis="y", alpha=0.3)

    legend = [
        Patch(color="tab:blue", label="Streamwise"),
        Patch(color="tab:orange", label="Lateral"),
        Patch(color="tab:green", label="Vertical"),
        Patch(color="black", label="Residual"),
    ]

    ax.legend(handles=legend)
    fig.tight_layout()

    return fig, ax


# %%
# key -> (label, color, side)
TERM_META = {
    # ---------------- LHS ----------------
    "storage":        ("Storage",                         "tab:blue",   "LHS"),
    "storage_time":   ("Eulerian storage",                "tab:blue",   "LHS"),
    "storage_motion": ("Coordinate motion",               "tab:cyan",   "LHS"),
    "storage_acc":    ("Storage acceleration\n$-\\rho a_T V$", "tab:red", "LHS"),
    "frame_acc":      ("Fictitious force\n$+\\rho a_T V$",     "salmon",  "LHS"),
    "adv_x":         ("Mean adv. x",    "tab:orange", "LHS"),
    "adv_y":         ("Mean adv. y",    "tab:orange", "LHS"),
    "adv_z":         ("Mean adv. z",    "tab:orange", "LHS"),
    # ---------------- RHS ----------------
    "p_xmin":         ("Pressure x−",     "tab:green",  "RHS"),
    "p_xmax":         ("Pressure x+",     "tab:green",  "RHS"),
    "rey_x":          ("Reynolds x",      "tab:brown",  "RHS"),
    "rey_y":          ("Reynolds y",      "tab:brown",  "RHS"),
    "rey_z":          ("Reynolds z",      "tab:brown",  "RHS"),
    "sgs_x":          ("SGS x",           "tab:pink",   "RHS"),
    "sgs_y":          ("SGS y",           "tab:pink",   "RHS"),
    "sgs_z":          ("SGS z",           "tab:pink",   "RHS"),
    "actuator":       ("Actuator force",  "tab:purple", "RHS"),
    "coriolis":       ("Coriolis",        "tab:gray",   "RHS"),
    "other":          ("Other forcing",   "tab:olive",  "RHS"),
    # -------------- Closure --------------
    "residual":       ("Residual",        "black",      "Closure"),
    # ---- Aggregates (line plots / reference comparisons) ----
    "adv_total":     ("Mean adv. total", "tab:orange", "LHS"),
    "p_net":          ("Pressure net",     "tab:green",  "RHS"),
    "rey_total":      ("Reynolds total",   "tab:brown",  "RHS"),
    "sgs_total":      ("SGS total",        "tab:pink",   "RHS"),
    "turb_total":     ("Reynolds + SGS",   "tab:red",    "RHS"),
    "thrust_cv":      ("Thrust (CV)",      "black",      "Closure"),
    "thrust_direct":  ("Thrust (actuator)", "tab:purple", "Closure"),
}

# Storage label depends on which form was used
STORAGE_LABELS = {
    "ground_eulerian": "Eulerian storage\n$\\rho\\int\\langle\\partial_t u\\rangle dV$",
    "ground_moving":   "Moving-box storage\n$\\rho\\, d/dt\\int_{V(t)} U dV$",
    "turbine":         "Eulerian storage +\ncoordinate motion",
}

SIDE_ORDER = ("LHS", "RHS", "Closure")

_STORAGE_KEYS = {
    False: ["storage"],
    True:  ["storage_time", "storage_motion"],
}
_STRESS_KEYS = {
    "direction": ["rey_x", "rey_y", "rey_z", "sgs_x", "sgs_y", "sgs_z"],
    "total":     ["rey_total", "sgs_total"],
    "combined":  ["turb_total"],
}


def bar_keys(split_storage=False, stress_detail="direction"):
    return (
        _STORAGE_KEYS[split_storage]
        + ["storage_acc", "frame_acc", "adv_x", "adv_y", "adv_z",
           "p_xmin", "p_xmax"]
        + _STRESS_KEYS[stress_detail]
        + ["actuator", "coriolis", "other", "residual"]
    )


LINE_KEYS = [
    "storage", "adv_total", "p_net",
    "rey_total", "sgs_total", "actuator", "residual",
]

# Terms hidden in bar plots when exactly zero
_DROP_IF_ZERO = {"storage_acc", "frame_acc", "storage_motion",
                 "coriolis", "other"}


def budget_values(m, scale=1.0):
    """Normalized budget terms from x_momentum_conservation_cv output."""
    vals = {
        "storage":        m["storage_time"] + m["storage_motion"],
        "storage_time":   m["storage_time"],
        "storage_motion": m["storage_motion"],
        "storage_acc":    m["storage_acceleration"],
        "frame_acc":      m["frame_acceleration_lhs"],
        "adv_x":         m["advection_x"],
        "adv_y":         m["advection_y"],
        "adv_z":         m["advection_z"],
        "adv_total":     m["advection"],
        "p_xmin":         m["pressure_faces"]["x_min"],
        "p_xmax":         m["pressure_faces"]["x_max"],
        "p_net":          m["pressure_force"],
        "rey_x":          m["reynolds_x"],
        "rey_y":          m["reynolds_y"],
        "rey_z":          m["reynolds_z"],
        "rey_total":      m["reynolds_force"],
        "sgs_x":          m["sgs_x"],
        "sgs_y":          m["sgs_y"],
        "sgs_z":          m["sgs_z"],
        "sgs_total":      m["sgs_force"],
        "turb_total":     m["reynolds_force"] + m["sgs_force"],
        "actuator":       m["actuator_force"],
        "coriolis":       m.get("coriolis_force", 0.0),
        "other":          m.get("other_force", 0.0),
        "residual":       m["residual"],
        "thrust_cv":      m.get("thrust_cv", np.nan),
        "thrust_direct":  m.get("thrust_direct", np.nan),
    }
    return {k: float(v) / scale for k, v in vals.items()}


def _label(key, frame):
    if key == "storage":
        return STORAGE_LABELS.get(frame, TERM_META[key][0])
    return TERM_META[key][0]


# %%
def plot_momentum_budget(
    momentum,
    phase=None,
    force_scale=None,
    show_acceleration_terms=True,
    split_storage=False,
    stress_detail="direction",     # "direction" | "total" | "combined"
    hide_zero_terms=True,
    references=None,
    ylim=(-2, 1),
):
    """
    X-momentum budget bar plot, LHS/RHS separated.

    references : list of dict, optional
        {"values": {key: normalized value}, "label": str,
         "color": str, "linestyle": str}
        Drawn as a horizontal tick on every bar whose key is present.
    """
    scale = 1.0 if force_scale is None else force_scale
    vals = budget_values(momentum, scale)
    frame = momentum["frame"]

    keys = bar_keys(split_storage, stress_detail)
    if not (show_acceleration_terms and frame == "turbine"):
        keys = [k for k in keys if k not in ("storage_acc", "frame_acc")]
    if hide_zero_terms:
        keys = [k for k in keys
                if not (k in _DROP_IF_ZERO and vals[k] == 0.0)]

    # Enforce contiguous LHS | RHS | Closure ordering
    keys.sort(key=lambda k: SIDE_ORDER.index(TERM_META[k][2]))

    labels = [_label(k, frame) for k in keys]
    colors = [TERM_META[k][1] for k in keys]
    sides  = [TERM_META[k][2] for k in keys]
    values = np.array([vals[k] for k in keys])
    positions = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(max(12, 0.7 * len(keys)), 5))
    ax.bar(positions, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)

    # Section dividers and headers
    trans = ax.get_xaxis_transform()
    start = 0
    for i, side in enumerate(SIDE_ORDER):
        n = sides.count(side)
        if n == 0:
            continue
        end = start + n
        ax.text((start + end - 1) / 2, 1.02, side, ha="center",
                transform=trans, weight="bold")
        if end < len(keys):
            ax.axvline(end - 0.5, color="black" if side == "LHS" else "gray",
                       linestyle="--" if side == "LHS" else ":",
                       linewidth=1.2 if side == "LHS" else 1.0)
        start = end

    # Reference ticks
    handles = []
    for ref in references or []:
        c  = ref.get("color", "k")
        ls = ref.get("linestyle", "--")
        for pos, k in zip(positions, keys):
            if k in ref["values"]:
                ax.hlines(ref["values"][k], pos - 0.42, pos + 0.42,
                          colors=c, linestyles=ls, linewidth=2.2, zorder=4)
        handles.append(Line2D([0], [0], color=c, linestyle=ls,
                              linewidth=2.2, label=ref["label"]))
    if handles:
        ax.legend(handles=handles, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), frameon=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("X-momentum contribution" if force_scale is None
                  else r"Normalized contribution, $C_i/F_{\mathrm{ref}}$")
    if ylim is not None:
        ax.set_ylim(*ylim)

    title = f"X-momentum budget ({frame} frame)"
    if phase is not None:
        title += f", Phase = {phase}"
    title += (f"\nResidual = {vals['residual']:.3e}"
              f"   |   T_CV = {vals['thrust_cv']:.3f},"
              f"  T_act = {vals['thrust_direct']:.3f}")
    ax.set_title(title, pad=28)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_budget_vs_phase(
    momenta, phases, keys=LINE_KEYS, force_scale=None, ax=None,
):
    """Budget terms vs. surge phase. `momenta` is a list of outputs."""
    scale = 1.0 if force_scale is None else force_scale
    data = {k: np.array([budget_values(m, scale)[k] for m in momenta])
            for k in keys}
    frame = momenta[0]["frame"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    for k in keys:
        ls = "--" if TERM_META[k][2] == "RHS" else "-"
        if k == "residual":
            ls = ":"
        ax.plot(phases, data[k], ls, color=TERM_META[k][1],
                label=_label(k, frame).replace("\n", " "), marker="o", ms=3)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"Surge phase $\phi$")
    ax.set_ylabel("X-momentum contribution" if force_scale is None
                  else r"$C_i/F_{\mathrm{ref}}$")
    ax.set_title(f"X-momentum budget vs. phase ({frame} frame)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax, data


# %%
def umm_cv_reference(sol, Ct_prime, rho=1.0, UInf=1.0, D=1.0):
    """
    UMM streamtube x-momentum terms, normalized by 0.5*rho*Ct'*A*UInf^2
    (the same normalization as the bars).

    Returns (values dict, x0). adv_total and p_net are defined at x = x0.
    """
    an = float(np.squeeze(sol.an))
    u4 = float(np.squeeze(sol.u4))
    dp = float(np.squeeze(sol.dp))        # p4 - p1
    x0 = float(np.squeeze(sol.x0)) * D    # x0 is returned as x0/D

    A_  = np.pi * D**2 / 4
    ud  = (1 - an) * UInf
    mdot = rho * A_ * ud
    A4  = A_ * ud / u4
    F_ref  = 0.5 * rho * Ct_prime * A_ * UInf**2
    thrust = 0.5 * rho * Ct_prime * A_ * ud**2

    values = {
        "actuator":   -thrust / F_ref,              # = -(1 - an)^2
        "adv_total": mdot * (u4 - UInf) / F_ref,
        "p_net":      -dp * A4 / F_ref,             # (p1 - p4) A4
    }

    closure = values["adv_total"] - values["p_net"] - values["actuator"]
    print(f"UMM: an={an:.4f}, u4={u4:.4f}, dp={dp:.4f}, x0/D={x0/D:.3f}")
    print(f"UMM streamtube momentum closure = {closure:.2e} (should be ~0)")
    return values, x0


# %%
Ct_prime = 2
f = 1.0
Av = 0.3
D = 1
R = D / 2
A = np.pi * R**2
rho = 1
UInf = 1

# %%
Lx, Ly, Lz = 25, 8.5, 8.5
n = 256
dx, dy, dz = Lx/n, Ly/n, Lz/n
filterwidth = np.sqrt(dx**2 + dy**2 + dz**2)
M_inv = np.array(1 + (Ct_prime * filterwidth) / (2 * np.sqrt(3 * np.pi)))

# %%
surge_run_id = 1
steady_run_id = 2

# %%
sim_folder =  os.path.join(au.DATA_PATH, "F_0030_SU_Files/")
surge_run_folder = au.get_run_folder(sim_folder, surge_run_id)
surge_sim = pio.BudgetIO(surge_run_folder, padeops=True, runid=surge_run_id, normalize_origin="turbine")
surge_data = pd.DataFrame(au.load_simulation_timeseries(surge_run_folder, "000" + str(surge_run_id), f, spinup_time=75))

# %%
steady_run_folder = au.get_run_folder(sim_folder, steady_run_id)
steady_sim = pio.BudgetIO(steady_run_folder, padeops=True, runid=steady_run_id, normalize_origin="turbine")

# %%
surge_data["ATurb"] = - (2*np.pi*f)**2 * surge_data["DeltaX"]
motion = surge_data.groupby("PhaseBin", sort=True).agg(
    xturb=("DeltaX", "mean"),
    uturb=("UTurb", "mean"),
    aturb=("ATurb", "mean"),
    udisk=("UDisk", "mean"),
) 
phases = motion.index.to_numpy(dtype=float)

# %%
x_bounds = (-4.5, 2.5)
y_bounds = (-2.0, 2.0)
z_bounds = (-2.0, 2.0)

# %%
phase = 0.5

xturb = motion.loc[phase, "xturb"]
uturb = motion.loc[phase, "uturb"]
aturb = motion.loc[phase, "aturb"]


x_cv, y_cv, z_cv, fields = get_cv_fields(
    surge_sim,
    phase,
    x_bounds,
    y_bounds,
    z_bounds,
    xturb,
)


momentum = x_momentum_conservation_cv(
    x_cv, y_cv, z_cv,
    **fields,
    u_t=uturb,
    a_t=aturb,
    frame="turbine",
)

# %%
steady_x_cv, steady_y_cv, steady_z_cv, steady_fields = get_cv_fields(
    steady_sim,
    None,
    x_bounds,
    y_bounds,
    z_bounds,
    0.0,
)

steady_mass = mass_conservation_cv(
    steady_x_cv, steady_y_cv, steady_z_cv,
    steady_fields["u"],
    steady_fields["v"],
    steady_fields["w"],
    u_t=0.0,
    rho=rho,
)

steady_momentum = x_momentum_conservation_cv(
    steady_x_cv, steady_y_cv, steady_z_cv,
    **steady_fields,
    u_t=0.0,
    a_t=0.0,
    frame="turbine",
    )

# %%
model = UnifiedMomentum()
umm_sol = model(Ct_prime, yaw=0.0, tilt=0.0)
ss_an = umm_sol.an
ss_CT = umm_sol.Ct
ss_u4 = umm_sol.u4

# %%
umm_ref, umm_x0 = umm_cv_reference(umm_sol, Ct_prime, rho=rho, UInf=UInf, D=D)

steady_scale = 0.5 * Ct_prime * rho * A * UInf**2
steady_ref = budget_values(steady_momentum, steady_scale)

REFS = [
    {"values": steady_ref, "label": "Steady LES", "color": "0.2", "linestyle": "--"},
    {"values": umm_ref,    "label": "UMM",        "color": "crimson", "linestyle": ":"},
]

# %%
-0.5 * rho * Ct_prime * A * (1 - ss_an)**2 * UInf**2

# %%
steady_momentum["actuator_force"]

# %%
fig, ax = plot_momentum_budget(
    steady_momentum, phase=None, show_acceleration_terms=False,
    force_scale=steady_scale, references=[REFS[1]],
)

for phase in phases:

    xturb, uturb, aturb = motion.loc[phase, ["xturb", "uturb", "aturb"]]

    x_cv, y_cv, z_cv, fields = get_cv_fields(
        surge_sim, phase, x_bounds, y_bounds, z_bounds, xturb)
    
    momentum = x_momentum_conservation_cv(
        x_cv, y_cv, z_cv, **fields,
        u_t=uturb, a_t=aturb, frame="turbine")

    fig, ax = plot_momentum_budget(
        momentum, phase=phase, show_acceleration_terms=False,
        force_scale=0.5 * Ct_prime * rho * A * (UInf - uturb)**2,
        references=REFS,
    )


# %%
def run_cv_budget(sim, phase, x_bounds, y_bounds, z_bounds, motion=None):
    """Momentum budget + normalization for one phase (None = steady)."""
    if phase is None:
        xturb = uturb = aturb = 0.0
    else:
        xturb, uturb, aturb = motion.loc[phase, ["xturb", "uturb", "aturb"]]

    x_cv, y_cv, z_cv, fields = get_cv_fields(
        sim, phase, x_bounds, y_bounds, z_bounds, xturb)
    momentum = x_momentum_conservation_cv(
        x_cv, y_cv, z_cv, **fields,
        u_t=uturb, a_t=aturb, frame="turbine")
    scale = 0.5 * Ct_prime * rho * A * (UInf - uturb)**2
    return momentum, scale


def sweep_xmax(sim, phase_list, xmax_values, x_min, y_bounds, z_bounds,
               motion=None):
    """Tidy DataFrame: one row per (phase, xmax), columns = normalized terms."""
    rows = []
    for phase in phase_list:
        for xmax in xmax_values:
            momentum, scale = run_cv_budget(
                sim, phase, (x_min, xmax), y_bounds, z_bounds, motion)
            rows.append({
                "phase": np.nan if phase is None else float(phase),
                "xmax": float(xmax),
                **budget_values(momentum, scale),
            })
    return pd.DataFrame(rows)


xmax_values = np.arange(1.0, 8.01, 0.5)   # user-defined; must stay inside the domain
x_min = x_bounds[0]

surge_cache, steady_cache = "cv_sweep_surge.csv", "cv_sweep_steady.csv"

if os.path.exists(surge_cache):
    surge_df = pd.read_csv(surge_cache)
else:
    surge_df = sweep_xmax(surge_sim, phases, xmax_values, x_min,
                          y_bounds, z_bounds, motion)
    surge_df.to_csv(surge_cache, index=False)

if os.path.exists(steady_cache):
    steady_df = pd.read_csv(steady_cache)
else:
    steady_df = sweep_xmax(steady_sim, [None], xmax_values, x_min,
                           y_bounds, z_bounds)
    steady_df.to_csv(steady_cache, index=False)

# %%
surge_df


# %%
def _term_panels(keys, ncols=5):
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    return fig, axes[:len(keys)]


def _finish_panels(fig, axes, keys, xlabel, title):
    for ax, k in zip(axes, keys):
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(TERM_META[k][0].replace("\n", " "), fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[-min(5, len(axes)):]:
        ax.set_xlabel(xlabel)

    # Shared, de-duplicated legend outside the panel group
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.suptitle(title)
    fig.supylabel(r"Normalized contribution, $C_i/F_{\mathrm{ref}}$")
    fig.tight_layout()
    fig.legend(seen.values(), seen.keys(), loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    return fig


# %%
TERM_META


# %%
def plot_budget_vs_xmax(surge_df, phase, steady_df=None, umm_ref=None,
                        umm_x0=None, keys=LINE_KEYS):
    d = surge_df[np.isclose(surge_df["phase"], phase)].sort_values("xmax")
    fig, axes = _term_panels(keys)

    for ax, k in zip(axes, keys):
        color = TERM_META[k][1]
        ax.plot(d["xmax"], d[k], "-o", color=color, ms=4, label="Surge LES")

        if steady_df is not None:
            s = steady_df.sort_values("xmax")
            ax.plot(s["xmax"], s[k], "--", color="0.3", label="Steady LES")

        if umm_ref is not None and k in umm_ref:
            if k == "actuator":
                ax.axhline(umm_ref[k], color="crimson", ls=":", lw=1.8,
                           label="UMM")
        #     elif umm_x0 is not None:
        #         ax.plot(umm_x0, umm_ref[k], "*", color="crimson", ms=13,
        #                 label=r"UMM (at $x_0$)")

        # if umm_x0 is not None:
        #     ax.axvline(umm_x0, color="crimson", lw=0.7, alpha=0.35)

    return _finish_panels(fig, axes, keys, r"$x_{max}/D$",
                          f"X-momentum budget vs. $x_{{max}}$, phase = {phase}")


for phase in phases:
    plot_budget_vs_xmax(surge_df, phase, steady_df, umm_ref, umm_x0)


# %%
def plot_budget_vs_phase(surge_df, xmax, steady_df=None, umm_ref=None,
                         keys=LINE_KEYS):
    d = surge_df[np.isclose(surge_df["xmax"], xmax)].sort_values("phase")
    fig, axes = _term_panels(keys)

    s = None
    if steady_df is not None:
        s = steady_df[np.isclose(steady_df["xmax"], xmax)]

    for ax, k in zip(axes, keys):
        ax.plot(d["phase"], d[k], "-o", color=TERM_META[k][1], ms=4,
                label="Surge LES")
        if s is not None and not s.empty:
            ax.axhline(s[k].iloc[0], color="0.3", ls="--", label="Steady LES")
        if umm_ref is not None and k in umm_ref and k =="actuator":
            lab = "UMM" if k == "actuator" else r"UMM (streamtube at $x_0$)"
            ax.axhline(umm_ref[k], color="crimson", ls=":", lw=1.8, label=lab)

    return _finish_panels(fig, axes, keys, "Phase",
                          f"X-momentum budget vs. phase, $x_{{max}}/D$ = {xmax}")


# %%
xmax_user = 1.5   # <-- choose

def _get_slice(df, sim, phase_list, motion=None):
    sub = df[np.isclose(df["xmax"], xmax_user)]
    if sub.empty:   # not in the sweep -> compute it
        sub = sweep_xmax(sim, phase_list, [xmax_user], x_min,
                         y_bounds, z_bounds, motion)
    return sub

surge_at_x  = _get_slice(surge_df, surge_sim, phases, motion)
steady_at_x = _get_slice(steady_df, steady_sim, [None])

plot_budget_vs_phase(surge_at_x, xmax_user, steady_at_x, umm_ref)


# %%
def to_uinf_norm(df, motion):
    """Convert U_rel^2-normalized sweep rows to U_inf^2 normalization."""
    out = df.copy()
    out["uturb"] = np.interp(out["phase"], motion.index, motion["uturb"])
    fac = ((UInf - out["uturb"]) / UInf) ** 2
    keys = [k for k in TERM_META if k in out.columns]
    out[keys] = out[keys].mul(fac, axis=0)
    out["qs_fac"] = fac
    return out


def plot_budget_vs_phase_limits(surge_df, xmax, steady_df, motion,
                                umm_ref=None, umm_x0=None, keys=LINE_KEYS):
    d = to_uinf_norm(surge_df[np.isclose(surge_df["xmax"], xmax)], motion)
    d = d.sort_values("phase")
    s = steady_df[np.isclose(steady_df["xmax"], xmax)].iloc[0]
    near_x0 = umm_x0 is not None and abs(xmax - umm_x0) < 0.5

    fig, axes = _term_panels(keys)
    for ax, k in zip(axes, keys):
        ax.plot(d["phase"], d[k], "-o", color=TERM_META[k][1], ms=4,
                label="Surge LES")
        ax.axhline(s[k], color="0.3", ls="--", label="Steady LES")
        ax.plot(d["phase"], s[k] * d["qs_fac"], ":", color="0.3", lw=2,
                label=r"Quasi-Steady $LES \times(U_rel/U_\infty)^2$ ")
        if umm_ref is not None and k in umm_ref and (k == "actuator" or near_x0):
            ax.plot(d["phase"], umm_ref[k] * d["qs_fac"], ":", color="crimson",
                    lw=1.8, label="UMM Turbine (quasi-steady)")

    return _finish_panels(
        fig, axes, keys, "Phase",
        rf"Budget vs. phase, $x_{{max}}/D$ = {xmax}, normalized by "
        r"$\frac{1}{2}\rho C_T' A U_\infty^2$")


plot_budget_vs_phase_limits(surge_at_x, xmax_user, steady_at_x, motion,
                            umm_ref, umm_x0)

# %%
surge_df

# %%
F_inf = 0.5 * Ct_prime * rho * A * UInf**2
omega = 2 * np.pi * f


def deficit_terms(sim, phase, xmax, motion=None):
    """
    Deficit-form x-momentum budget (turbine frame), normalized by
    0.5*rho*Ct'*A*UInf^2.

    LHS : storage + flux_in + flux_out + flux_lat + leak
    RHS : p_net + rey + sgs + actuator (+ coriolis + other)
    """
    if phase is None:
        xt = ut = at = 0.0
    else:
        xt, ut, at = motion.loc[phase, ["xturb", "uturb", "aturb"]]

    x, y, z, fl = get_cv_fields(sim, phase, (x_min, xmax), y_bounds, z_bounds, xt)
    m = x_momentum_conservation_cv(
        x, y, z, **fl, u_t=ut, a_t=at, frame="turbine",
    )
    u, v, w = fl["u"], fl["v"], fl["w"]

    du   = u - UInf          # deficit, = <v1> - U_rel
    ur   = u - ut            # <v1>
    Urel = UInf - ut

    def x_faces(q):
        # signed outward contributions of the inlet and outlet
        return -integrate_face(q[0], y, z), integrate_face(q[-1], y, z)

    def lat_faces(qy, qz):
        return (-integrate_face(qy[:, 0], x, z) + integrate_face(qy[:, -1], x, z)
                - integrate_face(qz[:, :, 0], x, y) + integrate_face(qz[:, :, -1], x, y))

    # ---- Deficit flux of the phase-mean relative flow (LHS) ----
    f_in, f_out = x_faces(du * ur)
    f_lat = lat_faces(du * v, du * w)

    # ---- Leak: U_rel * (net volume outflow). Zero if mass is conserved ----
    leak = rho * Urel * m["Q_net"]

    terms = {
        # LHS
        "storage":  m["storage_time"] + m["storage_motion"],   # d/dt int(du) dV
        "flux_in":  f_in,
        "flux_out": f_out,
        "flux_lat": f_lat,
        "leak":     leak,
        # RHS
        "p_net":    m["pressure_force"],       # independent of pressure reference
        "rey":      m["reynolds_force"],       # -rho oint <u'u_j'> n_j dS
        "sgs":      m["sgs_force"],            # -rho oint <tau_1j>  n_j dS
        "actuator": m["actuator_force"],
        "coriolis": m["coriolis_force"],
        "other":    m["other_force"],
        # Closure and checks
        "residual":   m["residual"],
        "adv_check": m["advection"] - (f_in + f_out + f_lat + leak),
        "acc_check":  m["storage_acceleration"] + m["frame_acceleration_lhs"],
        # Ground-frame wake momentum deficit flux at x+ (for the lag test)
        "wake_def_out": rho * integrate_face(u[-1] * (UInf - u[-1]), y, z),
    }
    return {
        "phase": np.nan if phase is None else float(phase),
        "xmax": float(xmax),
        "u_t": ut, "a_t": at,
        **{k: val / F_inf for k, val in terms.items()},
    }


xmax_fine = np.round(np.arange(1.0, 6.01, 0.2), 2)
deficit_cache = (f"cv_deficit_v2_xmin{x_min}_y{y_bounds[1]}_z{z_bounds[1]}"
                 f"_x{xmax_fine[0]}-{xmax_fine[-1]}-{len(xmax_fine)}.csv")

if os.path.exists(deficit_cache):
    ddf = pd.read_csv(deficit_cache)
else:
    rows = []
    for xm in xmax_fine:
        rows.append(deficit_terms(steady_sim, None, xm))
        rows += [deficit_terms(surge_sim, ph, xm, motion) for ph in phases]
    ddf = pd.DataFrame(rows)
    ddf.to_csv(deficit_cache, index=False)

print("max |adv_check| =", ddf["adv_check"].abs().max())   # ~1e-12
print("max |acc_check|  =", ddf["acc_check"].abs().max())    # ~1e-12

# %%
ddf

# %%
CONTRIB = {
    "Pressure (net)":           "tab:green",
    "Storage":                  "tab:blue",
    "Inlet flux x− (upstream)": "tab:cyan",
    "Outlet flux x+ (wake)":    "tab:orange",
    "Lateral deficit flux":     "gold",
    "Reynolds stress":          "tab:brown",
    "SGS stress":               "tab:pink",
    "Leak":                     "0.3",
    "Coriolis + other":         "tab:olive",
    "Residual":                 "0.7",
}


def thrust_deviation(ddf, xmax):
    """Contributions to dT = T_LES - T_QS (thrust on turbine, + downstream)."""
    sel = np.isclose(ddf["xmax"], xmax)
    s = ddf[sel & ddf["phase"].isna()].iloc[0]
    d = ddf[sel & ddf["phase"].notna()].sort_values("phase").reset_index(drop=True)
    fac = (1 - d["u_t"] / UInf) ** 2      # quasi-steady scaling with relative velocity

    def dev(k, qs=True):
        return d[k] - (s[k] * fac if qs else 0.0)

    out = d[["phase", "u_t", "a_t"]].copy()
    out["dT"] = -dev("actuator")

    # RHS forces enter T with +
    out["Pressure (net)"]   = dev("p_net")
    out["Reynolds stress"]  = dev("rey")
    out["SGS stress"]       = dev("sgs")
    out["Coriolis + other"] = dev("coriolis") + dev("other")

    # LHS terms enter T with -
    out["Storage"] = -dev("storage", qs=False)       # QS storage = 0
    out["Inlet flux x− (upstream)"] = -dev("flux_in")
    out["Outlet flux x+ (wake)"]    = -dev("flux_out")
    out["Lateral deficit flux"]     = -dev("flux_lat")
    out["Leak"] = -dev("leak")

    # Budget residual, with the two bookkeeping checks removed
    out["Residual"] = dev("residual") - dev("adv_check") - dev("acc_check")

    out["check"] = out[list(CONTRIB)].sum(axis=1) - out["dT"]   # ~0
    return out


def plot_thrust_deviation(dev, xmax, hide_zero=True):
    ph = dev["phase"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pos, neg = np.zeros_like(ph), np.zeros_like(ph)

    for lab, c in CONTRIB.items():
        yv = dev[lab].to_numpy()
        if hide_zero and np.nanmax(np.abs(yv)) < 1e-10:
            continue                      # e.g. Coriolis + other at Ro = 1e10
        ax.bar(ph, yv, 0.07, bottom=np.where(yv >= 0, pos, neg),
               color=c, label=lab)
        pos += np.clip(yv, 0, None)
        neg += np.clip(yv, None, 0)

    ax.plot(ph, dev["dT"], "k-o", lw=2, label=r"$\Delta T = T_{LES} - T_{QS}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Phase")
    ax.set_ylabel(r"Contribution to $\Delta T\,/\,\frac{1}{2}\rho C_T' A U_\infty^2$")
    ax.set_title(f"Where the quasi-steady thrust error comes from, $x_{{max}}/D$ = {xmax}")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(ph, dev["u_t"], ":", color="0.4", label=r"$u_t/U_\infty$ (right)")
    ax2.plot(ph, dev["a_t"] / omega, "--", color="0.6",
             label=r"$a_t/(\omega U_\infty)$ (right)")
    ax2.set_ylabel("Motion")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h1 + h2, l1 + l2, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    return fig


xmax_user = 2.0
dev_user = thrust_deviation(ddf, xmax_user)
print("closure check:", dev_user["check"].abs().max())
plot_thrust_deviation(dev_user, xmax_user)

# %%
# def damping_inertia(dev, cols=("dT", *CONTRIB)):
#     X = np.column_stack([np.ones(len(dev)), dev["u_t"], dev["a_t"]])
#     rows = {}
#     for c in cols:
#         yv = dev[c].to_numpy() * F_inf                 # dimensional force on turbine
#         b = np.linalg.lstsq(X, yv, rcond=None)[0]
#         ss = np.sum((yv - yv.mean()) ** 2)
#         rows["Total ΔT" if c == "dT" else c] = {
#             "mean": b[0] / F_inf,
#             "C": -b[1] / (rho * A * UInf),             # units of rho*A*U_inf
#             "M": -b[2] / (rho * R**3),                 # units of rho*R^3
#             "R2": 1 - np.sum((yv - X @ b) ** 2) / ss if ss > 0 else np.nan,
#         }
#     return pd.DataFrame(rows).T


# fit_user = damping_inertia(dev_user)
# print(fit_user.round(3))

# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# colors = ["k"] + list(CONTRIB.values())
# for ax, key, lab in [(axes[0], "C", r"Damping $C/(\rho A U_\infty)$"),
#                      (axes[1], "M", r"Added mass $M/(\rho R^3)$")]:
#     ax.bar(range(len(fit_user)), fit_user[key], color=colors)
#     ax.set_xticks(range(len(fit_user)))
#     ax.set_xticklabels(fit_user.index, rotation=40, ha="right")
#     ax.set_title(lab)
#     ax.axhline(0, color="k", lw=0.8)
#     ax.grid(axis="y", alpha=0.3)
# axes[1].axhline(8 / 3, color="crimson", ls="--", label=r"Solid disk $\frac{8}{3}\rho R^3$")
# fig.suptitle(f"Velocity-like vs acceleration-like thrust error, $x_{{max}}/D$ = {xmax_user}")
# fig.tight_layout()
# fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

# %%
# fits = {xm: damping_inertia(thrust_deviation(ddf, xm)) for xm in xmax_fine}

# fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
# for ax, key, lab in [(axes[0], "C", r"$C/(\rho A U_\infty)$"),
#                      (axes[1], "M", r"$M/(\rho R^3)$")]:
#     for name, c in [("Total ΔT", "k"), *CONTRIB.items()]:
#         ax.plot(xmax_fine, [fits[xm].loc[name, key] for xm in xmax_fine],
#                 "-" if name != "Total ΔT" else "--", color=c, lw=2 if name == "Total ΔT" else 1.4,
#                 label=name)
#     ax.set_xlabel(r"$x_{max}/D$")
#     ax.set_title(lab)
#     ax.axhline(0, color="k", lw=0.6)
#     ax.grid(alpha=0.3)
# seen = dict(zip(*axes[0].get_legend_handles_labels()[::-1]))
# fig.tight_layout()
# fig.legend(seen.values(), seen.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5),
#            frameon=False)

# %%
# def first_harmonic(phase, y):
#     th = 2 * np.pi * np.asarray(phase, float)
#     Mx = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th)])
#     c0, c1, s1 = np.linalg.lstsq(Mx, np.asarray(y, float), rcond=None)[0]
#     return c0, c1 - 1j * s1                      # y ≈ c0 + Re(Z e^{iθ})


# lag, amp = [], []
# for xm in xmax_fine:
#     d = ddf[np.isclose(ddf["xmax"], xm) & ddf["phase"].notna()].sort_values("phase")
#     _, ZT = first_harmonic(d["phase"], -d["actuator"])       # thrust
#     _, ZD = first_harmonic(d["phase"], d["wake_def_out"])    # wake deficit at x+
#     lag.append(-np.angle(ZD / ZT))
#     amp.append(abs(ZD) / abs(ZT))
# lag = np.unwrap(lag) / (2 * np.pi)                # cycles; + = deficit lags thrust
# amp = np.array(amp)

# slope, icpt = np.polyfit(xmax_fine, lag, 1)
# U_c = f * D / slope
# u4 = float(np.squeeze(umm_sol.u4))
# print(f"U_c = {U_c:.2f} U_inf  (expect between u4 = {u4:.2f} and 1)")

# fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# axes[0].plot(xmax_fine, lag, "o", color="tab:orange", label="LES")
# axes[0].plot(xmax_fine, slope * xmax_fine + icpt, "k-",
#              label=rf"fit: $U_c$ = {U_c:.2f}$U_\infty$")
# for uc, ls in [(1.0, ":"), (u4, "--"), ((1 + u4) / 2, "-.")]:
#     axes[0].plot(xmax_fine, icpt + xmax_fine * f / uc, ls, color="0.5", lw=1,
#                  label=rf"$U_c$ = {uc:.2f}")
# axes[0].set_ylabel("Lag of wake deficit behind thrust [cycles]")
# axes[1].semilogy(xmax_fine, amp, "o-", color="tab:orange")
# axes[1].set_ylabel(r"$|\hat{D}_{x+}| / |\hat{T}|$  (oscillation amplitude ratio)")
# for ax in axes:
#     ax.set_xlabel(r"$x_{max}/D$")
#     ax.grid(alpha=0.3, which="both")
# fig.tight_layout()
# fig.legend(*axes[0].get_legend_handles_labels(), loc="center left",
#            bbox_to_anchor=(1.0, 0.5), frameon=False)

# %%
# s_act = ddf[np.isclose(ddf["xmax"], xmax_user) & ddf["phase"].isna()]["actuator"].iloc[0]
# fac = (1 - dev_user["u_t"] / UInf) ** 2
# T_qs = -s_act * fac
# T_les = T_qs + dev_user["dT"]
# tot = fit_user.loc["Total ΔT"]
# T_mod = T_qs + tot["mean"] - (tot["C"] * rho * A * UInf * dev_user["u_t"]
#                               + tot["M"] * rho * R**3 * dev_user["a_t"]) / F_inf

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(dev_user["phase"], T_les, "o-", color="tab:purple", label="LES")
# ax.plot(dev_user["phase"], T_qs, ":", color="0.2", lw=2, label="Quasi-steady")
# ax.plot(dev_user["phase"], T_mod, "--", color="crimson", label=r"QS $- C u_t - M a_t$")
# ax.set_xlabel("Phase")
# ax.set_ylabel(r"$T/\frac{1}{2}\rho C_T' A U_\infty^2$")
# ax.grid(alpha=0.3)
# fig.tight_layout()
# fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

# %%
# tot

# %%
# omega = 2 * np.pi * f
# L_d_slope = np.polyfit(xmax_fine, np.log(amp), 1)[0]
# L_d = -1 / L_d_slope if L_d_slope < 0 else np.inf
# print(f"U_c = {U_c:.3f},  L_d = {L_d:.3f}")

# sel = np.isclose(ddf["xmax"], xmax_user)
# les = ddf[sel & ddf["phase"].notna()].sort_values("phase")
# a_s = 1 - np.sqrt(-ddf[sel & ddf["phase"].isna()]["actuator"].iloc[0])

# # Smooth periodic motion from the measured phase means
# _, Zu = first_harmonic(motion.index.to_numpy(float), motion["uturb"].to_numpy())
# u_t_fn = lambda t: np.real(Zu * np.exp(1j * omega * t))
# x_t_fn = lambda t: np.real(Zu / (1j * omega) * np.exp(1j * omega * t))


# # %% Wake-memory induction model
# def kernel(s, L_d=np.inf):
#     """Axial induction at the disk centre from a wake ring at distance s."""
#     k = R**2 / (R**2 + s**2) ** 1.5
#     return k * np.exp(-s / L_d) if np.isfinite(L_d) else k


# def kernel_norm(L_d=np.inf, s_max=60.0, n=60001):
#     s = np.linspace(0, s_max, n)
#     return np.trapezoid(kernel(s, L_d), s)


# def induced_velocity(t, a_s, U_c, L_d=np.inf, g=0.0, n_age=8000):
#     """w(t) = g*w_QS(t) + (1-g)*(sum over wake shed at earlier times)."""
#     age = np.linspace(0, 40.0 / U_c, n_age)            # wake out to ~40 D
#     t_shed = t - age
#     # distance of each wake slice behind the current disk position
#     s = U_c * age - (x_t_fn(t) - x_t_fn(t_shed))
#     ds_dage = U_c - u_t_fn(t_shed)
#     w_qs_shed = a_s * (UInf - u_t_fn(t_shed))          # strength set when shed
#     w_wake = np.trapezoid(kernel(s, L_d) * ds_dage * w_qs_shed, age) / kernel_norm(L_d)
#     return g * a_s * (UInf - u_t_fn(t)) + (1 - g) * w_wake


# def predict_actuator(phase_eval, a_s, U_c, L_d=np.inf, g=0.0):
#     t = np.asarray(phase_eval, float) / f
#     w = np.array([induced_velocity(ti, a_s, U_c, L_d, g) for ti in t])
#     u_d = UInf - u_t_fn(t) - w
#     return -(u_d / UInf) ** 2, w                       # F_AD / F_inf, w


# def H_model(k_red, U_c, L_d=np.inf, g=0.0, s_max=60.0, n=300001):
#     """Small-amplitude transfer function w_hat / w_QS_hat."""
#     s = np.linspace(0, s_max, n)
#     ks = kernel(s, L_d)
#     om = np.atleast_1d(k_red) * UInf / D
#     Hw = np.array([np.trapezoid(ks * np.exp(-1j * o * s / U_c), s) for o in om])
#     return g + (1 - g) * Hw / np.trapezoid(ks, s)


# # %% LES transfer function at this frequency
# ph = les["phase"].to_numpy()
# ut = les["u_t"].to_numpy()
# u_d_les = np.sqrt(-les["actuator"].to_numpy()) * UInf
# w_les = UInf - ut - u_d_les
# _, Zw = first_harmonic(ph, w_les)
# _, Zq = first_harmonic(ph, a_s * (UInf - ut))
# H_les = Zw / Zq
# k_les = omega * D / UInf

# # Fit the immediate fraction g (the only free parameter) at this frequency
# Hw = H_model(k_les, U_c, L_d, g=0.0)[0]
# g_fit = np.real(np.conj(1 - Hw) * (H_les - Hw)) / abs(1 - Hw) ** 2
# print(f"LES:        gain = {abs(H_les):.2f}, lag = {-np.degrees(np.angle(H_les)):.0f} deg")
# print(f"wake only:  gain = {abs(Hw):.2f}, lag = {-np.degrees(np.angle(Hw)):.0f} deg")
# print(f"g fit = {g_fit:.2f}")

# %%
# k_grid = np.logspace(-1, np.log10(30), 120)
# fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# for gv, ls, lab in [(0.0, "--", "Wake memory only"),
#                     (g_fit, "-", f"Wake memory + immediate (g = {g_fit:.2f})")]:
#     Hk = H_model(k_grid, U_c, L_d, gv)
#     axes[0].semilogx(k_grid, abs(Hk), ls, color="crimson", label=lab)
#     axes[1].semilogx(k_grid, -np.degrees(np.angle(Hk)), ls, color="crimson")
# for ax, val in [(axes[0], abs(H_les)), (axes[1], -np.degrees(np.angle(H_les)))]:
#     ax.plot(k_les, val, "o", color="tab:purple", ms=9, label="LES")
#     ax.axvline(k_les, color="0.7", lw=0.8)
#     ax.set_xlabel(r"Reduced frequency $\omega D/U_\infty$")
#     ax.grid(alpha=0.3, which="both")
# axes[0].axhline(1, color="0.2", ls=":", label="Quasi-steady")
# axes[0].set_ylabel(r"Induction gain $|\hat w/\hat w_{QS}|$")
# axes[1].set_ylabel("Induction lag [deg]")

# phase_fine = np.linspace(0, 1, 101)
# t_f = phase_fine / f
# axes[2].plot(ph, les["actuator"], "o", color="tab:purple", label="LES")
# axes[2].plot(phase_fine, -(1 - a_s) ** 2 * (1 - u_t_fn(t_f) / UInf) ** 2,
#              ":", color="0.2", lw=2, label="Quasi-steady")
# for gv, ls in [(0.0, "--"), (g_fit, "-")]:
#     F_mod, _ = predict_actuator(phase_fine, a_s, U_c, L_d, gv)
#     axes[2].plot(phase_fine, F_mod, ls, color="crimson")
# axes[2].set_xlabel("Phase")
# axes[2].set_ylabel(r"$F_{AD}/\frac{1}{2}\rho C_T' A U_\infty^2$")
# axes[2].grid(alpha=0.3)

# seen = {}
# for ax in axes:
#     for h, l in zip(*ax.get_legend_handles_labels()):
#         seen.setdefault(l, h)
# fig.tight_layout()
# fig.legend(seen.values(), seen.keys(), loc="center left",
#            bbox_to_anchor=(1.0, 0.5), frameon=False)

# %%
# # %% Parameter-free vortex-wake dynamic inflow (linear theory, on-axis kernel)
# def vortex_wake_model(Ct_prime, Zu, f, n_cycles=20, dt=0.005, L_wake=40.0,
#                       L_d=np.inf, n_iter=6):
#     """
#     Time-march w(t) = (1/2 rho A) * sum_k k(s) T(t') dt'.
#     Zu: complex first harmonic of u_t (so u_t = Re[Zu e^{i w t}]).
#     Returns phase, u_t, w, F_AD/F_inf over the last cycle.
#     """
#     om = 2 * np.pi * f
#     a0 = Ct_prime / (4 + Ct_prime)                 # linear-theory steady induction
#     n_hist = int(L_wake / ((1 - a0) * UInf) / dt)
#     nt = n_hist + int(n_cycles / f / dt)
#     t = (np.arange(nt) - n_hist) * dt

#     u_t = np.real(Zu * np.exp(1j * om * t))
#     x_t = np.real(Zu / (1j * om) * np.exp(1j * om * t))
#     k = lambda s: (R**2 / (R**2 + s**2) ** 1.5) * np.exp(-np.clip(s, 0, None) / L_d)

#     T = np.zeros(nt); w = np.zeros(nt); X = np.zeros(nt)
#     # Initialize the wake history with the steady solution
#     T_s = 0.5 * rho * Ct_prime * A * (UInf * (1 - a0)) ** 2
#     T[:n_hist] = T_s; w[:n_hist] = a0 * UInf
#     X[:n_hist] = x_t[n_hist] + (n_hist - np.arange(n_hist)) * dt * UInf * (1 - a0)

#     for n in range(n_hist, nt):
#         X[:n] += (UInf - w[:n]) * dt * 0 + 0          # placeholder (see below)
#         # rings move with local flow; use each ring's release-time induction
#         X[:n] += (UInf - w[:n]) * dt
#         X[n] = x_t[n]
#         s = X[:n] - x_t[n]
#         w_old = np.sum(k(s) * T[:n]) * dt / (2 * rho * A)
#         wn = w[n - 1]
#         for _ in range(n_iter):                     # implicit: newest ring sees current T
#             Tn = 0.5 * rho * Ct_prime * A * max(UInf - u_t[n] - wn, 0) ** 2
#             wn = w_old + 0.5 * k(0.0) * Tn * dt / (2 * rho * A)
#         w[n], T[n] = wn, Tn

#     last = t >= (n_cycles - 1) / f
#     phase = (t[last] * f) % 1.0
#     F = -(UInf - u_t[last] - w[last]) ** 2 / UInf**2
#     o = np.argsort(phase)
#     return phase[o], u_t[last][o], w[last][o], F[o]


# _, Zu = first_harmonic(motion.index.to_numpy(float), motion["uturb"].to_numpy())
# ph_m, ut_m, w_m, F_m = vortex_wake_model(Ct_prime, Zu, f)

# %%
# # %% Compare with LES and quasi-steady, and get the transfer function
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# axes[0].plot(les["phase"], les["actuator"], "o", color="tab:purple", label="LES")
# axes[0].plot(ph_m, -(1 - a_s)**2 * (1 - ut_m)**2, ":", color="0.2", lw=2,
#              label="Quasi-steady")
# axes[0].plot(ph_m, F_m, "-", color="crimson", label="Vortex-wake model (no fit)")
# axes[0].set_ylabel(r"$F_{AD}/\frac{1}{2}\rho C_T'AU_\infty^2$")

# axes[1].plot(les["phase"], w_les, "o", color="tab:purple")
# axes[1].plot(ph_m, a_s * (UInf - ut_m), ":", color="0.2", lw=2)
# axes[1].plot(ph_m, w_m, "-", color="crimson")
# axes[1].set_ylabel(r"Induced velocity $w/U_\infty$")
# for ax in axes:
#     ax.set_xlabel("Phase"); ax.grid(alpha=0.3)
# fig.tight_layout()
# fig.legend(*axes[0].get_legend_handles_labels(), loc="center left",
#            bbox_to_anchor=(1.0, 0.5), frameon=False)

# _, Zw_m = first_harmonic(ph_m, w_m); _, Zq_m = first_harmonic(ph_m, a_s * (UInf - ut_m))
# H_m = Zw_m / Zq_m
# print(f"LES:   gain {abs(H_les):.2f}, lag {-np.degrees(np.angle(H_les)):.0f} deg")
# print(f"Model: gain {abs(H_m):.2f}, lag {-np.degrees(np.angle(H_m)):.0f} deg")

# # Sensitivity to the only "soft" ingredient
# for Ld in (np.inf, 10.0, 5.0, 2.0):
#     *_, F_L = vortex_wake_model(Ct_prime, Zu, f, L_d=Ld)
#     print(f"L_d = {Ld}: F range {F_L.min():.3f} to {F_L.max():.3f}")

# %%
# # %% First-principles C and M via the induction transfer function H
# from scipy.special import ellipk, ellipe

# a = a_s
# ud0 = UInf * (1 - a)
# om = 2 * np.pi * f
# G = Ct_prime * a * (1 - a)                       # thrust sensitivity factor

# def CM_from_H(H, omega=om):
#     Mfac = A * UInf / (omega * R**3)
#     return G * np.real(1 - H), G * Mfac * np.imag(1 - H)

# def H_from_CM(C_t, M_t, omega=om):
#     Mfac = A * UInf / (omega * R**3)
#     return 1 - (C_t + 1j * M_t / Mfac) / G

# # Kernels: induction at the disk from a ring at distance s (unit steady integral)
# def k_axis(s):
#     return R**2 / (R**2 + np.asarray(s)**2) ** 1.5

# _xg, _wg = np.polynomial.legendre.leggauss(400)
# _r, _wr = 0.5 * R * (_xg + 1), 0.5 * R * _wg
# def k_disk(s, eps=0.0):
#     """Disk-averaged axial induction from a ring (elliptic integrals).
#     eps: LES actuator smoothing width (from the input file, not fitted)."""
#     s = np.sqrt(np.atleast_1d(s)[:, None]**2 + eps**2)
#     den = (R + _r)**2 + s**2
#     m = 4 * R * _r / den
#     uz = (ellipk(m) + (R**2 - _r**2 - s**2) / ((R - _r)**2 + s**2) * ellipe(m)) \
#          / (2 * np.pi * np.sqrt(den))
#     return 2 * (2 / R**2) * np.sum(uz * _r * _wr, axis=1)

# s_grid = np.unique(np.r_[np.geomspace(1e-5, 1.0, 3000), np.linspace(1.0, 80.0, 16000)])

# def H_ring(omega, kernel):
#     ks = kernel(s_grid)
#     q = np.atleast_1d(omega)[:, None] / ud0
#     K = np.trapezoid(ks * np.exp(-1j * q * s_grid), s_grid, axis=1) / np.trapezoid(ks, s_grid)
#     return K / ((1 - a) + a * K)

# def H_pitt_peters(omega):
#     m = (8 / 3) * R**3 / A
#     return 1 / (1 + 1j * np.atleast_1d(omega) * m / (2 * UInf))

# eps_AD = 0.0   # <-- set to your actuator-disk smoothing width to include it
# models = {
#     "Vortex rings, on-axis":        lambda w: H_ring(w, k_axis),
#     "Vortex rings, disk-averaged":  lambda w: H_ring(w, k_disk),
#     "Vortex rings, disk-avg + ε":   lambda w: H_ring(w, lambda s: k_disk(s, eps_AD)),
#     "Pitt–Peters (8/3 ρR³)":        H_pitt_peters,
# }

# # %% Comparison table at this frequency
# tot, unc = fit_user.loc["Total ΔT"], fit_user.loc["Leak + residual"]
# H_les = H_from_CM(tot["C"], tot["M"])
# rows = {"LES (fit)": (H_les, tot["C"], tot["M"])}
# for name, Hf in models.items():
#     H = Hf(om)[0]
#     rows[name] = (H, *CM_from_H(H))
# tab = pd.DataFrame({n: {"gain": abs(v[0]), "lag [deg]": -np.degrees(np.angle(v[0])),
#                         "C~": v[1], "M~": v[2]} for n, v in rows.items()}).T
# print(tab.round(3))
# print(f"LES uncertainty from residual: C~ ±{abs(unc['C']):.3f}, M~ ±{abs(unc['M']):.3f}")

# # %% Predicted C~(k) and M~(k): the first-principles curves
# k_red = np.logspace(-1, 1.5, 120)
# omg = k_red * UInf / D
# fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
# for (name, Hf), ls in zip(models.items(), ["--", "-", "-.", ":"]):
#     C_k, M_k = CM_from_H(Hf(omg), omg)
#     axes[0].semilogx(k_red, C_k, ls, label=name)
#     axes[1].semilogx(k_red, M_k, ls)
# k_les = om * D / UInf
# axes[0].errorbar(k_les, tot["C"], yerr=abs(unc["C"]), fmt="o", color="k", label="LES")
# axes[1].errorbar(k_les, tot["M"], yerr=abs(unc["M"]), fmt="o", color="k")
# axes[0].axhline(G, color="0.6", lw=0.8, label="Frozen-induction limit")
# axes[0].set_ylabel(r"$\tilde C = C/(\rho A U_\infty)$")
# axes[1].set_ylabel(r"$\tilde M = M/(\rho R^3)$")
# for ax in axes:
#     ax.set_xlabel(r"Reduced frequency $\omega D/U_\infty$")
#     ax.grid(alpha=0.3, which="both")
# fig.tight_layout()
# fig.legend(*axes[0].get_legend_handles_labels(), loc="center left",
#            bbox_to_anchor=(1.0, 0.5), frameon=False)
