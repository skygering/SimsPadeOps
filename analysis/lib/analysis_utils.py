import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import padeopsIO as pio
import pyGCS as pg
from scipy.signal import find_peaks
import statistics
import glob
from pathlib import Path
import lib.quick_metadata_plots as mplts
import pandas as pd

DATA_PATH = os.environ['SCRATCH'] + "/DataPadeOps/"

def arg_parser(arg_list = ["write_dir", "filename"]):
    parser = argparse.ArgumentParser()
    for a in arg_list:
        parser.add_argument(a)
    args = parser.parse_args()
    return args

def get_run_folder(sim_folder, runid):
    print(runid)
    run_str = "Sim_000"
    if runid > 99:
        run_str = "Sim_0"
    elif runid > 9:
        run_str = "Sim_00"
    run_str += str(runid)
    return os.path.join(sim_folder, run_str)

def power_to_Cp(power, uinf = 1, rho = 1, tilt = 0):
    """
    Calculate Cp from PadeOps turbine power input for simple momentum theory of an actuator disk
    """
    return power / (0.5 * rho * math.pi * 0.5**2 * (uinf * np.cos(tilt))**3)

def vel_to_a(udisk, uinf = 1, tilt = 0):
    a = 1 - (udisk / (uinf * np.cos(tilt)))
    # TODO: want to remove the turbine motion from both terms
    # urel = uturb + udisk
    # uinf_adjusted = uinf - uturb
    # a = 1 - (urel / uinf_adjusted)
    return a

def analytical_a(CT):
    # note that CT is actually CT'
    return CT / (4 + CT)

def a_to_Cp(a, alg = "classical"):
    return 4 * a * (1 - a)**2

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def x_zoom_plot(zoom, xs, ys):
    s_i = find_nearest(xs, zoom[0])
    e_i = find_nearest(xs, zoom[1])
    return xs[s_i : e_i + 1], ys[s_i : e_i + 1]

def avg_cp_info(sim_nums, sim_folder, peak = False, trough = False):
    cp_sol = [0.0 for _ in sim_nums]
    cells = [0.0 for _ in sim_nums]
    for i, e in enumerate(sim_nums):
        run_str = "Sim_000" if e < 10 else "Sim_00"
        run_str += str(e)
        run_folder = os.path.join(sim_folder, run_str)
        sim = pio.BudgetIO(run_folder, padeops = True, runid = e)
        dt = sim.input_nml["input"]["dt"]
        cells[i] = sim.input_nml["input"]["nx"] * sim.input_nml["input"]["ny"] * sim.input_nml["input"]["nz"]
        trans_tau = int(math.ceil(50 / dt) + 1)
        power = sim.read_turb_power("all", turb=1)[trans_tau:]
        Cp = [power_to_Cp(p) for p in power]
        if peak:
            cp_peak_idx, _ = find_peaks(Cp)
            cp_peak_vals = [Cp[pi] for pi in cp_peak_idx]
            cp_sol[i] = statistics.mean(cp_peak_vals)
        elif trough:
            cp_trough_idx, _ = find_peaks([-c for c in Cp])
            cp_trough_vals = [Cp[ti] for ti in cp_trough_idx]
            cp_sol[i] = statistics.mean(cp_trough_vals)
        else:
            cp_sol[i] = statistics.mean(Cp)
    return cp_sol, cells


def plot_gci_cp(sim_nums, sim_folder, plot_title = None, zoom = None, to_plot = True, gci = False, peak = False, trough = False):
    fig, ax = plt.subplots(figsize=(9, 3))
    labels = ["coarse", "medium", "fine"]
    cp_sol = [0, 0, 0]
    cells = [0, 0, 0]
    for i, e in enumerate(sim_nums):
        run_folder = os.path.join(sim_folder, "Sim_000" + str(e))
        sim = pio.BudgetIO(run_folder, padeops = True, runid = e)
        dt = sim.input_nml["input"]["dt"]
        cells[i] = sim.input_nml["input"]["nx"] * sim.input_nml["input"]["ny"] * sim.input_nml["input"]["nz"]
        trans_tau = int(math.ceil(50 / dt) + 1)
        # TODO: ask Kirby if default should be "all", rather than None
        # as it was confusing and when it is saved as a file first, they all printed I think?
        power = sim.read_turb_power("all", turb=1)[trans_tau:]
        Cp = [power_to_Cp(p) for p in power]
        time = [50 + dt * n for n in range(len(Cp))]
        if zoom is not None:
            s_i = find_nearest(time, zoom[0])
            e_i = find_nearest(time, zoom[1])
            time = time[s_i : e_i + 1]
            Cp = Cp[s_i : e_i + 1]
        if gci:
            if peak:
                cp_peak_idx, _ = find_peaks(Cp)
                cp_peak_vals = [Cp[pi] for pi in cp_peak_idx]
                cp_sol[i] = statistics.mean(cp_peak_vals)
            elif trough:
                cp_trough_idx, _ = find_peaks([-c for c in Cp])
                cp_trough_vals = [Cp[ti] for ti in cp_trough_idx]
                cp_sol[i] = statistics.mean(cp_trough_vals)
            else:
                cp_sol[i] = statistics.mean(Cp)
        if to_plot:
            ax.plot(time, Cp, label = labels[i], lw=0.7)
    if to_plot:
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(sim_folder, plot_title))
    if gci:
        gci = pg.GCI(dimension=3, simulation_order=4, volume=25*10*5, cells=cells, solution=cp_sol)
        print("Solution Values: " + str(cp_sol))
        print("GCI Values: " + str(gci.get('gci')))
        print("Asymptotic GCI: " + str(gci.get('asymptotic_gci')))
        print("Refinement Ratio: " + str(gci.get('refinement_ratio')))


def get_TI_fact(path, logfile, start_TIDX):
    log_file_dict = pio.query_logfile(os.path.join(path, logfile), search_terms=["TI_fact"], crop_equal = False)
    return np.average(log_file_dict["TI_fact"][start_TIDX:])

def get_TI_inst(path, logfile, start_TIDX):
    log_file_dict = pio.query_logfile(os.path.join(path, logfile), search_terms=["TI_inst"], crop_equal = False)
    return np.average(log_file_dict["TI_inst"][start_TIDX:])

def load_simulation_timeseries(run_folder, id_str, freq, spinup_time=75):
    sim = pio.BudgetIO(run_folder, padeops=True, runid=0, normalize_origin="turbine")

    power = sim.read_turb_power("all", turb=1)
    uvel = sim.read_turb_uvel("all", turb=1)

    n_dumped = min(len(power), len(uvel))
    power, uvel = power[:n_dumped], uvel[:n_dumped]

    if n_dumped == 0:
        raise RuntimeError("No turbine data found")

    # -----------------------------
    # locate one or more log files
    # -----------------------------
    log_files = sorted(glob.glob(f"*{id_str}*.o[0-9]*", root_dir=run_folder))

    if not log_files:
        extracted = extract_sim_log_from_batches(run_folder)
        if extracted:
            log_files = [extracted.name]

    if not log_files:
        raise RuntimeError("Could not locate any log file")

    log_paths = [os.path.join(run_folder, lf) for lf in log_files]

    # -----------------------------
    # parse + merge logs
    # -----------------------------
    if len(log_paths) == 1:
        log = _parse_one_log(log_paths[0])["log"]
    else:
        infos = [_parse_one_log(p) for p in log_paths]
        log = _concat_logs_prefer_last(infos)  # restart overlaps keep second run values

    # -----------------------------
    # align with turbine outputs
    # -----------------------------
    time = np.asarray(log["Time"])[:n_dumped]
    tidx_raw = np.asarray(log["TIDX"])
    tidx = np.insert(tidx_raw, 0, 0).astype(int)[:n_dumped]

    mask = time > spinup_time
    time, tidx = time[mask], tidx[mask]

    # tilt (optional)
    tilt_deg = np.asarray(log["tilt"])
    if tilt_deg.size == 0:
        tilt = np.zeros_like(time)
    else:
        tilt = np.deg2rad(tilt_deg[:n_dumped][mask])

    # phase (optional)
    phase_arr = np.asarray(log["phase"])
    if phase_arr.size == 0:
        if freq > 0 and len(time) > 1:
            T = 1.0 / freq
            dt = time[1] - time[0]
            dp = dt / T
            start_phase = np.mod(time[0] / T, 1.0)
            phase = np.mod(start_phase + np.arange(len(time)) * dp, 1.0)
        else:
            phase = np.zeros_like(time)
    else:
        phase = phase_arr[:n_dumped][mask]

    data = {
        "Time": time,
        "TIDX": tidx,
        "Tilt": tilt,
        "UTurb": np.asarray(log["uturb"])[:n_dumped][mask],
        "DeltaX": np.asarray(log["delta"])[:n_dumped][mask],
        "Phase": phase,
        "Power": np.asarray(power)[mask],
        "UDisk": np.asarray(uvel)[mask],
    }
    return data

def _parse_one_log(path):
    log = pio.query_logfile(
        path,
        search_terms=["tilt", "uturb", "Time", "TIDX", "delta", "phase"],
        crop_equal=False,
    )

    for k in ["Time", "TIDX", "tilt", "uturb", "delta", "phase"]:
        if k not in log:
            log[k] = np.array([])

    time = np.asarray(log["Time"])
    tidx = np.asarray(log["TIDX"]).astype(int) if len(log["TIDX"]) else np.array([], dtype=int)

    with open(path, "r", errors="ignore") as f:
        txt = f.read(20000)
    is_restart = "RESTART FILE USED" in txt

    return {
        "path": path,
        "is_restart": is_restart,
        "first_time": time[0] if time.size else np.inf,
        "first_tidx": tidx[0] if tidx.size else np.iinfo(np.int64).max,
        "log": log,
    }

def _concat_logs_prefer_last(log_infos):
    # base run first, restart later
    log_infos = sorted(log_infos, key=lambda d: (d["is_restart"], d["first_tidx"], d["first_time"]))

    keys = ["Time", "TIDX", "tilt", "uturb", "delta", "phase"]
    out = {
        k: np.concatenate([np.asarray(info["log"][k]) for info in log_infos if np.asarray(info["log"][k]).size > 0])
           if any(np.asarray(info["log"][k]).size > 0 for info in log_infos) else np.array([])
        for k in keys
    }

    # de-dup by TIDX if available; keep LAST occurrence
    key = out["TIDX"] if out["TIDX"].size else out["Time"]
    if key.size:
        rev = key[::-1]
        _, rev_idx = np.unique(rev, return_index=True)   # first in reversed == last in original
        keep = (key.size - 1 - rev_idx)
        keep.sort()
        out = {k: v[keep] if v.size else v for k, v in out.items()}

    return out

def extract_sim_log_from_batches(sim_dir):
    """
    Extract per-simulation log from batch output files.
    Returns path to extracted log file or None if not found.
    Prefers logs ending with end_token1 (success); falls back to end_token2 (error)
    only if no successful run is found across all batch logs.
    """
    sim_dir = Path(sim_dir)
    sim_name = sim_dir.name
    parent = sim_dir.parent
    batch_logs = sorted(parent.glob("*.o[0-9]*"))
    if not batch_logs:
        return None

    start_token = f"Running {sim_name}"
    end_token1 = f"Finished {sim_name}"
    end_token2 = f"ERROR: {sim_name}"

    backup_extracted = None
    backup_source = None

    for batch_log in batch_logs:
        with batch_log.open("r", errors="ignore") as f:
            lines = f.readlines()

        inside = False
        extracted = []
        found_end = None

        for line in lines:
            if start_token in line:
                inside = True
                extracted = [line]  # Reset in case of multiple runs in same file
                found_end = None
                continue
            if inside:
                extracted.append(line)
                if end_token1 in line:
                    found_end = "success"
                    break
                elif end_token2 in line:
                    found_end = "error"
                    break

        if found_end == "success" and extracted:
            out_file = sim_dir / f"{sim_name}_from_{batch_log.name}"
            with out_file.open("w") as out:
                out.write(f"# Extracted from batch log: {batch_log}\n")
                out.write(f"# Simulation: {sim_name}\n\n")
                out.writelines(extracted)
            return out_file

        elif found_end == "error" and extracted and backup_extracted is None:
            # Save the first error log as a backup, keep searching
            backup_extracted = extracted
            backup_source = batch_log

    # No successful run found — fall back to the error log if we have one
    if backup_extracted and backup_source:
        out_file = sim_dir / f"{sim_name}_from_{backup_source.name}"
        with out_file.open("w") as out:
            out.write(f"# Extracted from batch log: {backup_source}\n")
            out.write(f"# Simulation: {sim_name}\n\n")
            out.writelines(backup_extracted)
        return out_file

    return None

def build_sim_dataframe(data, meta):
    df = pd.DataFrame({
        "id": meta["id"],
        "CT_prime": meta["CT_prime"],
        "Surge_Amplitude": meta["Surge_Amplitude"],
        "Pitch_Amplitude": meta["Pitch_Amplitude"],
        "Frequency": meta["Frequency"],
        "nx": meta["nx"],
        "ny": meta["ny"],
        "nz": meta["nz"],
        "Lx": meta["Lx"],
        "Ly": meta["Ly"],
        "Lz": meta["Lz"],
        "dt": meta["dt"],
        "filterWidth": meta["filterWidth"],
        "useCorrection": meta["useCorrection"],
        **data
    })

    # detect periods
    phase = df["Phase"].values
    period = np.zeros(len(phase), dtype=int)
    period[1:] = np.cumsum(np.diff(phase) < 0)
    df["Period"] = period
    return df



# def get_instantaneous_data(sim_folder, runid, tidx = "all", field = "u", **kwargs):
#     run_folder = get_run_folder(sim_folder, runid)
#     sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine")
#     # first and second dimension 
#     # 
#     if tidx == "all"
#         tidx_list = sim.unique_tidx()
#         ntimesteps = len(tidx_list)
#     else:
#         tidx_list = [tidx]
#         ntimesteps = 1
#     for tidx_val in tidx_list: