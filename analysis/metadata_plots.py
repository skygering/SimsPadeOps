import padeopsIO as pio
import analysis_utils as au
# from pathlib import Path
import matplotlib.pyplot as plt
# import numpy as np
import os
import csv

suite_path = au.DATA_PATH + "F_0000_X_Files/"
sub_folders = [f.path for f in os.scandir(suite_path) if f.is_dir()].sort()

# Open the CSV file for reading
fields = []
rows = []
with open(suite_path + "sim_ids.csv", mode='r') as file:
    csvreader = csv.reader(file)
    # extracting field names through first row
    fields = next(csvreader)
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

fig, ax = plt.subplots(figsize=(9, 3))

for i, folder in enumerate(sub_folders):
    sim = pio.BudgetIO(folder, padeops = True, runid = 1)
    dt = sim.input_nml["input"]["dt"]
    trans_tau = (50 / dt) + 1
    power = sim.read_turb_power(turb=1)[(trans_tau):-1]
    time = [dt * n for n in range(trans_tau, len(power))]

    label = ""
    nfields = len(fields)
    for j, fld in enumerate(fields):
        label += f"{fld}: {rows[i][j]}"
        if nfields == j + 1: 
            label += ", "

    ax.plot(time, power, label=label, lw=0.7)

plt.legend(loc="upper left")



# args = au.arg_parser()
# out_dir = Path(au.DATA_PATH + args.write_dir)
# out_dir = args.write_dir

sim = pio.BudgetIO(out_dir, padeops = True, runid = 1)
# sim = pio.BudgetIO(out_dir, npz = True, filename = args.filename)
dt = sim.input_nml["input"]["dt"]
power = sim.read_turb_power(turb=1)
time = [dt * n for n in range(0, len(power))]
fig, ax = plt.subplots(figsize=(9, 3))


ax.plot(time, power, label='A', lw=0.7)

ax.set_xlabel('Time step (-)')
# plt.show()