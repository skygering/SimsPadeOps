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
plt.savefig(suite_path + 'suite_power.png')