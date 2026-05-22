# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: simspadeops-YbNxeWqo-py3.11
#     language: python
#     name: python3
# ---

# %%
import os
import analysis_utils as au
import quick_metadata_plots as qmplt
import padeopsIO as pio
import matplotlib.pyplot as plt
import numpy as np
import streamtube
import pandas as pd
import seaborn as sns

# %%
import matplotlib as mpl
# %matplotlib inline
mpl.rcParams['figure.dpi'] = 300

# %%
sim_folder =  os.path.join(au.DATA_PATH, "F_0024_SU_Files/")
run_folder = au.get_run_folder(sim_folder, 0)
sim = pio.BudgetIO(run_folder, padeops = True, runid = 0, normalize_origin="turbine", verbose=False)

# %%
sim.existing_terms()

# %%
xlim = [-2, 15]
ylim =  [-2.5, 2.5]
zlim = 0
phases = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# %%
ds = sim.slice(budget_terms=["ubar", "pbar"], xlim = xlim, ylim = ylim, zlim = 0, phase = phases[2])
ds['ubar'].imshow(cmap = "bwr",)
ds['pbar'].imshow(cmap = "bwr", vmin = 0.4, vmax = -0.4)


# %%
