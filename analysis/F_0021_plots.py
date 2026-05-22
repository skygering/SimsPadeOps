# %% [markdown]
# Here, I am exploring the output of the surging inflow cases. This is to determine if my phase budget / wave budget code within PadeOps is correct. I started by just running a simple case with non-dimensional amplitude of 1.0 and non-dimensional frequency of 1.0 as well. Everything is still non-dimensionalized with respect to a non-existant turbine. 
#
# I ran three cases originally:
# - `Sim_0000_Time_Budget0`: this is the time budget before I made any changes
# - `Sim_0000_Time_Budget1`: this is the time budget with the changes to the time average budget files
# - `Sim_0001_Phase_Budget0`: this is the time and phase averaged budgets based on the turbine inflow
#
# With this high of a frequency, it was hard to get a handle on what was going on in the budgets because things were changing so quickly.
#
# I then ran two more cases, which have a lower amplitude (0.4) and a lower frequency (0.2).
# - `Sim_0000`: this is the time budget before I made any changes
# - `Sim_0001`: this is the time budget with the changes to the time average budget files
#
# After I make sure that this is working, I will work on writing up the offline budget calculations for the surging inflow. 
#
# After that, I will add back in the phase-averaged budget for moving turbines and test my code there as well.

# %%
dpi = 300
# %matplotlib inline

# %%
import os
import analysis_utils as au
import padeopsIO as pio
import quick_metadata_plots as mplts

# %%
sim_21_all_folder = os.path.join(au.DATA_PATH, "F_0021_Files")
time_budget = "Sim_0000"  # f = 0.2, A = 0.4, time budget only
phase_budget = "Sim_0001"   # f = 0.2, A = 0.4, both time and phase budget
phase_turb_budget = "Sim_0002"   # f = 0.2, A = 0.4, both time and phase budget with turbine

# %%
sim_time_budget = pio.BudgetIO(os.path.join(sim_21_all_folder, time_budget), padeops=True, runid=0, normalize_origin="turbine")
sim_phase_budget = pio.BudgetIO(os.path.join(sim_21_all_folder, phase_budget), padeops=True, runid=0, normalize_origin="turbine")
sim_phase_turb_budget = pio.BudgetIO(os.path.join(sim_21_all_folder, phase_turb_budget), padeops=True, runid=0, normalize_origin="turbine")

# %%
sim_phase_turb_budget.slice(budget_terms=['ubar'], xlim=[-5, 20], ylim=[-5, 5], zlim=0)

# %%
ds_time0 = sim_time_budget.slice(budget_terms=['ubar'], xlim=[-5, 20], ylim=[-5, 5], zlim=0)
ds_time0['ubar'].imshow(vmin = 0.4, vmax = 1.6)

# %%
ds_phase = sim_phase_budget.slice(budget_terms=['ubar'], xlim=[-5, 20], ylim=[-5, 5], zlim=0)
ds_phase['ubar'].imshow(vmin = 0.4, vmax = 1.6)

# %%
for i in [0.1 * i for i in range(10)]:
    print(f"Phase {i}")
    ds_phase_i = sim_phase_budget.slice(budget_terms=['ubar'], xlim=[-5, 20], ylim=[-5, 5], zlim=0, phase = i)
    ds_phase_i['ubar'].imshow(vmin = 0.4, vmax = 1.6)

# %%
ds_phase_turb = sim_phase_turb_budget.slice(budget_terms=['ubar'], xlim=[-5, 20], ylim=[-5, 5], zlim=0)
ds_phase_turb['ubar'].imshow(vmin = 0.4, vmax = 1.6)

# %%
for i in [0.1 * i for i in range(10)]:
    print(f"Phase {i}")
    ds_phase_turb_i = sim_phase_turb_budget.slice(budget_terms=['ubar'], xlim=[-5, 20], ylim=[-5, 5], zlim=0, phase = i)
    ds_phase_turb_i['ubar'].imshow(vmin = 0.4, vmax = 1.6)

# %%
