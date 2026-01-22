# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math


# %% [markdown]
# # Load Dataset and Minimal Cleaning

# %%
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# %%
old_df = pd.read_csv("/Users/sky/src/HowlandLab/data/target_grid_search.csv")
df = old_df.copy(deep = True)
df = df.dropna()
df

# %%
old_df = pd.read_csv("/Users/sky/src/HowlandLab/data/target_grid_search.csv")
df = old_df.copy(deep = True)
df = df.dropna()
# add in new columns of ease of use
df['Movement'] = df.apply(lambda row: ("Stationary" if row.marker == "o" else ("Surging" if row.marker == "s" else "Pitching")), axis = 1)
df["Resolution (ny)"] = df["ny"].astype(str)
df["h"] = df.apply(lambda row: round(math.sqrt((25/row.nx)**2 + 2 * (10/row.ny)**2), ndigits = 2), axis = 1)
df["f"] = df.apply(lambda row: math.trunc(row.filterFactor*10) / 10, axis = 1)
# remove unneeded columns/values
cols_to_keep = ["Movement", "Resolution (ny)", "filterFactor", "f", "useCorrection", "turbulence", "CT_prime"]
df = df.drop_duplicates(subset = cols_to_keep, keep = 'last')


# %%
def get_movement(df, type_str):
    return df[df["Movement"] == type_str]
def get_stationary(df):
    return get_movement(df, "Stationary")
def get_surging(df):
    return get_movement(df, "Surging")
def get_pitching(df):
    return get_movement(df, "Pitching")


# %%
def get_mean(df):
    return df[cols_to_keep + ["mean_CT", "mean_an", "mean_Cp"]]
def get_std(df):
    return df[cols_to_keep + ["std_CT", "std_an", "std_Cp"]]
def get_skew(df):
    return df[cols_to_keep + ["skewness_CT", "skewness_an", "skewness_Cp"]]
def get_kurtosis(df):
    return df[cols_to_keep + ["kurtosis_CT", "kurtosis_an", "kurtosis_Cp"]]


# %%
mean_df = get_mean(df)
std_df = get_std(df)
skew_df = get_skew(df)
kurtosis_df = get_kurtosis(df)


# %% [markdown]
# # Plot Dataset Basics

# %%
def analytical_a(CT):
    # note that CT is actually CT'
    return CT / (4 + CT)

def a_to_Cp(a, alg = "classical"):
    return 4 * a * (1 - a)**2


# %%
def add_reference_lines(axes, ctp_vals, analytical_an, analytical_ct, analytical_cp, label, color):
    ((ax0, ax1), (ax2, ax3)) = axes
    sns.lineplot(ax = ax0, x = analytical_an, y = analytical_ct, color = color, label = label)
    sns.lineplot(ax = ax1, x = ctp_vals, y = analytical_ct, color = color)
    sns.lineplot(ax = ax2, x = analytical_an, y = analytical_cp, color = color)
    sns.lineplot(ax = ax3, x = ctp_vals, y = analytical_cp, color = color)
    return


# %%
def four_plot_layout(df, CT_key, an_key, Cp_key, classical = False, pitch = False, surge = False, title = "", ax_label_type = "", fill_between = False, **kwargs):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ((ax0, ax1), (ax2, ax3)) = axes

    sns.scatterplot(ax = ax0, data = df, x = an_key, y = CT_key, legend = True, **kwargs)
    sns.scatterplot(ax = ax1, data = df, x = "CT_prime", y = CT_key, legend = False, **kwargs)
    sns.scatterplot(ax = ax2, data = df, x = an_key, y = Cp_key, legend = False, **kwargs)
    sns.scatterplot(ax = ax3, data = df, x = "CT_prime", y = Cp_key, legend = False, **kwargs)

    ctp_vals = np.linspace(0.75, 6.25, 50)
    analytical_an = [analytical_a(ctp) for ctp in ctp_vals]
    analytical_ct = [ctp * (1 - analytical_an[i])**2 for (i, ctp) in enumerate(ctp_vals)]
    analytical_cp = [a_to_Cp(a) for a in analytical_an]
    if classical:
        # classical momentum values for statinary turbine
        label = "Classical"
        color = "k"
        add_reference_lines(axes, ctp_vals, analytical_an, analytical_ct, analytical_cp, label, color)
    if surge:
            s = 0.5
            surge_cp = [cp * (1 + (3 * s **2) / 2) for cp in analytical_cp]
            label = "Johlas (Surge)"
            color = "b"
            add_reference_lines(axes, ctp_vals, analytical_an, analytical_ct, surge_cp, label, color)
    if pitch:
            ps = [np.cos(5 * np.pi / 180 * np.sin(x))**3 for x in np.linspace(0, 2 * np.pi, 250)]
            pitch_cp = [cp * np.mean(ps) for cp in analytical_cp]
            label = "Johlas (Pitch)"
            color = "g"
            add_reference_lines(axes, ctp_vals, analytical_an, analytical_ct, pitch_cp, label, color)

    if fill_between:
        mean_of_ct_vals = df.groupby('CT_prime')[CT_key].mean()
        std_of_ct_vals = df.groupby('CT_prime')[CT_key].std()
        ax1.fill_between(mean_of_ct_vals.index, mean_of_ct_vals + std_of_ct_vals, mean_of_ct_vals - std_of_ct_vals, color='grey', alpha=0.2)
        mean_of_cp_vals = df.groupby('CT_prime')[Cp_key].mean()
        std_of_cp_vals = df.groupby('CT_prime')[Cp_key].std()
        ax3.fill_between(mean_of_cp_vals.index, mean_of_cp_vals + std_of_cp_vals, mean_of_cp_vals - std_of_cp_vals, color='grey', alpha=0.2)


    fig.suptitle(title)
    ax0.set(xlabel = "", ylabel=ax_label_type + ' $C_T$')
    ax1.set(xlabel = "", ylabel = "")
    ax2.set(ylabel= ax_label_type + ' $C_p$', xlabel= ax_label_type + ' $a_n$')
    ax3.set(xlabel= '$C_T\'$', ylabel = "")

    leg = ax0.legend()
    bb = leg.get_bbox_to_anchor().transformed(ax0.transAxes.inverted())
    xOffset = 1.7
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg = leg.set_bbox_to_anchor(bb, transform = ax0.transAxes)
    return fig, axes

# %% [markdown]
# # Plot All Data

# %%
fig, ax = four_plot_layout(df, "mean_CT", "mean_an", "mean_Cp", color ="grey", title = "Simulations Means", ax_label_type = "Mean", classical = True);

# %% [markdown]
# # Plot Data by Movement

# %%
palette = "viridis_r"
alpha = 0.7

# %%
fig, axes = four_plot_layout(df, "mean_CT", "mean_an", "mean_Cp", color ="grey", title = "Simulations Means by Movement Type", ax_label_type = "Mean", hue = "Movement", style = "Movement", palette = palette, alpha = alpha, fill_between= True, classical = True);

# %% [markdown]
# ## Plot Data by Resolution

# %%
four_plot_layout(get_stationary(df), "mean_CT", "mean_an", "mean_Cp", title =  "Stationary Simulations Means by Resolution $h$\nwhere $h = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$", ax_label_type = "Mean", hue = "h", style = "turbulence", palette = palette, alpha = alpha, fill_between=True, classical = True );

# %%
four_plot_layout(get_surging(df), "mean_CT", "mean_an", "mean_Cp", title =  "Surging Simulations Means by Resolution $h$\nwhere $h = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$", ax_label_type = "Mean", hue = "h", style = "turbulence", palette = palette, alpha = alpha, fill_between=True, surge = True);

# %%
four_plot_layout(get_pitching(df), "mean_CT", "mean_an", "mean_Cp", title =  "Pitching Simulations Means by Resolution $h$\nwhere $h = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$", ax_label_type = "Mean", hue = "h", style = "turbulence", palette = palette, alpha = alpha, fill_between=True, pitch = True);

# %% [markdown]
# At this point, I feel it is safe to drop the two coarsest resolutions, as they fall outside of the standard deviation of the means of $C_p$ and $C_T$. Thus we only keep resolutions that are $h < 0.25$.

# %%
high_res_df = df[df["h"] < 0.25]
four_plot_layout(high_res_df, "mean_CT", "mean_an", "mean_Cp", title = "Simulations Means by Resolution (ny)", ax_label_type = "Mean", hue = "Resolution (ny)", style = "Movement", alpha = alpha, palette = palette, classical = True, surge=True, pitch=True);

# %%
high_res_df

# %% [markdown]
# # Plot Data by Filter Factor

# %%
analytical_an = [analytical_a(ctp) for ctp in [1.0, 4.0]]
analytical_cp = [a_to_Cp(a) for a in analytical_an]

def johlas(cp, sf, sa, pa):
    v = (2 * sf * sa) / (1 - np.cos(sf * np.pi))
    x_disp = (1 + (3 * (v**2)) / 2)
    theta_disp = np.mean([np.cos(pa * np.pi / 180 * np.sin(x))**3 for x in np.linspace(0, 2 * np.pi, 250)])
    return cp * x_disp * theta_disp
surge_cp = [johlas(cp, 1.0, 0.5, 0.0) for cp in analytical_cp]
pitch_cp = [johlas(cp, 1.0, 0.0, 5.0) for cp in analytical_cp]


# %%
high_res_df

# %%
df_les_vars = high_res_df
df_les_vars = df_les_vars[df_les_vars["Movement"] != "Stationary"]
df_les_vars = df_les_vars[(df_les_vars["CT_prime"] < 6) & (df_les_vars["h"] > 0.08)] # want CT' < 4 and 256 x 128 x 128 resolution data
df_les_vars = df_les_vars[(df_les_vars["filterFactor"] < 4) & (df_les_vars["filterFactor"] > 0.6)]
df_les_vars = df_les_vars[(df_les_vars["turbulence"] != True) | (df["useCorrection"] != True)]
i = df_les_vars[((df_les_vars.turbulence) & ( df_les_vars.useCorrection))].index
df_les_vars.drop(i)

style = df_les_vars[['useCorrection', 'turbulence']].apply(
    lambda row: f"{row.useCorrection}, {row.turbulence}", axis=1)
style.name = 'Correction, Turbulence'
hue = df_les_vars["CT_prime"]
hue.name = "$C_T^'$"

palette =['tab:blue', 'tab:orange']
style_order = ["False, False", "True, False", "False, True"]

g = sns.FacetGrid(df_les_vars, col = "Movement")
surge_ax, pitch_ax = g.axes.flat
surge_ax.axhline(y = surge_cp[0], color = palette[0], label = "Johlas")
surge_ax.axhline(y = surge_cp[1], color = palette[1])
pitch_ax.axhline(y = pitch_cp[0], color = palette[0])
pitch_ax.axhline(y = pitch_cp[1], color = palette[1])

surge_ax.axhline(y = analytical_cp[0], color = palette[0], linestyle=':', label = "Classical")
surge_ax.axhline(y = analytical_cp[1], color = palette[1], linestyle=':')
pitch_ax.axhline(y = analytical_cp[0], color = palette[0], linestyle=':', label = "Classical")
pitch_ax.axhline(y = analytical_cp[1], color = palette[1], linestyle=':')
g.map_dataframe(sns.scatterplot, x = "filterFactor", y = "mean_Cp", hue = hue, style = style, style_order = style_order, palette = palette)
g.set_ylabels("Mean $C_p$")

g.add_legend(title = "LES Parameters")

# %%
pitch_data = df_les_vars[df_les_vars["Movement"] == "Pitching"]
pitch_data["percent_diff_classical"] = pitch_data.apply((lambda row: 100 * (row.mean_Cp - (analytical_cp[0] if row.CT_prime == 1 else analytical_cp[1])) / (analytical_cp[0] if row.CT_prime == 1 else analytical_cp[1])), axis = 1)

fig, ax = plt.subplots(1, 1)
fig.suptitle("Pitching $C_p$ % Difference from Classical Momentum Theory: $C_T^\' = 1.0$")
ax.set_xlabel('Filter Factor')
ax.set_ylabel('% Difference in $C_P$')
sns.scatterplot(data = pitch_data, x = "filterFactor", y = "percent_diff_classical", hue = hue, style = style, style_order = style_order, palette = palette)

filters = np.linspace(0.9, 2.6, num = 100)
sns.lineplot(x = filters, y = 100 * (pitch_cp[0] - analytical_cp[0]) / analytical_cp[0], color = palette[0], label = "Johlas")
sns.lineplot(x = filters, y = 100 * (pitch_cp[1] - analytical_cp[1]) / analytical_cp[1], color = palette[1])

sns.lineplot(x = filters, y = 0, color = palette[0], label = "Classical", linestyle=':')
sns.lineplot(x = filters, y = 0, color = palette[1], linestyle=':')

leg = ax.legend()
bb = leg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
xOffset = 0.45
bb.x0 += xOffset
bb.x1 += xOffset
leg = leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

# %%
pitch_data = pitch_data.sort_values(by=['CT_prime', 'filterFactor'])
pitch_data[(pitch_data["useCorrection"] == True) & (pitch_data["turbulence"] == False)][["nx", "ny", "filter", "filterFactor", "useCorrection", "CT_prime", "mean_Cp"]]

# %%
surge_data = df_les_vars[df_les_vars["Movement"] == "Surging"]
surge_data["percent_diff_classical"] = surge_data.apply((lambda row: 100 * (row.mean_Cp - (analytical_cp[0] if row.CT_prime == 1 else analytical_cp[1])) / (analytical_cp[0] if row.CT_prime == 1 else analytical_cp[1])), axis = 1)
surge_data = surge_data.sort_values(by=['CT_prime', 'filterFactor'])

# %%
surge_data[(surge_data["useCorrection"] == True) & (surge_data["turbulence"] == False)][["nx", "ny", "filter", "filterFactor", "useCorrection", "CT_prime", "mean_Cp"]]

# %%
aspect = 1.5
g = sns.FacetGrid(df_les_vars, row = "Movement", sharey = False, aspect = aspect)
g.map_dataframe(sns.scatterplot, x = "filterFactor", y = "std_Cp", hue = "CT_prime", style = style, style_order = style_order, palette = palette)
g.set_ylabels("Standard Deviation $C_p$")
g.add_legend(title = "LES Parameters")

# %%
g = sns.FacetGrid(df_les_vars, row = "Movement", sharey = False, aspect = aspect)
g.map_dataframe(sns.scatterplot, x = "filterFactor", y = "skewness_Cp", hue = "CT_prime", style = style, style_order = style_order, palette = palette)
g.set_ylabels("Skew $C_p$")
g.add_legend(title = "LES Parameters")

# %%
g = sns.FacetGrid(df_les_vars, row = "Movement", sharey = False, aspect = aspect)
g.map_dataframe(sns.scatterplot, x = "filterFactor", y = "kurtosis_Cp", hue = "CT_prime", style = style, style_order = style_order, palette = palette)
g.set_ylabels("Kurtosis $C_p$")
g.add_legend(title = "LES Parameters")

# %%
high_res_df

# %%
four_plot_layout(get_pitching(high_res_df), "mean_CT", "mean_an", "mean_Cp", title =  "Pitching Simulations Means by Filter Factor", ax_label_type = "Mean", hue = "f", style = "useCorrection", palette = palette, alpha = alpha, pitch = True);

# %%
four_plot_layout(get_pitching(high_res_df), "std_CT", "std_an", "std_Cp", title =  "Pitching Simulations Standard Deviation by Filter Factor", ax_label_type = "STD", hue = "f", style = "useCorrection", palette = palette, alpha = alpha);

# %%
four_plot_layout(get_pitching(high_res_df), "std_CT", "std_an", "std_Cp", title =  "Pitching Simulations Standard Deviation by Turbulence", ax_label_type = "STD", hue = "turbulence", style = "useCorrection", palette = palette, alpha = alpha);

# %%
four_plot_layout(get_stationary(high_res_df), "mean_CT", "mean_an", "mean_Cp", title =  "Stationary Simulations Means by Filter Factor", ax_label_type = "Mean", hue = "f", style = "useCorrection", palette = palette, alpha = alpha, classical = True);

# %%
four_plot_layout(get_surging(high_res_df), "mean_CT", "mean_an", "mean_Cp", title =  "Surging Simulations Means by Filter Factor", ax_label_type = "Mean", hue = "f", style = "useCorrection", palette = palette, alpha = alpha, surge = True);

# %%
four_plot_layout(get_surging(high_res_df), "std_CT", "std_an", "std_Cp", title =  "Surging Simulations Standard Deviation by Filter Factor", ax_label_type = "STD", hue = "f", style = "useCorrection", palette = palette, alpha = alpha);

# %%
four_plot_layout(df, "mean_CT", "mean_an", "mean_Cp",  title = "Simulations Means by use of Turbulence", ax_label_type = "Mean", hue = "turbulence", palette = palette, alpha = alpha);

# %%
four_plot_layout(df, "std_CT", "std_an", "std_Cp",  title = "Simulations Standard Deviations by use of Turbulence", ax_label_type = "STD", hue = "turbulence", palette = palette, alpha = alpha);

# %%
g = sns.FacetGrid(df, col = "Movement")
g.map_dataframe(sns.scatterplot, x = "CT_prime", y = "mean_Cp", hue = "filterFactor", palette = palette, alpha = alpha)
g.add_legend(title = "filterFactor")

# %% [markdown]
# This is pretty hard to interpret, so I want to try another formulation.

# %% [markdown]
# ## Mean ($C_T' = 4.0$)

# %%
import seaborn as sns
cm = sns.color_palette("vlag", as_cmap=True)
all_indices = ["Movement", "Resolution (ny)", "filterFactor", "CT_prime", "useCorrection", "turbulence"]
movement_indices = ["Resolution (ny)", "filterFactor", "CT_prime", "useCorrection", "turbulence"]

# %%
df = df[(df["nx"] > 128) & (df["CT_prime"] < 6)]
df = df[(df["CT_prime"] == 4)]
df_ff1_cf = df[(df['filterFactor'] == 1.0) & (df['useCorrection'] == False)]
df_ff2_cf = df[(df['filterFactor'] == 2.5) & (df['useCorrection'] == False)]
df_ff3_ct = df[(df['filterFactor'] == 1.5) & (df['useCorrection'] == True)]
df = pd.concat([df_ff1_cf, df_ff2_cf, df_ff3_ct], axis = 0)

# %%
indexed_mean_df = mean_df.set_index(all_indices)
indexed_mean_df

# %%
stationary_df = mean_df[mean_df["Movement"] == "Stationary"]
stationary_df = stationary_df.set_index(movement_indices)
stationary_df.style.background_gradient(cmap=cm, subset = ["mean_Cp"])

# %%
surging_df = mean_df[mean_df["Movement"] == "Surging"]
surging_df = surging_df.set_index(movement_indices)
surging_df.style.background_gradient(cmap=cm, subset = ["mean_Cp"])

# %%
pitching_df = mean_df[mean_df["Movement"] == "Pitching"]
pitching_df = pitching_df.set_index(movement_indices)
pitching_df.style.background_gradient(cmap=cm, subset = ["mean_Cp"])

# %% [markdown]
# ## STD ($C_T' = 4.0$)

# %%
indexed_std_df = std_df.set_index(all_indices)
indexed_std_df.style.background_gradient(cmap=cm, subset = ["std_Cp"])

# %%
stationary_df = std_df[std_df["Movement"] == "Stationary"]
stationary_df = stationary_df.set_index(movement_indices)
stationary_df.style.background_gradient(cmap=cm, subset = ["std_Cp"])

# %%
surging_df = std_df[std_df["Movement"] == "Surging"]
surging_df = surging_df.set_index(movement_indices)
surging_df.style.background_gradient(cmap=cm, subset = ["std_Cp"])

# %%
pitching_df = std_df[std_df["Movement"] == "Pitching"]
pitching_df = pitching_df.set_index(movement_indices)
pitching_df.style.background_gradient(cmap=cm, subset = ["std_Cp"])

# %% [markdown]
# ## Skew ($C_T' = 4.0$)

# %%
indexed_skew_df = skew_df.set_index(all_indices)
indexed_skew_df.style.background_gradient(cmap=cm, subset = ["skewness_Cp"])

# %%
surging_df = skew_df[skew_df["Movement"] == "Surging"]
surging_df = surging_df.set_index(movement_indices)
surging_df.style.background_gradient(cmap=cm, subset = ["skewness_Cp"])

# %%
pitching_df = skew_df[skew_df["Movement"] == "Pitching"]
pitching_df = pitching_df.set_index(movement_indices)
pitching_df.style.background_gradient(cmap=cm, subset = ["skewness_Cp"])

# %% [markdown]
# ## Kurtosis ($C_T' = 4.0$)

# %%
indexed_kurtosis_df = kurtosis_df.set_index(all_indices)
indexed_kurtosis_df.style.background_gradient(cmap=cm, subset = ["kurtosis_Cp"])

# %%
surging_df = kurtosis_df[kurtosis_df["Movement"] == "Surging"]
surging_df = surging_df.set_index(movement_indices)
surging_df.style.background_gradient(cmap=cm, subset = ["kurtosis_Cp"])

# %%
pitching_df = kurtosis_df[kurtosis_df["Movement"] == "Pitching"]
pitching_df = pitching_df.set_index(movement_indices)
pitching_df.style.background_gradient(cmap=cm, subset = ["kurtosis_Cp"])
