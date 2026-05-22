import numpy as np
import math

rho, uinf, D, dt = 1, 1, 1

# calculate needed values
def calc_an(df, ud_key = "UDisk_Turb", uinf_key = "UInf_Turb"): # calcualte an with respect to relative-inflow
    u_inf = df[uinf_key]* np.cos(np.deg2rad(df["Tilt"]))
    return 1 - (df[ud_key] / u_inf)

def calc_ct(df, ud_key = "UDisk_Turb", uinf_key = "UInf_Ground"): # calcualte CT with respect to freestream inflow
    u_inf = df[uinf_key]
    return np.sign(df[ud_key]) * df["Ct_prime"] * ((df[ud_key])**2 / (u_inf)**2)

def calc_cp(df, uinf_key = "UInf_Ground"): # calcualte CT with respect to freestream inflow
    u_inf = df[uinf_key]
    return df.Power / (0.5 * rho * math.pi * (D/2)**2 * (u_inf)**3)