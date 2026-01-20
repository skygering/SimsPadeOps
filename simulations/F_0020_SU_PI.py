# Here I am trying to descern what are reasonable values for joint surging/pitching runs

import numpy as np

theta0 = [4, 8, 12, 16, 20]
fs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
R = 150 / 240  # IEA 15MW has 150m hub height and a 240m diameter
t = np.linspace(0, 4 / fs, num = 100)

v0 = []

for t0 in theta0:
    for f in fs:
        omega = 2 * np.pi * f
        t0r = np.deg2rad(t0)
        theta = t0r * np.sin(omega * t)
        theta_dot = t0r * omega * np.cos(omega * f)
        z = R * np.cos(theta)
        vx = np.max(np.abs(theta_dot * z))
        v0.append((t0, f, vx))

vs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
all_runs = []
for (t, f, v) in v0:
    extra_vs = vs[vs > v]
    all_vs = np.append(extra_vs, v)
    all_vs = np.sort(all_vs)
    for val in all_vs:
        all_runs.append((t, f, val)) # maybe I don't need all of them to begin with...

print(all_runs) # ~100 runs

