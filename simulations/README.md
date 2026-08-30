# Simulations Guide

This documentation contains explanations of each of the simulations "created" through the files in this folder. The `.py` files define the parameters for a given simulation, and create the needed input files for each simulation using the `write_padeops_files` function in `jinja_sim_utils.py`.

## Floating Simulations

Each of the floating turbine simulation files will start with an `F`. After this, it will be followed by an ID `XXXX` for the number of the simulation. There will then be some combination of `PI` (pitch), `SU` (surge), and `SW` (sway) symbols. If it isn't moving, the key will be `X`.

Here is a list of the simulations with the ID, as well as the keys. I have also noted which key are varied / the purpose of the simulation. 

| ID   | Keys | Notes | Date |
|------|------|-------|------|
| 0000 | X    | Varying dt and CT' to test needed timestep | 01/21/25|
| 0000 | SU_PI_X | Varying grid size needed for accurate resolution | 01/22/25|
| 0001 | X    | Varying CT' with constant dt to test instability | 01/23/25|
| 0002 | SU_PI_X | | 02/17/25 |

## Fixed-Bottom Simulations

Each of the fixed bottom turbine simulation files will start with an `B`. After this, it will be followed by an ID `XXXX` for the number of the simulation.

| ID   | Notes | Date |
|------|-------|------|
| 0000 | Single cT and dt to check instability without moving turbine | 01/23/25|
| 0001 | Sweep of static yaw and tilt to check tiled UMM derivation | 06/23/25|

Oops I failed to use this - better late than never! 

## Quals
F_0029 is the final runs from quals! Many of the ones right before that are sensitivity testing!

## Post-Quals but still Surge
F_0031 something with phase budgets...
F_0030 something with phase budgets... 
-----
#### scratch error deleted these files that made these and analysis scrips... will need to re-make if needed
F_0032 tests higher frequencies f = 4 and f = 8 -> plataue from quals f = 2
F_0033 tests what happens when you swap the order of flow update and turbine movement -> no effeect
F_0034 tests the ability to use time series for surge and CT' in ADM5 -> looks good!
F_0035 tests the ability to switch between using CT and CT' in ADM5
----
F_0036 re-runs quals data but with CT' = 1
F_0037 runs OpenFAST data in a 2x2 simulation matrix with mean/timeseries CT' and sinusoidal/time series surge -> uses OpenFAST data where: HWindSpeed,WaveHs,WaveTp = 6.0,5.0,8.0
F_0038 runs OpenFAST data in a 2x2 simulation matrix with mean/timeseries CT' and sinusoidal/time series surge -> uses OpenFAST data where: HWindSpeed,WaveHs,WaveTp = 10.0,5.0,8.0

