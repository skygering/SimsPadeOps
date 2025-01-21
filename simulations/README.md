# Simulations Guide

This documentation contains explanations of each of the simulations "created" through the files in this folder. The `.py` files define the parameters for a given simulation, and create the needed input files for each simulation using the `write_padeops_files` function in `jinja_sim_utils.py`.

## Floating Simulations

Each of the floating turbine simulation files will start with an `F`. After this, it will be followed by an ID `XXXX` for the number of the simulation. There will then be some combination of `PI` (pitch), `SU` (surge), and `SW` (sway) symbols. If it isn't moving, the key will be `X`.

Here is a list of the simulations with the ID, as well as the keys. I have also noted which key are varied / the purpose of the simulation. 

| ID   | Keys | Notes | Date |
|------|------|-------|------|
| 0000 | X    | Varying dt and CT' to test needed timestep | 01/21/25|




