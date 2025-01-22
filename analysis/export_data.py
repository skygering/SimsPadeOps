import padeopsIO as pio
import argparse
import os
import json
from pathlib import Path

DATA_PATH = os.environ['SCRATCH'] + "/Data/"


parser = argparse.ArgumentParser()
parser.add_argument("write_dir")
parser.add_argument("filename") 
args = parser.parse_args()

# TODO - Make this command line runnable
# https://docs.python.org/3/howto/argparse.html
# Then I could add an alias that zips and downloads from command line
out_dir = Path(DATA_PATH + args.write_dir)

x, y, z = 25, 10, 5
sim = pio.BudgetIO(out_dir, padeops = True, runid = 1)
sim.write_metadata(out_dir, args.filename, "npz", x, y, z)