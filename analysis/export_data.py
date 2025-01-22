import padeopsIO as pio
import argparse
import os

DATA_PATH = os.environ['SCRATCH'] + "/Data/"


parser = argparse.ArgumentParser()
parser.add_argument("write_dir")
parser.add_argument("filename") 
args = parser.parse_args()

# TODO - Make this command line runnable
# https://docs.python.org/3/howto/argparse.html
# Then I could add an alias that zips and downloads from command line
pio.write_npz(write_dir = DATA_PATH + args.write_dir, filename = args.filename)