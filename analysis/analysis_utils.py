import argparse
import os

DATA_PATH = os.environ['SCRATCH'] + "/Data/"


def arg_parser(arg_list = ["write_dir", "filename"]):
    parser = argparse.ArgumentParser()
    for a in arg_list:
        parser.add_argument(a)
    args = parser.parse_args()
    return args