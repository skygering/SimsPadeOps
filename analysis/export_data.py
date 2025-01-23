import padeopsIO as pio
from pathlib import Path
import analysis_utils as au

args = au.arg_parser()
out_dir = Path(au.DATA_PATH + args.write_dir)

x, y, z = 25, 10, 5  # these aren't used
sim = pio.BudgetIO(out_dir, padeops = True, runid = 1)
sim.write_metadata(out_dir, args.filename, "npz", x, y, z)