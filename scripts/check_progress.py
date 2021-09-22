#!/usr/bin/env python3
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Check progress in a folder")

parser.add_argument("folder", help="folder to check")

parser.add_argument("num_steps", type=int, help="number of full steps")

parser.add_argument("--name", default="", help="name to filter files with")


args = parser.parse_args()


folder=Path(args.folder)

if not folder.exists():
    raise ValueError
    
fil_name = "**/*.gz" if args.name == "" else f"**/*{args.name}*.gz"

for l in folder.glob(fil_name):
    #print(l)
    res = pd.read_csv(l, index_col=0)
    if len(res) < args.num_steps:
        print(l.stem+l.suffix)
        print(len(res))
