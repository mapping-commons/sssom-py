"""Constants for test cases."""

import os
import pathlib

cwd = pathlib.Path(__file__).parent.resolve()
data_dir = os.path.join(cwd, "data")

test_out_dir = os.path.join(cwd, "tmp")
os.makedirs(test_out_dir, exist_ok=True)
