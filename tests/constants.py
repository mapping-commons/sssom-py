"""Constants for test cases."""

import os

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, "data")

test_out_dir = os.path.join(cwd, "tmp")
os.makedirs(test_out_dir, exist_ok=True)
