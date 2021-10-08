"""Constants for test cases."""

import pathlib

cwd = pathlib.Path(__file__).parent.resolve()
data_dir = cwd / "data"

test_out_dir = cwd / "tmp"
test_out_dir.mkdir(parents=True, exist_ok=True)
