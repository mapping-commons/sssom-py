"""Constants."""

import os
import pathlib

CWD = pathlib.Path(__file__).parent.resolve()
SCHEMA_YAML = os.path.join(CWD, "sssom.yaml")
