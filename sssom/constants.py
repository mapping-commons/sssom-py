"""Constants."""

import os
import pathlib

HERE = pathlib.Path(__file__).parent.resolve()
SCHEMA_YAML = os.path.join(HERE, "sssom.yaml")
PREFIX_RECON_YAML = os.path.join(HERE, "prefix_reconciliation.yaml")
