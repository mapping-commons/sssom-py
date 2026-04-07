#!/usr/bin/env bash

set -e

uv run sssom convert data/basic.tsv -O json -o validate_data/basic.tsv.json
uv run sssom convert data/basic.tsv -O owl -o validate_data/basic.tsv.owl
uv run sssom convert data/basic.tsv -O rdf -o validate_data/basic.tsv.rdf
uv run sssom convert data/basic.tsv -O tsv -o validate_data/basic.tsv.tsv
uv run sssom convert data/cob-to-external.tsv -O json -o validate_data/cob-to-external.tsv.json
uv run sssom convert data/cob-to-external.tsv -O owl -o validate_data/cob-to-external.tsv.owl
uv run sssom convert data/cob-to-external.tsv -O rdf -o validate_data/cob-to-external.tsv.rdf
uv run sssom convert data/cob-to-external.tsv -O tsv -o validate_data/cob-to-external.tsv.tsv
uv run sssom parse data/hp-base.json -I obographs-json -o validate_data/hp-base.json.tsv
uv run sssom parse data/oaei-ordo-hp.rdf -I alignment-api-xml -o validate_data/oaei-ordo-hp.rdf.tsv
