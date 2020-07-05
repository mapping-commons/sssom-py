# Python Utilities for SSSOM

See https://github.com/OBOFoundry/SSSOM

## Install

```bash
pip install .
```

## Command Line

The `sssom` script is a wrapper for multiple sub-commands

The main one is `convert`

```bash
sssom convert -i tests/data/basic.tsv -o basic.ttl
```

## Python Classes

We define a schema (biolinkml) defined using the SSSOM source TSV: [schema/](schema/)

This is used to autogenerate the python datamodel: [sssom/](sssom/)


TODO docs


