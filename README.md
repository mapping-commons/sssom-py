# Python Utilities for SSSOM

SSSOM (Simple Standard for Sharing Ontology Mappings) is a TSV and RDF/OWL standard for ontology mappings

See https://github.com/OBOFoundry/SSSOM

This is a python library and command line toolkit for working with SSSOM. It also defines a schema for SSSOM

## Documentation

See https://sssom-py.readthedocs.io/

## Schema

See the [schema/](schema) folder for source schema in YAML, plus
derivations to JSON-Schema, ShEx, etc. 

## Testing

`tox` is similar to `make`, but specific for Python software projects. Its
configuration is stored in [`tox.ini`](tox.ini) in different "environments"
whose headers look like `[testenv:...]`. All tests can be run with:

```shell
$ pip install tox
$ tox
```

A specific environment can be run using the `-e` flag, such as `tox -e lint` to run
the linting environment.
