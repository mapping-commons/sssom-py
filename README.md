# Python Utilities for SSSOM

SSSOM (Simple Standard for Sharing Ontology Mappings) is a TSV and RDF/OWL standard for ontology mappings

See https://github.com/OBOFoundry/SSSOM

This is a python library and command line toolkit for working with SSSOM. It also defines a schema for SSSOM

## Documentation

See 
* [old](https://sssom-py.readthedocs.io/) or
* [new](https://mapping-commons.github.io/sssom-py/index.html#)

### Deploy documentation
```shell
make sphinx
make deploy-docs
```

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

## Outstanding Contributors

Outstanding contributors are groups and institutions that have helped with organising the ODK
development, providing funding, advice and infrastructure. We are very grateful for all your
contribution - the project would not exist without you!

### Harvard Medical School

<img width="250" src="https://hms.harvard.edu/themes/harvardmedical/logo.svg" alt="Harvard Medical School Logo" />

The [INDRA Lab](https://indralab.github.io), a part of the
[Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/about/)
and the [Harvard Program in Therapeutic Science (HiTS)](https://hits.harvard.edu), is interested in
natural language processing and large-scale knowledge assembly. Their work on SSSOM is funded by the
DARPA Young Faculty Award W911NF2010255 (PI: Benjamin M. Gyori).
