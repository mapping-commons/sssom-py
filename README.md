# Python Utilities for SSSOM

<p align="center">
    <a href="https://github.com/mapping-commons/sssom-py/actions/workflows/qc.yml">
        <img alt="Tests" src="https://github.com/mapping-commons/sssom-py/actions/workflows/qc.yml/badge.svg" />
    </a>
    <a href="https://pypi.org/project/sssom">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/sssom" />
    </a>
    <a href="https://pypi.org/project/sssom">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/sssom" />
    </a>
    <a href="https://github.com/mapping-commons/sssom-py/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/sssom" />
    </a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    </a>
</p>

<img src="https://github.com/tis-lab/closed-illustrations/raw/master/logos/sssom-logos/sssom_logo_black_banner.png" />

SSSOM (Simple Standard for Sharing Ontology Mappings) is a TSV and RDF/OWL standard for ontology mappings

```
WARNING: 
    The export formats (json, rdf) of sssom-py are not yet finalised! 
    Please expect changes in future releases!
```

See https://github.com/OBOFoundry/SSSOM

This is a python library and command line toolkit for working with SSSOM. It also defines a schema for SSSOM.

## Documentation

See [documentation](https://mapping-commons.github.io/sssom-py/index.html#)

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

Outstanding contributors are groups and institutions that have helped with organising the SSSOM
Python package's development, providing funding, advice and infrastructure. We are very grateful
for all your contribution - the project would not exist without you!

### Harvard Medical School

<img width="250" src="https://hms.harvard.edu/themes/harvardmedical/logo.svg" alt="Harvard Medical School Logo" />

The [INDRA Lab](https://indralab.github.io), a part of the
[Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/about/)
and the [Harvard Program in Therapeutic Science (HiTS)](https://hits.harvard.edu), is interested in
natural language processing and large-scale knowledge assembly. Their work on SSSOM is funded by the
DARPA Young Faculty Award W911NF2010255 (PI: Benjamin M. Gyori).

https://indralab.github.io
 
