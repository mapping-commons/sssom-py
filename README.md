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
    <a href="https://doi.org/10.5281/zenodo.14296666"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.14296666.svg" alt="DOI"></a>
</p>

<img src="https://github.com/tis-lab/closed-illustrations/raw/master/logos/sssom-logos/sssom_logo_black_banner.png" />

A Python library and command line interface (CLI) for working with
[SSSOM (Simple Standard for Sharing Ontology Mappings)](https://github.com/mapping-commons/sssom).

## Getting Started

A SSSOM TSV can be parsed with

```python
import sssom

# other SSSOM files can be found on https://mapping-commons.github.io
url = "https://raw.githubusercontent.com/mapping-commons/mh_mapping_initiative/master/mappings/mp_hp_eye_impc.sssom.tsv"

# TSV can be parsed into a mapping set dataframe object,
# which includes a pandas DataFrame, a curies.Converter,
# and metadata
msdf = sssom.parse_tsv(url)

# SSSOM comes with several "write" functions
sssom.write_tsv(msdf, "test.tsv")
sssom.write_json(msdf, "test.json")
sssom.write_owl(msdf, "test.owl")
sssom.write_rdf(msdf, "test.ttl")
```

> [!WARNING]  
> The export formats (json, rdf) of sssom-py are not yet finalised! Expect changes in future releases.

## Documentation

See [documentation](https://mapping-commons.github.io/sssom-py/index.html#)

### Deploy documentation

```console
$ make sphinx
$ make deploy-docs
```

## Schema

See the [schema/](schema) folder for source schema in YAML, plus
derivations to JSON-Schema, ShEx, etc. 

## Testing

`tox` is similar to `make`, but specific for Python software projects. Its
configuration is stored in [`tox.ini`](tox.ini) in different "environments"
whose headers look like `[testenv:...]`. All tests can be run with:

```console
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

## Citation

SSSOM itself can be cited with:

```bibtex
@article{10.1093/database/baac035,
    author = {Matentzoglu, Nicolas and Balhoff, James P and Bello, Susan M and Bizon, Chris and Brush, Matthew and Callahan, Tiffany J and Chute, Christopher G and Duncan, William D and Evelo, Chris T and Gabriel, Davera and Graybeal, John and Gray, Alasdair and Gyori, Benjamin M and Haendel, Melissa and Harmse, Henriette and Harris, Nomi L and Harrow, Ian and Hegde, Harshad B and Hoyt, Amelia L and Hoyt, Charles T and Jiao, Dazhi and Jiménez-Ruiz, Ernesto and Jupp, Simon and Kim, Hyeongsik and Koehler, Sebastian and Liener, Thomas and Long, Qinqin and Malone, James and McLaughlin, James A and McMurry, Julie A and Moxon, Sierra and Munoz-Torres, Monica C and Osumi-Sutherland, David and Overton, James A and Peters, Bjoern and Putman, Tim and Queralt-Rosinach, Núria and Shefchek, Kent and Solbrig, Harold and Thessen, Anne and Tudorache, Tania and Vasilevsky, Nicole and Wagner, Alex H and Mungall, Christopher J},
    title = {A Simple Standard for Sharing Ontological Mappings (SSSOM)},
    journal = {Database},
    volume = {2022},
    pages = {baac035},
    year = {2022},
    month = {05},
    issn = {1758-0463},
    doi = {10.1093/database/baac035},
    url = {https://doi.org/10.1093/database/baac035},
    eprint = {https://academic.oup.com/database/article-pdf/doi/10.1093/database/baac035/43832024/baac035.pdf},
}
```

To cite the SSSOM-py software package specifically, use:

```bibtex
@software{sssom-py,
  author       = {Harshad Hegde and
                  Nico Matentzoglu and
                  Charles Tapley Hoyt and
                  Chris Mungall and
                  Joe Flack and
                  Benjamin M. Gyori and
                  Damien Goutte-Gattat and
                  Glass and
                  Syphax Bouazzouni},
  title        = {mapping-commons/sssom-py: v0.4.15 release (minor
                   fixes)
                  },
  month        = dec,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.15},
  doi          = {10.5281/zenodo.14296666},
  url          = {https://doi.org/10.5281/zenodo.14296666},
}
```
 
