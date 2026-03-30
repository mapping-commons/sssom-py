# Installation

## Requirements

sssom-py requires **Python 3.10** or later.

## Install from PyPI

```bash
pip install sssom
```

### Optional extras

Some commands require optional dependencies. Install them as needed:

```bash
# For the `partition` and `cliquesummary` commands
pip install sssom[networkx]

# For the `dosql` and `filter` commands
pip install sssom[pansql]

# For the `correlations` command
pip install sssom[scipy]

# For the `serve-rdf` command (SPARQL endpoint)
pip install sssom[rdflib-endpoint]

# Install everything
pip install sssom[networkx,pansql,scipy,rdflib-endpoint]
```

## Install for development

Clone the repository and install with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/mapping-commons/sssom-py
cd sssom-py
poetry install --all-extras
```

## Verify installation

```bash
sssom --version
```

You should see the installed version number printed to the terminal.
