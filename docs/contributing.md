# Contributing

Contributions to sssom-py are welcome! Here's how to get started.

## Development setup

```bash
git clone https://github.com/mapping-commons/sssom-py
cd sssom-py
poetry install --all-extras
```

## Running tests

```bash
# Run all tests
poetry run pytest

# Run with tox (includes linting, type checking, etc.)
pip install tox
tox
```

## Code quality

The project uses:

- **black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for static type checking

Run formatting before committing:

```bash
poetry run black src/ tests/
poetry run isort src/ tests/
```

## Building documentation locally

```bash
poetry install --extras docs
poetry run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Project structure

```
sssom-py/
├── src/sssom/          # Main package
│   ├── cli.py          # Click CLI commands
│   ├── parsers.py      # File parsing (TSV, JSON, RDF, OBO Graphs)
│   ├── writers.py      # Output writers (TSV, OWL, RDF, JSON)
│   ├── util.py         # Core utilities (merge, diff, filter, etc.)
│   ├── io.py           # High-level I/O operations
│   ├── validators.py   # Schema validation
│   ├── rdf_util.py     # RDF utilities
│   ├── rdf_internal.py # Internal RDF serialisation
│   ├── constants.py    # Constants and schema helpers
│   ├── context.py      # Prefix map / context handling
│   ├── cliques.py      # Clique detection (requires networkx)
│   ├── sparql_util.py  # SPARQL endpoint queries
│   └── obo.epm.json    # OBO Extended Prefix Map
├── tests/              # Test suite
├── docs/               # Documentation source (mkdocs)
├── mkdocs.yml          # MkDocs configuration
├── pyproject.toml      # Project configuration
└── tox.ini             # Tox test environments
```
