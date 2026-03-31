# Python API

sssom-py exposes a public API through the top-level `sssom` package. The most commonly used functions are re-exported for convenience:

```python
import sssom

# Parsing
msdf = sssom.parse_tsv("mappings.sssom.tsv")
msdf = sssom.parse_csv("mappings.csv")
msdf = sssom.parse_sssom_table("mappings.sssom.tsv")

# Writing
sssom.write_tsv(msdf, "output.sssom.tsv")
sssom.write_json(msdf, "output.json")
sssom.write_owl(msdf, "output.owl")
sssom.write_rdf(msdf, "output.ttl")

# Utilities
diff = sssom.compare_dataframes(df1, df2)
filtered = sssom.filter_redundant_rows(df)
groups = sssom.group_mappings(df)
collapsed = sssom.collapse(df)
```

## Key classes

| Class | Description |
|-------|-------------|
| `MappingSetDataFrame` | Core container holding a pandas DataFrame, a `curies.Converter`, and metadata. |
| `MappingSetDocument` | Document-level representation of a mapping set. |
| `MappingSet` | Schema class representing a set of mappings (from `sssom-schema`). |
| `Mapping` | Schema class representing a single mapping (from `sssom-schema`). |

## Module overview

| Module | Description |
|--------|-------------|
| [`sssom.parsers`](parsers.md) | Parse SSSOM files from TSV, JSON, RDF, OBO Graphs, and more. |
| [`sssom.writers`](writers.md) | Write SSSOM data to various output formats. |
| [`sssom.util`](util.md) | Utility functions for manipulation, merging, filtering. |
| [`sssom.io`](io.md) | High-level I/O operations used by the CLI. |
| [`sssom.validators`](validators.md) | Schema validation for SSSOM files. |
| [`sssom.rdf_util`](rdf_util.md) | RDF-related utilities (rewiring, serialisation). |
| [`sssom.constants`](constants.md) | Constants, enums, and schema helpers. |
