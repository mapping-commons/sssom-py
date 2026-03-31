# CLI Reference

sssom-py provides the `sssom` command line tool with commands for all common mapping operations.

## Global options

```bash
sssom [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Increase verbosity (use `-vv` for debug). |
| `-q`, `--quiet` | Suppress all output except errors. |
| `--version` | Show the version and exit. |

## Commands at a glance

| Command | Description |
|---------|-------------|
| [`convert`](commands.md#sssom-convert) | Convert a file between formats (TSV, OWL, RDF, JSON). |
| [`parse`](commands.md#sssom-parse) | Parse a file in any supported format into SSSOM TSV. |
| [`validate`](commands.md#sssom-validate) | Produce an error report for an SSSOM file. |
| [`diff`](commands.md#sssom-diff) | Compare two SSSOM files. |
| [`merge`](commands.md#sssom-merge) | Merge multiple SSSOM files into one. |
| [`filter`](commands.md#sssom-filter) | Filter mappings by dynamic column queries. |
| [`sort`](commands.md#sssom-sort) | Sort rows and columns canonically. |
| [`split`](commands.md#sssom-split) | Split a file by prefixes. |
| [`dedupe`](commands.md#sssom-dedupe) | Remove lower-confidence duplicate mappings. |
| [`annotate`](commands.md#sssom-annotate) | Annotate mapping set metadata. |
| [`dosql`](commands.md#sssom-dosql) | Run SQL queries over SSSOM files. |
| [`invert`](commands.md#sssom-invert) | Swap subject and object IDs. |
| [`remove`](commands.md#sssom-remove) | Remove mappings present in another file. |
| [`partition`](commands.md#sssom-partition) | Partition into strongly connected components. |
| [`cliquesummary`](commands.md#sssom-cliquesummary) | Summarise each clique in a file. |
| [`sparql`](commands.md#sssom-sparql) | Run a SPARQL query against an endpoint. |
| [`crosstab`](commands.md#sssom-crosstab) | Cross-tabulate mappings by categories. |
| [`correlations`](commands.md#sssom-correlations) | Calculate correlations between mapping categories. |
| [`ptable`](commands.md#sssom-ptable) | Convert to ptable format for boomer. |
| [`rewire`](commands.md#sssom-rewire) | Rewire an ontology using mappings. |
| [`reconcile-prefixes`](commands.md#sssom-reconcile-prefixes) | Reconcile prefixes using a YAML config. |
| [`serve-rdf`](commands.md#sssom-serve-rdf) | Serve mappings as an RDF SPARQL endpoint. |

See the [full command reference](commands.md) for detailed options and examples.
