# Quick Start

## Python library

### Parse an SSSOM TSV file

```python
import sssom

# Parse a local file
msdf = sssom.parse_tsv("my_mappings.sssom.tsv")

# Parse from a URL
url = "https://raw.githubusercontent.com/mapping-commons/mh_mapping_initiative/master/mappings/mp_hp_eye_impc.sssom.tsv"
msdf = sssom.parse_tsv(url)
```

The returned `MappingSetDataFrame` object contains:

- `msdf.df` -- a pandas `DataFrame` with the mapping rows
- `msdf.converter` -- a `curies.Converter` for prefix handling
- `msdf.metadata` -- a dictionary of mapping set metadata

### Write to different formats

```python
# Write back to TSV
sssom.write_tsv(msdf, "output.sssom.tsv")

# Convert to other formats
sssom.write_json(msdf, "output.json")
sssom.write_owl(msdf, "output.owl")
sssom.write_rdf(msdf, "output.ttl")
```

!!! warning
    The export formats (JSON, RDF) are not yet finalised. Expect changes in future releases.

### Inspect mappings

```python
import sssom

msdf = sssom.parse_tsv("my_mappings.sssom.tsv")

# Access the DataFrame
print(f"Number of mappings: {len(msdf.df)}")
print(msdf.df.head())

# Access metadata
print(msdf.metadata)
```

### Compare two mapping sets

```python
import sssom

msdf1 = sssom.parse_tsv("set1.sssom.tsv")
msdf2 = sssom.parse_tsv("set2.sssom.tsv")

diff = sssom.compare_dataframes(msdf1.df, msdf2.df)
print(f"Common mappings: {len(diff.common_tuples)}")
print(f"Unique to set1: {len(diff.unique_tuples1)}")
print(f"Unique to set2: {len(diff.unique_tuples2)}")
```

### Filter redundant mappings

```python
import sssom

msdf = sssom.parse_tsv("my_mappings.sssom.tsv")
filtered_df = sssom.filter_redundant_rows(msdf.df)
print(f"Before: {len(msdf.df)} rows, After: {len(filtered_df)} rows")
```

## Command line

### Parse and convert

```bash
# Parse an SSSOM file (validates and pretty-prints)
sssom parse my_mappings.sssom.tsv

# Convert TSV to OWL
sssom convert my_mappings.sssom.tsv -o output.owl -O owl

# Convert TSV to RDF (Turtle)
sssom convert my_mappings.sssom.tsv -o output.ttl -O rdf
```

### Compare mapping sets

```bash
sssom diff set1.sssom.tsv set2.sssom.tsv -o diff.sssom.tsv
```

### Validate

```bash
sssom validate my_mappings.sssom.tsv
```

### Merge multiple files

```bash
sssom merge set1.sssom.tsv set2.sssom.tsv -o merged.sssom.tsv
```

### Query with SQL

```bash
sssom dosql -Q "SELECT * FROM df1 WHERE confidence > 0.5 ORDER BY confidence" my_mappings.sssom.tsv
```

### Filter mappings

```bash
sssom filter --subject_id HP:% --object_id MP:% my_mappings.sssom.tsv
```

For the full list of CLI commands, see the [CLI Reference](cli/index.md).
