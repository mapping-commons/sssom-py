## Serialization Functions for SSSOM

This module contains functions for serializing SSSOM mapping set dataframes to various formats, including TSV, OWL, JSON, FHIR JSON, and RDF.

### Writers

The following table lists the supported writer functions and their corresponding output formats:

| Writer Function | Output Format | Description |
|-----------------|---------------|-------------|
| `write_table` | TSV | This function writes a `MappingSetDataFrame` object to a file as a table. |
| `write_owl` | OWL | This function writes a `MappingSetDataFrame` object to a file as OWL. |
| `write_ontoportal_json` | Ontoportal JSON | This function writes a `MappingSetDataFrame` object to a file as the ontoportal mapping JSON model. |
| `write_fhir_json` | FHIR JSON | This function writes a `MappingSetDataFrame` object to a file as FHIR ConceptMap JSON. |
| `write_json` | JSON | This function writes a `MappingSetDataFrame` object to a file as JSON. |
| `write_rdf` | RDF | This function writes a `MappingSetDataFrame` object to a file as RDF. |

The `get_writer_function()` function can be used to obtain the appropriate writer function based on the desired output format.

### Converters

The following converter functions are available for converting mapping set dataframes to various formats:

| Converter Function | Output Format | Description |
|--------------------|---------------|-------------|
| `to_owl_graph` | OWL graph | This function converts a `MappingSetDataFrame` object to OWL in an RDF graph. |
| `to_rdf_graph` | RDF graph | This function converts a `MappingSetDataFrame` object to an RDF graph. |
| `to_fhir_json` | FHIR JSON | This function converts a `MappingSetDataFrame` object to a JSON object. |
| `to_json` | JSON | This function converts a `MappingSetDataFrame` object to a JSON object. |
| `to_ontoportal_json` | Ontoportal JSON | This function converts a `MappingSetDataFrame` object to a list of ontoportal mapping JSON objects. |

### Support Methods

The following support methods are available for use with the serialization functions:

| Method | Description |
|--------|-------------|
| `_get_separator` | Returns the appropriate separator character for the specified table format. |
| `_inject_annotation_properties` | Injects annotation properties into an RDF graph. |

### Writing Tables

The `write_tables()` function can be used to write a collection of mapping set dataframes to TSV files.

### Additional Information

- For more information on SSSOM, see the [SSSOM documentation](https://mapping-commons.github.io/sssom/).
- For more information on the SSSOM Python package, see the [SSSOM Python package documentation](https://mapping-commons.github.io/sssom-py/).