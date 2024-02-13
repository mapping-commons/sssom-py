.. _sssom_cli:

SSSOM Command Line Interface
===========================

.. _cli_usage:

Usage
-----

To use SSSOM's command-line interface (CLI), install SSSOM using pip:

.. code-block:: bash

  pip install sssom

Once SSSOM is installed, you can run the CLI by typing `sssom` followed by the desired subcommand. For example, to convert a file from one format to another, you would use the `convert` subcommand:

.. code-block:: bash

  sssom convert input.tsv output.sssom.tsv

For a complete list of subcommands and options, see the `--help` option:

.. code-block:: bash

  sssom --help

.. _cli_subcommands:

Subcommands
-----------

The following subcommands are available:

* `convert`: Convert a file from one format to another.
* `help`: Echo help for subcommands.
* `parse`: Parse a file in one of the supported formats (such as obographs) into an SSSOM TSV file.
* `validate`: Produce an error report for an SSSOM file.
* `split`: Split input file into multiple output broken down by prefixes.
* `ptable`: Convert an SSSOM file to a ptable for `kboom/boomer <https://github.com/INCATools/boomer>`_.
* `dedupe`: Remove lower confidence duplicate lines from an SSSOM file.
* `dosql`: Run a SQL query over one or more SSSOM files.
* `sparql`: Run a SPARQL query.
* `diff`: Compare two SSSOM files.
* `partition`: Partition an SSSOM into one file for each strongly connected component.
* `cliquesummary`: Calculate summaries for each clique in a SSSOM file.
* `crosstab`: Write sssom summary cross-tabulated by categories.
* `correlations`: Calculate correlations.
* `merge`: Merge multiple MappingSetDataFrames into one.
* `rewire`: Rewire an ontology using equivalent classes/properties from a mapping file.
* `reconcile_prefixes`: Reconcile prefix_map based on provided YAML file.
* `sort`: Sort DataFrame columns canonically.
* `filter`: Filter a dataframe by dynamically generating queries based on user input.
* `annotate`: Annotate metadata of a mapping set.
* `remove`: Remove mappings from an input mapping.
* `invert`: Invert subject and object IDs such that all subjects have the prefix provided.

.. _cli_options:

Options
-------

The following options are common across all subcommands:

* `-v`, `--verbose`: Increase verbosity.
* `-q`, `--quiet`: Silence all output except for errors.
* `--version`: Print the version number and exit.

In addition to these common options, each subcommand has its own set of options. For more information, see the help for each subcommand.

.. _cli_examples:

Examples
--------

The following are some examples of how to use the SSSOM CLI:

* To convert a file from TSV to RDF/XML:

.. code-block:: bash

  sssom convert input.tsv output.owl --output-format rdfxml

* To parse an OBO file into an SSSOM TSV file:

.. code-block:: bash

  sssom parse input.obo output.sssom.tsv

* To validate an SSSOM file:

.. code-block:: bash

  sssom validate input.sssom.tsv

* To split an SSSOM file into multiple files, one for each prefix:

.. code-block:: bash

  sssom split input.sssom.tsv output_directory

* To convert an SSSOM file to a ptable for `kboom/boomer <https://github.com/INCATools/boomer>`_:

.. code-block:: bash

  sssom ptable input.sssom.tsv output.ptable

* To remove lower confidence duplicate lines from an SSSOM file:

.. code-block:: bash

  sssom dedupe input.sssom.tsv output.sssom.tsv

* To run a SQL query over one or more SSSOM files:

.. code-block:: bash

  sssom dosql -Q "SELECT * FROM df1 WHERE confidence>0.5 ORDER BY confidence" file1.sssom.tsv file2.sssom.tsv

* To run a SPARQL query:

.. code-block:: bash

  sssom sparql -u http://example.org/sparql -g my_graph -q "SELECT * WHERE { ?s ?p ?o }"

* To compare two SSSOM files:

.. code-block:: bash

  sssom diff file1.sssom.tsv file2.sssom.tsv output.sssom.tsv

* To partition an SSSOM into one file for each strongly connected component:

.. code-block:: bash

  sssom partition input.sssom.tsv output_directory

* To calculate summaries for each clique in a SSSOM file:

.. code-block:: bash

  sssom cliquesummary input.sssom.tsv output.tsv

* To write sssom summary cross-tabulated by categories:

.. code-block:: bash

  sssom crosstab input.sssom.tsv output.tsv

* To calculate correlations:

.. code-block:: bash

  sssom correlations input.sssom.tsv output.tsv

* To merge multiple MappingSetDataFrames into one:

.. code-block:: bash

  sssom merge input1.sssom.tsv input2.sssom.tsv output.sssom.tsv

* To rewire an ontology using equivalent classes/properties from a mapping file:

.. code-block:: bash

  sssom rewire -i input.owl -m mapping.tsv -o output.owl

* To reconcile prefix_map based on provided YAML file:

.. code-block:: bash

  sssom reconcile_prefixes input.sssom.tsv reconcile_prefix_file.yaml output.sssom.tsv

* To sort DataFrame columns canonically:

.. code-block:: bash

  sssom sort input.sssom.tsv output.sssom.tsv

* To filter a dataframe by dynamically generating queries based on user input:

.. code-block:: bash

  sssom filter input.sssom.tsv output.sssom.tsv --subject_id x:% --subject_id y:% --object_id y:% --object_id z:%

* To annotate metadata of a mapping set:

.. code-block:: bash

  sssom annotate input.sssom.tsv output.sssom.tsv --mapping_set_id http://example.org/abcd

* To remove mappings from an input mapping:

.. code-block:: bash

  sssom remove input.sssom.tsv output.sssom.tsv --remove-map remove_map.sssom.tsv

* To invert subject and object IDs such that all subjects have the prefix provided:

.. code-block:: bash

  sssom invert input.sssom.tsv output.sssom.tsv --subject-prefix x:

.. _cli_contributing:

Contributing
------------

Contributions to SSSOM are welcome! To contribute, please follow these steps:

1. Fork the SSSOM repository on GitHub.
2. Create a new branch for your changes.
3. Make your changes.
4. Add tests for your changes.
5. Push your changes to your fork.
6. Create a pull request from your fork to the upstream SSSOM repository.

.. _cli_additional_info:

Additional Information
----------------------

For more information, please see the SSSOM documentation: https://sssom.readthedocs.io/.
