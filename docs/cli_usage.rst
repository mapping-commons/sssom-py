CLI
===

The `sssom` script is a wrapper for multiple sub-commands

The main one is `convert`

.. code-block:: bash

    sssom convert -i tests/data/basic.tsv -o basic.ttl
    
.. click:: sssom.cli:main
    :prog: sssom
    :nested: full
