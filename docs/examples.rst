Examples
========

``parse`` command
------------------

The ``parse`` command is a way to import a mapping file. In this example, the file `basic.tsv <https://github.com/mapping-commons/sssom-py/blob/master/tests/data/basic.tsv>`_
will be parsed. The CLI command is as follows:

.. code-block:: bash
    
    sssom parse basic.tsv

This results in the contents of the file displayed on the terminal.
If the result is needed to be exported into another tsv, an ``--output`` 
parameter could be passed and the command will look like this:

.. code-block:: bash

    sssom parse basic.tsv --output parsed_basic.tsv

``convert`` command
-------------------

The ``convert`` command converts files from one format to another.

.. code-block:: bash


