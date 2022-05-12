Quick Install Guide
===================

Install Python
--------------

Get the latest version of Python at https://www.python.org/downloads/ or with your operating system’s package manager.

You can verify that Python is installed by typing ``python`` from your shell; you should see something like:

.. code-block:: bash
    Python 3.9.5 (v3.9.5:0a7dcbdb13, May  3 2021, 13:17:02) 
    [Clang 6.0 (clang-600.0.57)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>>


The installation for requires Python 3.7 or greater.

Install SSSOM-Py for users
----------------------

Install using ``pip``

.. code-block:: bash

    pip install sssom


Install SSSOM-Py for developers
---------------------------


To build directly from source, first clone the GitHub repository,

.. code-block:: bash

    git clone https://github.com/mapping-commons/sssom-py
    cd sssom-py


Then install the necessary dependencies listed in ``setup.cfg``.

.. code-block:: bash

    python setup.py install



For convenience, make use of the ``venv`` module in Python 3 to create a lightweight virtual environment:

.. code-block:: bash

   . environment.sh

   python setup.py install
