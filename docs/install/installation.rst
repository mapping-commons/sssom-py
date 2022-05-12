Installation
============

The installation for requires Python 3.6 or greater.

Installation for users
----------------------

Install using ``pip``

.. code-block:: bash

    pip install sssom


Installation for developers
---------------------------


To build directly from source, first clone the GitHub repository,

.. code-block:: bash

    git clone https://github.com/mapping-commons/sssom-py
    cd sssom-py


Then install the necessary dependencies listed in ``requirements.txt``.

.. code-block:: bash

    pip3 install -r requirements.txt



For convenience, make use of the ``venv`` module in Python 3 to create a lightweight virtual environment:

.. code-block:: bash

   . environment.sh

   pip install -r requirements.txt
