Update Documentation
====================

Using your terminal, ``cd`` to the 'docs' directory within the project.

.. code-block:: bash

    cd docs

New page
--------

If you need to add a new page create a new file in this directory (docs/*filename*.rst). Update the content 
within *filename*.rst to the documentation you wish to add using the guidance provided by the 
`Sphinx <https://www.sphinx-doc.org/en/master/contents.html>`_ web page.

Update and page deployment
-----------

This is done automatically by GitHub Actions through a workflow as soon as the 
pull request is merged to the ``master`` branch.
