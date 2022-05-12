Documentation Update and Redeployment
=====================================

Using your terminal, ``cd`` to the 'sphinx' directory within the project.
::
cd sphinx

New page
--------

If you need to add a new page create a new file under the sphinx directory (sphinx/*filename*.rst). Update the content 
within *filename*.rst to the documentation you wish to add using the guidance provided by the 
`Sphinx <https://www.sphinx-doc.org/en/master/contents.html>`_ web page.

Update and page deployment
-----------

This is done automatically by GitHub Actions through a workflow as soon as a 
pull request is merged to the ``master`` branch.



