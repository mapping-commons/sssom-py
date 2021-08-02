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

Update page
-----------

If you need to update a particular page, go to the corresponding ``.rst`` file within the 'sphinx' folder and make changes accordingly.

Deploy page
------------
Using the terminal within the 'sphinx' directory, type ``make clean`` and then followed by ``make html``. This creates the HTML
files corresponding to each ``*.rst`` files within the 'sphinx/_build/html' directory. Copy-paste the contents from the 'sphinx/_build/html'
directory into the 'docs' directory in the project. Commit the changes into your GitHub project repository and see the changes 
reflected on the `documentation page <https://mapping-commons.github.io/sssom-py/index.html>`_


