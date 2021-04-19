Installation
============

Stable Version
--------------
We recommend that you create a Conda environment for working with LEGWORK.
You can do this by running

.. code-block:: bash

    conda create --name legwork numpy scipy astropy numba matplotlib seaborn jupyter ipython

And then activate the environment by running

.. code-block:: bash

    conda activate legwork

LEGWORK is available for installation on PyPI and you can install the latest
stable version using ``pip`` (we recommend doing this within a conda environment
as described above) with

.. code-block:: bash

    pip install legwork

.. tip::

    If you see an error about llvmlite of the form "*ERROR: Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.*" this is due to the nature of the llvmlite and numba packages
    and can be avoided by instead running

    .. code-block:: bash

        pip install legwork --ignore-installed llvmlite

Development Version
-------------------
The latest development version is available directly from our `GitHub Repo
<https://github.com/katiebreivik/LEGWORK>`_: ::

    git clone https://github.com/katiebreivik/LEGWORK
    cd LEGWORK
    pip install .
