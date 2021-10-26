Installation
============

.. tabs::

    .. tab:: Stable (from PyPI)

        We recommend that you create a Conda environment for working with LEGWORK.
        You can do this by running

        .. code-block:: bash

            conda create --name legwork numpy scipy astropy numba matplotlib seaborn jupyter ipython

        And then activate the environment by running

        .. code-block:: bash

            conda activate legwork

        Once within the environment, LEGWORK is available for installation on PyPI which lets you install the latest
        stable version using ``pip``. One last dependency is the schwimmbad package which is only available
        on PyPI not conda and therefore also needs to be brought in with ``pip``. So now to complete the installation just run

        .. code-block:: bash

            pip install schwimmbad legwork

        and you should be all set! Check out our `quickstart tutorial <notebooks/Quickstart.ipynb>`__ to learn some LEGWORK basics.

        .. tip::

            If you see an error about llvmlite of the form "*ERROR: Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.*" this is due to the nature of the llvmlite and numba packages
            and can be avoided by instead running

            .. code-block:: bash

                pip install schwimmbad legwork --ignore-installed llvmlite

    .. tab:: Development (from GitHub)

        The latest development version is available directly from our `GitHub Repo
        <https://github.com/TeamLEGWORK/LEGWORK>`_:

        .. code-block:: bash
        
            git clone https://github.com/TeamLEGWORK/LEGWORK
            cd LEGWORK
            pip install .
