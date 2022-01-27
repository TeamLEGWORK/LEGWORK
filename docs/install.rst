Installation
============

.. tabs::

    .. tab:: Stable (from PyPI)

        We recommend that you create a Conda environment for working with LEGWORK.
        You can do this by running

        .. code-block:: bash

            conda create --name legwork numba>=0.50 numpy>=1.16 astropy>=4.0 scipy>=1.5.0 matplotlib>=3.3.2 seaborn>=0.11.1

        And then activate the environment by running

        .. code-block:: bash

            conda activate legwork

        Once within the environment, LEGWORK is available for installation on PyPI which lets you install the latest
        stable version using ``pip``. So now to complete the installation just run the following and LEGWORK, as well as its dependencies, will be installed

        .. code-block:: bash

            pip install legwork

        .. tip::

            If you see an error about llvmlite of the form "*ERROR: Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.*" this is due to the nature of the llvmlite and numba packages
            and can be avoided by instead running

            .. code-block:: bash

                pip install legwork --ignore-installed llvmlite

        Finally, if you also want to work with the notebooks in the tutorials and/or demos you'll also need to run this

        .. code-block:: bash

            conda install jupyter ipython

        and you should be all set! Check out our `quickstart tutorial <notebooks/Quickstart.ipynb>`__ to learn some LEGWORK basics.

    .. tab:: Development (from GitHub)
        
        .. warning::

            We don't guarantee that there won't be mistakes or bugs in the development version, use at your own risk!

        The latest development version is available directly from our `GitHub Repo
        <https://github.com/TeamLEGWORK/LEGWORK>`_. To start, clone the repository onto your machine:

        .. code-block:: bash
        
            git clone https://github.com/TeamLEGWORK/LEGWORK
            cd LEGWORK

        Next, we recommend that you create a Conda environment for working with LEGWORK.
        You can do this by running

        .. code-block:: bash

            conda create --name legwork numba>=0.50 numpy>=1.16 astropy>=4.0 scipy>=1.5.0 matplotlib>=3.3.2 seaborn>=0.11.1

        And then activate the environment by running

        .. code-block:: bash

            conda activate legwork

        At this point, all that's left to do is install LEGWORK!

        .. code-block:: bash

            pip install .

        Keep in mind that if you want to work with the notebooks in the tutorials/demos you'll also need to run the following

        .. code-block:: bash

            conda install jupyter ipython