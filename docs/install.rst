Installation
============

.. tabs::

    .. tab:: Stable (with conda)

        This is our recommend installation method! Follow the steps below to start using ``LEGWORK``!

        #. Create a new conda environment for LEGWORK

            .. code-block:: bash

                conda create --name legwork "python=3.13" pip -c conda-forge

        #. Activate the environment by running

            .. code-block:: bash

                conda activate legwork

        #. Install LEGWORK (and its dependencies) into the environment

            .. code-block:: bash

                pip install legwork

        and you should be all set! Check out our `quickstart tutorial <notebooks/Quickstart.ipynb>`__ to learn some LEGWORK basics.
        Note that if you also want to work with the notebooks in the tutorials and/or demos you'll also need to install jupyter/ipython in this environment!

    .. tab:: Stable (without conda)

        We don't recommend installing ``LEGWORK`` without a conda environment but if you prefer to do it this
        way then all you need to do is run

        .. code-block:: bash

            pip install legwork

        and you should be all set! Check out our `quickstart tutorial <notebooks/Quickstart.ipynb>`__ to learn some LEGWORK basics.
        Note that if you also want to work with the notebooks in the tutorials and/or demos you'll also need to install jupyter/ipython in this environment!

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

            conda create --name legwork "python=3.13" pip -c conda-forge

        And then activate the environment by running

        .. code-block:: bash

            conda activate legwork

        At this point, all that's left to do is install LEGWORK! We recommend an editable install so that
        your changes to the source are picked up immediately, along with the ``test`` extra so that you can
        run the test suite.

        .. code-block:: bash

            pip install -e ".[test]"

        and you should be all set! Check out our `quickstart tutorial <notebooks/Quickstart.ipynb>`__ to learn some LEGWORK basics.
        Note that if you also want to work with the notebooks in the tutorials and/or demos you'll also need to install jupyter/ipython in this environment!
