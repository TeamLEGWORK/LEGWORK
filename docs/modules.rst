LEGWORK modules and API reference
=================================

.. toctree::
   :maxdepth: 4

``LEGWORK`` is composed of 7 different modules, each with a different focus. The diagram below illustrates
how each module is connected to the others as well as listing the general purpose of each module. In particular,
note that the ``source`` module provides a simple interface to the functions in all other modules!

.. image:: https://github.com/TeamLEGWORK/LEGWORK-paper/raw/main/src/static/package_overview.png
   :width: 600
   :alt: Package structure graph
   :align: center

The rest of this page contains the API reference for each individual function in every module, feel free to
use the table of contents on the left to easily navigate to the function you need.

.. automodapi:: legwork.evol

.. tip::
    Feeling a bit spun around by all this binary evolution? Check out our
    tutorial on using functions in the ``evol`` module `here! <notebooks/Evolution.ipynb>`__

.. automodapi:: legwork.psd

.. automodapi:: legwork.snr

.. automodapi:: legwork.source

.. tip::
    Unable to find the source of your issues? Never fear! Check out our
    tutorial on using the Source class `here! <notebooks/Source.ipynb>`__

.. automodapi:: legwork.strain

.. tip::
    Feeling a little strained trying to parse these docs? Check out our
    tutorial on using functions in the ``strain`` module `here! <notebooks/Strains.ipynb>`__

.. automodapi:: legwork.utils

.. automodapi:: legwork.visualisation

.. tip::
    Not quite sure how things are working vis-à-vis visualisation? Check out our
    tutorial on using functions in the ``visualisation`` module `here! <notebooks/Visualisation.ipynb>`__