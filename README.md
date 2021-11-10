<p align="center">
    <img width="500", src="https://raw.githubusercontent.com/TeamLEGWORK/LEGWORK/main/docs/images/legwork.png">
</p>

<h2 align="center">
    The <b>L</b>ISA <b>E</b>volution and <b>G</b>ravitational <b>W</b>ave <b>OR</b>bit <b>K</b>it
    <br>
    <a href="https://github.com/TeamLEGWORK/LEGWORK-paper">
        <img src="https://img.shields.io/badge/release paper-repo-blue.svg?style=flat&logo=GitHub" alt="Read the article"/>
    </a>
    <a href="https://codecov.io/gh/TeamLEGWORK/LEGWORK">
        <img src="https://codecov.io/gh/TeamLEGWORK/LEGWORK/branch/main/graph/badge.svg?token=FUG4RFYCWX"/>
    </a>
    <a href='https://legwork.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/legwork/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="mailto:tomjwagg@gmail.com?cc=kbreivik@flatironinstitute.org">
        <img src="https://img.shields.io/badge/contact-authors-blueviolet.svg?style=flat" alt="Email the authors"/>
    </a>
</h2>

<p align="center">
    A python package that does the <code>LEGWORK</code> for you by evolving binaries,
    computing signal-to-noise ratios for binary systems potentially observable with LISA
    and visualising the results.
</p>

# Installation
## Stable Version
We recommend that you create a Conda environment for working with LEGWORK.
You can do this by running

    conda create --name legwork numpy scipy astropy numba matplotlib seaborn jupyter ipython

And then activate the environment by running

    conda activate legwork

Once within the environment, LEGWORK is available for installation on PyPI which lets you install the latest
stable version using ``pip``. One last dependency is the schwimmbad package which is only available
on PyPI not conda and therefore also needs to be brought in with ``pip``. So now to complete the installation just run

    pip install schwimmbad legwork

and you should be all set! Check out our [quickstart tutorial](https://legwork.readthedocs.io/en/latest/notebooks/Quickstart.html) to learn some LEGWORK basics.


If you see an error about llvmlite of the form "*ERROR: Cannot uninstall 'llvmlite'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.*" this is due to the nature of the llvmlite and numba packages
and can be avoided by instead running

    pip install schwimmbad legwork --ignore-installed llvmlite

## Development Version
The latest development version is available directly from our
[GitHub Repo](https://github.com/TeamLEGWORK/LEGWORK)

    git clone https://github.com/TeamLEGWORK/LEGWORK
    cd LEGWORK
    pip install .

# Documentation
You can find our documentation [here](https://legwork.readthedocs.io/en/latest/)
and explore various tutorials as well details on every module and function.

# Development
The source code is available [here](https://github.com/TeamLEGWORK/LEGWORK)
on GitHub and if you have an idea for a new feature, notice a bug or something else entirely,
please [open an issue](https://github.com/TeamLEGWORK/LEGWORK/issues/new) and let us know — we'd love to hear about it!
