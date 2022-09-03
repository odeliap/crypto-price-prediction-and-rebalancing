Setting up crypto-price-prediction-and-rebalancing
==================================================

This section will teach you how to download and set up crypto-price-prediction-and-rebalancing to predict
cryptocurrency prices and build a automatically rebalancing index fund.

Download the Repository
------------------------

1. Download the crypto-price-prediction-and-rebalancing repository from GitHub at
https://github.com/odeliap/crypto-price-prediction-and-rebalancing

Setting up the Virtual Environment
-----------------------------------

This project utilizes python version 3.9.7, but it should be compatible with other versions of python3. To set this
version, I used ``pyenv`` (common python version management tool). To install ``pyenv``, follow the directions at the
`pyenv repository`_::

    pyenv install 3.9.7
    pyenv local 3.9.7

This project uses ``venv`` for the virtual environment and install the necessary libraries::

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt


Documentation
--------------

Following the video tutorial `How to Document using Sphinx`_,
I setup  ``sphinx`` for this project's documentation, which should be installed with the other project requirements.
After installation, run::

    sphinx-quickstart

and answer the questions to setup the documentation in the **docs** directory. Note: hitting enter will
accept the default value. Check the **docs** directory to confirm all the expected files are there.

For the rest of this section, move to the **docs** directory root. Next, we will install the theme for this project's
documentation with::

    pip install sphinx-rtd-theme


Now, to view the documentation run::

    make html
    cd build/html
    open -a"Google Chrome" index.html

This example uses **Google Chrome**, but you can replace ``Google Chrome`` with the browser of your choice
in the command above. This should open the documentation in this browser.


.. _pyenv repository: https://github.com/pyenv/pyenv#getting-pyenv

.. _How to Document using Sphinx: https://www.youtube.com/playlist?list=PLE72UCmIe7T9HewaqCUhKqiMK3LxYStjy

