# Cryptocurrency Price Prediction and Automated Index Fund Rebalancing 💸
Repository for cryptocurrency price prediction with automated index-fund rebalancing on predicted prices.

This project uses sentiment analysis and an LSTM model for cryptocurrency price predictions.

## Setup Virtual Environment

This project utilizes python version 3.9.12, but it should be compatible with other versions of python3 
except for version 3.9.7 which is incompatible with `streamlit`. To set this version, I used `pyenv` 
(common python version management tool). To install `pyenv`, follows the directions at the 
[pyenv repository](https://github.com/pyenv/pyenv#getting-pyenv).

```commandline
pyenv install 3.9.12
pyenv local 3.9.12
```

This project uses `venv` for the virtual environment and install the necessary libraries.

```commandline
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```


## Documentation

Following the video tutorial [How to Document using Sphinx](https://www.youtube.com/playlist?list=PLE72UCmIe7T9HewaqCUhKqiMK3LxYStjy), 
I setup  `sphinx` for this project's documentation, which should be installed with the other project requirements. 
After installation, run

```commandline
sphinx-quickstart
```

and answer the questions to setup the documentation in the __docs__ directory. Note: hitting enter will 
accept the default value. Check the __docs__ directory to confirm all the expected files are there.

For the rest of this section, move to the __docs__ directory root. Next, we will install the theme for this project's 
documentation with:

```commandline
pip install sphinx-rtd-theme
```

Now, to view the documentation run:

```commandline
make html
cd build/html
open -a"Google Chrome" index.html
```

This example uses __Google Chrome__, but you can replace `Google Chrome` with the browser of your choice 
in the command above. This should open the documentation in this browser.


## Run Streamlit App

Prior to running the streamlit app, we need to export the path to this project. Once you have the path to this project, 
execute the following export in the terminal

```
export PYTHONPATH="${PYTHONPATH}:/path/to/this/project/"
```

where ```path/to/this/project``` is replaced with the actual path.

Next, in the terminal at the root of this project execute

```commandline
streamlit run src/pipeline/app.py
```

The terminal will display a `localhost` link from this command. Press this link and open the page or copy-paste the 
link into your browser, and the app should appear!