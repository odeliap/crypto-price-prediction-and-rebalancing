# Cryptocurrency Price Prediction and Automated Index Fund Rebalancing 💸
Repository for cryptocurrency price prediction with automated index-fund rebalancing on predicted prices.

This project uses sentiment analysis and an LSTM model for cryptocurrency price predictions. To run this project, 
follow the below instructions.

## Setup Virtual Environment

This project utilizes python version `3.9.12`, but it should be compatible with other versions of `python3`
except for version `3.9.7` which is incompatible with `streamlit`. To set this version, I used `pyenv` 
(common python version management tool). To install `pyenv`, follows the directions at the 
[pyenv repository](https://github.com/pyenv/pyenv#getting-pyenv).

```commandline
pyenv install 3.9.12
pyenv local 3.9.12
```

This project uses `venv` for the virtual environment. If you do not have `venv`, you should be able to use 
the pre-installed virtual environment tool included in your python version. Activate your virtual environment 
by running:

```commandline
python -m venv .venv
. .venv/bin/activate
```

To exit the virtual environment at any time, simply run:

```commandline
deactivate
```

Next, in your virtual environment, install the necessary dependencies by running:

```commandline
pip install -r requirements.txt
```

All the package versions have been set to future-proof this repository. These versions can be viewed under the
**requirements.txt** file found under the root folder of this project.

You should be all set to run this project!


## Run The Pipeline

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

Once you are ready, run the **kick off pipeline** button that appears, and you should see the pipeline run.


## Additional Documentation

Following the video tutorial [How to Document using Sphinx](https://www.youtube.com/playlist?list=PLE72UCmIe7T9HewaqCUhKqiMK3LxYStjy), 
I setup  `sphinx` for this project's additional documentation, which should be installed with the other project requirements. 
If you wish to view this additional documentation, after setting up the virtual environment, run

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

## Code Architecture Overview

The code in this project has been organized under subpackages. These subpackages are:

* data_cleaner
* evaluation
* index_fund_rebalancing
* models
* pipeline
* scraper
* sentiment_analysis

### Data Cleaning

**data_cleaner** holds all the code related to cleaning and pre-processing the found datasets from Kaggle into clean 
and cohesive files which can be ingested by the sentiment analyzer. There exists a class for each coin under this 
subpackage with logic for processing the coin-specific files. Additionally, there is a _Utils_ file for code that is 
commonly used across the coins' separate classes.

All the datasets that are cleaned by these coin-specific cleaning classes can be found under the _datasets_ 
subdirectory. All the outputs from this cleaning process are stored under the _outputs_ subdirectory.

### Data Scraping

**scraper** holds all the code related to scraping cryptocurrency data. This includes the classes built out to use 
found APIs for getting cryptocurrency news. The following files exist in this subpackage:

* CryptoNewsApiScraper.py
* NewsApiScraper.py
* OpenblenderApiScraper.py
* ScratchScraper.py
* Utils.py

The _CryptoNewsApiScraper.py_ file holds a class for utilizing the **Crypto News API** to get cryptocurrency news.
Similarly, the _NewsApiScraper.py_ file holds a class for using the **News API**, organized to call API for each coin to 
create a csv file with the related news. The _OpenblenderApiScraper.py_ file has methods to get relevant datasets from 
the **OpenBlender API**. The _ScratchScraper.py_ file is incomplete but has methods to support scraping news. The 
_Utils.py_ file holds methods that are commonly used across all these files.

All the datasets "scraped" by these processes are stored under the _datasets_ subdirectory.

### Sentiment Analysis

**sentiment_analysis** holds all the code related to the sentiment analysis process. There is a single class here 
called _Sentiment_Analyzer_, which relates to the process of sentiment analysis. This class uses **VADER Sentiment 
Analysis** (a library for sentiment analysis) to perform this operation. The call to main loops over the cleaned files
(outputted from the cleaning step) and, for each file, it performs sentiment analysis and saves the outputs to a new csv 
file.

### LSTM Price Predictions

**models** holds all the models related to cryptocurrency price prediction. There are a number of classes here. The 
main class (used in the pipeline) for LSTM price predictions is _SentimentPriceLSTMModel_. The other model classes 
located here were used in the evaluation process. The _Unpickler_ holds methods for unpickling objects (used in loading 
stored scalers and models). The _Utils_ file holds utility methods used across a number of the model classes.

#### Evaluation

**evaluation** holds code related to the evaluation of this project. The _EvaluationInputGenerator_ class is used to 
create constant-sentiment datasets which gets inputted to the _EvaluationSentimentPriceLSTMModel_ (a replica of the 
_SentimentPriceLSTMModel_ with slight variations) to support evaluation of the sentiment data in advising cryptocurrency 
price predictions. The _RebalancingEvaluation_ class utilizes methods from its respective utils class to generate 
reporting for the index fund re-balancing.

### Index Fund Rebalancing

**index_fund_rebalancing** holds all the code for index fund re-balancing.

### End-to-End Pipeline

**pipeline** holds the end-to-end pipeline and supporting materials. The streamlit app structure is found here 
additionally. The _Pipeline_ class runs the end-to-end pipeline, _app_ runs the streamlit app (calling the pipeline), 
and _Utils_ holds common utility functions.