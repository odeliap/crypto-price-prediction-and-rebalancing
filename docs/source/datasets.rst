.. _datasets:

Dataset Sourcing and Creation
==============================

In this section, you will learn about:

* :ref:`Dataset Sourcing <sourcing>`
* :ref:`Dataset Creation <creatingdatasets>`

This section will teach you everything I did to source and create datasets for this project.

.. _sourcing:

Sourcing Datasets
------------------

As part of this project, I sourced available datasets for historical cryptocurrency prices and news headlines around
cryptocurrencies from Kaggle. I pulled the following datasets for inclusion in my final dataset for sentiment analysis.

Price Datasets
~~~~~~~~~~~~~~~

* `Bitcoin Price`_
* `Ethereum Price 1`_
* `Ethereum Price 2`_
* `Solana Price`_

News Datasets
~~~~~~~~~~~~~~

* `Bitcoin News 1`_
* `Bitcoin News 2`_
* `Ethereum News`_
* `Solana News`_

These raw datasets are found and stored in this project under the **src/data_cleaner/datasets** folder.


.. _creatingdatasets:

Scraping Data and Sourcing with APIs
------------------------------------

After sourcing datasets for this project, I was still missing crucial data for the sentiment analysis and machine
learning aspects of this project. So, I attempted to use existing APIs to collect more data. This was largely
unsuccessful since the APIs have limits on querying. The classes built out to attempt to get more data are found under
the **scr/scraper** package. In the future, these could be used to source more data if someone is willing to pay for
premium subscription plans and/or individually sort through the html for different cryptocurrency blogs to scrape news
from these directly using the **ScratchScraper** class included.

I used the scraping APIs from:

* `News API`_; and
* `Crypto News API`_.

The news APIs only had access to news data, so I built these to scrape news data.

I also included an **OpenBlender** class to pull from their API to retrieve found datasets. If you run this class,
these datasets will get stored under *src/scraper/datasets* as well.

Note: None of the datasets from these APIs and scraper(s) were used to build the datasets used in this project.
However, the **NewsApiScraper** is used as part of the end-to-end pipeline to get news data for predicting near future
prices.

Cleaning the Datasets
---------------------

With the price and news datasets found on Kaggle, I proceeded to make a clean, cohesive dataset for each coin. To do so,
I cleaned all the timestamps to a uniform standard, joined all the news headlines, grouping them by timestamp, and
removed all rows with empty entries. A class was made for each coin to perform this cleaning, which all utilize a common
**Utils** class. These coin-specific classes and this **Utils** class are found under **src/data_cleaner**.

The output from this cleaning step is stored under **src/data_cleaner/outputs**.


.. _Bitcoin News 1: https://www.kaggle.com/datasets/c5e1371384af39901791384a29d20195e0e3d4068b68fb1b12d58caf5a76ff33?select=bitcoin_news_coin_telegraph0.csv
.. _Bitcoin News 2: https://www.kaggle.com/muhammedabdulazeem/bitcoin-price-prediction
.. _Ethereum News: https://www.kaggle.com/datasets/mathurinache/ethereum-tweets
.. _Solana News: https://www.kaggle.com/datasets/aglitoiumarius/rsolana-comments-202001202204

.. _Bitcoin Price: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
.. _Ethereum Price 1: https://www.kaggle.com/datasets/ranugadisansagamage/ethereum-crypto-price
.. _Ethereum Price 2: https://www.kaggle.com/datasets/psycon/ethusdt-2017-to-2022
.. _Solana Price: https://www.kaggle.com/datasets/varpit94/solana-data

.. _News API: https://newsapi.org
.. _Crypto News API: https://cryptonews-api.com