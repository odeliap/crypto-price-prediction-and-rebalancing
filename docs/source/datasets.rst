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

* `Cryptocurrency Historical Prices`_

* `TOP 50 Cryptocurrencies Historical Prices`_

* `Top 100 Cryptocurrencies Historical Dataset`_

* `Cryptoindex.com 100`_


News Datasets
~~~~~~~~~~~~~~

* `Bitcoin Sentiment Analysis`_

* `News about major cryptocurrencies 2013-2018 (40k)`_

* `Cryptocurrency News Tweet`_

Most of these datasets are found and stored in this project under the root **datasets** folder.

Note: the **OpenBlender** datasets I had to pull from their API, so these datasets are stored under
*src/scraper/datasets*. Additionally, the **Cryptoindex.com 100** dataset was not used for sentiment analysis or
model training but is used for the evalution of this project's index fund to compare against a known index fund's
performance.


.. _creatingdatasets:

Creating Datasets
------------------

After sourcing datasets for this project, I was still missing crucial data for the sentiment analysis and machine
learning aspects of this project. So, I collected my own data and then joined it with the sourced data to create fully
viable datasets.

To create my own datasets, I used the scraping APIs from:

* `News API`_; and
* `Crypto News API`_.

I needed to supplement both the crypto prices and crypto news datasets. The news APIs only had access to news data, so
used those to scrape news data.



.. _Bitcoin Sentiment Analysis: https://www.kaggle.com/code/codeblogger/bitcoin-sentiment-analysis/data
.. _Cryptocurrency Historical Prices: https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory?resource=download
.. _TOP 50 Cryptocurrencies Historical Prices: https://www.kaggle.com/datasets/odins0n/top-50-cryptocurrency-historical-prices
.. _Top 100 Cryptocurrencies Historical Dataset: https://www.kaggle.com/datasets/kaushiksuresh147/top-10-cryptocurrencies-historical-dataset
.. _News about major cryptocurrencies 2013-2018 (40k): https://www.kaggle.com/datasets/kashnitsky/news-about-major-cryptocurrencies-20132018-40k

.. _News API: https://newsapi.org
.. _Crypto News API: https://cryptonews-api.com
.. _Cryptoindex.com 100: https://openblender.io/#/search
.. _Cryptocurrency News Tweet: https://openblender.io/#/search