"""
Scraper for scraping news data

Made with the help of:
https://towardsdatascience.com/scraping-1000s-of-news-articles-using-10-simple-steps-d57636a49755
https://towardsdatascience.com/web-scraping-news-articles-in-python-9dd605799558
"""

# ------------- Libraries -------------
import logging

from bs4 import BeautifulSoup
import sys, time
import requests

from typing import List

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------
pagesToGet = 1


# ------------- Class -------------
class Scraper:

    def __init__(self, urls: List[str], filename: str, coin: str):
        """
        Initialize Scraper object.

        :param urls: list of urls to scrape data for
        :type List[str]

        :param filename: file to save scraped data to
        :type str

        :param coin: coin to scrape data for
        :type str
        """
        self.upperframe = []
        self.file = open(filename, "w", encoding = 'utf-8')

        for url in urls:
            for page in range(1, pagesToGet+1):
                logging.info(f'processing page: {page}')

                try:
                    page = self.simpleGetRequest(url)
                    time.sleep(2)
                    logging.info(f'Scraping text from page {url}')
                    logging.info(page.headers.get("content-type", "unknown"))
                    self.scrapeText(page, coin)
                    time.sleep(2)
                except Exception as e:
                    pass

        self.file.close()

    @staticmethod
    def simpleGetRequest(url: str) -> requests.Response:
        """
        Makes a simple get request (fetches a page).

        :param url: url to get request for
        :type str

        :return: page: response from get request
        :rtype requests.Response
        """
        try:
            page = requests.get(url)
            status_code = str(page.status_code)
            if status_code.startswith('4') or status_code.startswith('5'):
                raise Exception(f'Non-200 status code {status_code} received')
            else:
                return page
        except Exception as e:
            error_type, error_obj, error_info = sys.exc_info()
            logging.info(f'{error_type}, line: {error_info.tb_lineno}')
            raise Exception(f'Error processing url: {url}')


    def scrapeText(self, page: requests.Response, text: str):
        """
        Queries page contents for text substring.

        :param page: page to scrape text for
        :type requests.Response

        :param text: text to query page contents for
        :type str

        :return:
        """
        if text in page.text:
            soup = BeautifulSoup(page.text, 'html.parser')
            frame = []
            links = soup.findAll('div', attrs={'class':'newsletters-individualstyles__ArticleWrapper-sc-160pv05-1 ehdCdZ'})
            logging.info(f'length of links: {len(links)}')
            headers = "Headline, Contents, Link, Date, Source\n"
            self.file.write(headers)

            # TODO: FIXME (needs to be specific to each page as well)
            for j in links:
                headline = j.find("h6", attrs={'class': 'typography_StyledTypography-owin6q-0 kWutUc'}).text.strip()
                contents = j.fing("h6", attrs={'class': 'display-desktop-block display-tablet-block display-mobile-block'})
                print(contents)
                link = page.url
                link += j.find("div", attrs={'class': 'display-desktop-block display-tablet-block display-mobile-block '}).find('a')['href'].strip()
                date = j.find('div', attrs={'class': 'display-desktop-block display-tablet-block display-mobile-block ac-publishing-date'}).find('footer').text[-14:-1].strip()
                source = page.url
                frame.append((headline, link, date, source))
                self.file.write(
                    headline.replace(",", "^") + "," + contents.replace(",", "^") + "," + link + "," + date.replace(",", "^") + "," + source.replace(",", "^"))
            self.upperframe.extend(frame)


if __name__ == "__main__":
    directory = 'datasets/news/unprocessed'

    cryptocurrencies = ['Bitcoin']
    #cryptocurrencies = ['Bitcoin', 'Ethereum', 'Tether', 'USD Coin', 'BNB', 'Binance USD', 'XRP', 'Cardano', 'Solana', 'Dogecoin', 'Polkadot', 'Polygon', 'Dai', 'SHIBA INU', 'TRON']

    urls = ["https://www.coindesk.com/newsletters/the-node/"]
    #urls = ["https://www.coindesk.com/newsletters/the-node/", "https://bitcoinmagazine.com/articles", "https://cointelegraph.com/tags/bitcoin", "https://cointelegraph.com/tags/ethereum", "https://bitcoinist.com/category/bitcoin/", "https://fintechmagazine.com/articles"]

    for coin in cryptocurrencies:
        filename = f'{coin.lower()}_scraped_news.csv'
        scraper = Scraper(urls, f'{directory}/{filename}', coin)