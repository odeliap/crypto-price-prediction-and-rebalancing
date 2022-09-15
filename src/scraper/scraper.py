"""
Scraper for scraping news data

Made with the help of: https://towardsdatascience.com/scraping-1000s-of-news-articles-using-10-simple-steps-d57636a49755
"""

# ------------- Libraries -------------
import logging

from bs4 import BeautifulSoup
import sys, time
import requests

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------
pagesToGet = 1


# ------------- Class -------------
class Scraper:

    def __init__(self, url: str, filename: str):
        """
        Initialize Scraper object.

        :param url: url to scrape data for
        :type str

        :param filename: file to save scraped data to
        :type str
        """
        self.upperframe = []
        for page in range(1, pagesToGet+1):
            logging.info(f'processing page: {page}')
            self.url = url

        self.page = self.simpleGetRequest()
        time.sleep(2)
        logging.info(f'Scraping text from page {url}')
        logging.info(self.page.headers.get("content-type", "unknown"))
        time.sleep(2)

        self.file = open(filename, "w", encoding = 'utf-8')


    def simpleGetRequest(self) -> requests.Response:
        """
        Makes a simple get request (fetches a page).

        :return: page: response from get request
        :rtype requests.Response
        """
        try:
            page = requests.get(self.url)
            status_code = str(page.status_code)
            if status_code.startswith('4') or status_code.startswith('5'):
                raise Exception(f'Non-200 status code {status_code} received')
            else:
                return page
        except Exception as e:
            error_type, error_obj, error_info = sys.exc_info()
            logging.info(f'Error processing url: {self.url}')
            logging.info(f'{error_type}, line: {error_info.tb_lineno}')
            pass


    def lookForText(self, text: str):
        """
        Queries page contents for text substring.

        :param text: text to query page contents for
        :type str

        :return:
        """
        if text in self.page.text:
            soup = BeautifulSoup(self.page.text, 'html.parser')
            frame = []
            links = soup.findAll('li', attrs={'class':'o-listicle_item'})
            logging.info(f'length of links: {links}')
            headers = "Statement, Link, Date, Source, Label\n"
            self.file.write(headers)

            for j in links:
                Statement = j.find("div", attrs={'class': 'm-statement__quote'}).text.strip()
                Link = "https://www.politifact.com"
                Link += j.find("div", attrs={'class': 'm-statement__quote'}).find('a')['href'].strip()
                Date = j.find('div', attrs={'class': 'm-statement__body'}).find('footer').text[-14:-1].strip()
                Source = j.find('div', attrs={'class': 'm-statement__meta'}).find('a').text.strip()
                Label = j.find('div', attrs={'class': 'm-statement__content'}).find('img', attrs={
                    'class': 'c-image__original'}).get('alt').strip()
                frame.append((Statement, Link, Date, Source, Label))
                self.file.write(
                    Statement.replace(",", "^") + "," + Link + "," + Date.replace(",", "^") + "," + Source.replace(",",
                                                                                                                   "^") + "," + Label.replace(
                        ",", "^") + "\n")
            self.upperframe.extend(frame)


if __name__ == "__main__":
    directory = 'datasets/news/unprocessed/'

    cryptocurrencies = ['Bitcoin', 'Ethereum', 'Tether', 'USD Coin', 'BNB', 'Binance USD', 'XRP', 'Cardano', 'Solana', 'Dogecoin', 'Polkadot', 'Polygon', 'Dai', 'SHIBA INU', 'TRON']

    urls = [] # TODO: fill with urls

    for crypto in cryptocurrencies:
        filename = f'{crypto.lower()}_scraped_news.csv'

        for url in urls:
            scraper = Scraper(url, filename)