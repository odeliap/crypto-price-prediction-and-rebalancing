"""
Scraper for scraping news data

Made with the help of:
https://towardsdatascience.com/scraping-1000s-of-news-articles-using-10-simple-steps-d57636a49755
https://towardsdatascience.com/web-scraping-news-articles-in-python-9dd605799558
"""

# ------------- Libraries -------------
import logging

from bs4 import BeautifulSoup
import sys
import time
import requests

from typing import List

# Set logging level
logging.basicConfig(level=logging.INFO)

# ------------- Constants -------------
pages_to_get = 1


# ------------- Class -------------
class ScratchScraper:

    def __init__(
        self,
        urls: List[str],
        filename: str,
        coin_name: str
    ) -> None:
        """
        Initialize Scraper object.

        Parameters
        __________
        urls : list of strings
            Urls to scrape data from.
        filename : string
            File name to save scraped data to.
        coin_name : string
            Coin for which to scrape data.
        """
        self.upperframe = []
        self.file = open(filename, "w", encoding='utf-8')

        for url in urls:
            for page in range(1, pages_to_get+1):
                logging.info(f'processing page: {page}')

                try:
                    page = self.simple_get_request(url)
                    time.sleep(2)
                    logging.info(f'Scraping text from page {url}')
                    logging.info(page.headers.get("content-type", "unknown"))
                    self.scrape_text(page, coin_name)
                    time.sleep(2)
                except Exception as e:
                    pass

        self.file.close()

    @staticmethod
    def simple_get_request(url: str) -> requests.Response:
        """
        Makes a simple get request (fetches a page).

        Parameters
        __________
        url : string
            Url to get request for.

        Returns
        _______
        page : requests response object
            Response from get request.
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

    def scrape_text(self, page: requests.Response, text: str) -> None:
        """
        Queries page contents for text substring.

        Parameters
        __________
        page : requests response object
            Page for which to scrape text.
        text : string
            Text to query page contents for.
        """
        if text in page.text:
            soup = BeautifulSoup(page.text, 'html.parser')
            frame = []
            links = soup.findAll(
                'div',
                attrs={'class': 'newsletters-individualstyles__ArticleWrapper-sc-160pv05-1 ehdCdZ'}
            )
            logging.info(f'length of links: {len(links)}')
            headers = "Headline, Contents, Link, Date, Source\n"
            self.file.write(headers)

            # This needs to be specific to each page,
            # so this needs to be updated before querying a new page.
            # This should be fixed in the future.
            for j in links:
                headline = j.find("h6", attrs={'class': 'typography_StyledTypography-owin6q-0 kWutUc'}).text.strip()
                contents = j.fing(
                    "h6",
                    attrs={'class': 'display-desktop-block display-tablet-block display-mobile-block'}
                )
                print(contents)
                link = page.url
                link += j.find(
                    "div",
                    attrs={'class': 'display-desktop-block display-tablet-block display-mobile-block '}
                ).find('a')['href'].strip()
                date = j.find(
                    'div',
                    attrs={'class': 'display-desktop-block display-tablet-block display-mobile-block ac-publishing-date'
                           }
                ).find('footer').text[-14:-1].strip()
                source = page.url
                frame.append((headline, link, date, source))
                self.file.write(
                    headline.replace(",", "^") + "," + contents.replace(",", "^") + "," + link + "," +
                    date.replace(",", "^") + "," + source.replace(",", "^"))
            self.upperframe.extend(frame)


if __name__ == "__main__":
    directory = 'datasets/news/unprocessed'

    cryptocurrencies = ['Bitcoin']
    # Once the scraping is fixed to be page-specific,
    # we can uncomment this commented out line of code in place of the line above
    # cryptocurrencies = ['Bitcoin', 'Ethereum', 'Tether', 'USD Coin', 'BNB', 'Binance USD', 'XRP', 'Cardano', 'Solana',
    #                    'Dogecoin', 'Polkadot', 'Polygon', 'Dai', 'SHIBA INU', 'TRON']

    scraping_urls = ["https://www.coindesk.com/newsletters/the-node/"]
    # Once the scraping is fixed to be page-specific,
    # we can uncomment this commented out line of code in place of the line above
    # scraping_urls = ["https://www.coindesk.com/newsletters/the-node/", "https://bitcoinmagazine.com/articles",
    #                 "https://cointelegraph.com/tags/bitcoin", "https://cointelegraph.com/tags/ethereum",
    #                 "https://bitcoinist.com/category/bitcoin/", "https://fintechmagazine.com/articles"]

    for coin in cryptocurrencies:
        scraping_filename = f'{coin.lower()}_scraped_news.csv'
        scraper = ScratchScraper(scraping_urls, f'{directory}/{scraping_filename}', coin)
