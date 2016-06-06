from bs4 import BeautifulSoup
import requests
import re
from trigram_tagger import SubjectTrigramTagger
from nltk.corpus import stopwords
stop = stopwords.words('english')

def download_document(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    title = soup.find('title').get_text()
    document = ' '.join([p.get_text() for p in soup.find_all('p')])
    return document

if __name__ == '__main__':
    url = 'http://techcrunch.com/2016/05/26/snapchat-series-f/?ncid=tcdaily'
    document = download_document(url)
    # extract_subject()
