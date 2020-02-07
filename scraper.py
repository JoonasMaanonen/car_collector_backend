from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib.request as urllib2
import requests
import shutil
import os
import argparse

def download_image(image_url, dir, id):
    try:
        response = requests.get(image_url, stream=True)
        response.raw.decode_content = True

        file = open(f'{dir}/{id}.jpg', 'wb')
        shutil.copyfileobj(response.raw, file)
    except requests.exceptions.MissingSchema:
        print('Corrupted image!')


parser = argparse.ArgumentParser(description='Scrape car images from autoscout24 website')
parser.add_argument('--num_pages', '-n', type=int, help='Num pages to load per car model')

args = parser.parse_args()

url = "https://www.autoscout24.com/lst/audi/a3"
DATA_DIR = 'testi_autot'
MODEL_DUMMY = 'Audi_a3'
DIR = f'{DATA_DIR}/{MODEL_DUMMY}'

try:
    os.mkdir(DATA_DIR)
    os.mkdir(DIR)
except FileExistsError:
    print(f'Directory already exists')

NUM_PAGES = args.num_pages
image_id = 0

for page_num in range(1, NUM_PAGES):
    print(f"Scraping page {page_num}/{NUM_PAGES-1}...")
    url = f"https://www.autoscout24.com/lst/audi/a3?sort=price&desc=0&size=20&page={page_num}"

    while 1:
        response = requests.get(url)
        page = response.text
        soup = BeautifulSoup(page, "html.parser")
        aas = soup.find_all('img', class_='lazyload')
        if aas:
            break
        else:
            print('We failed no imgs')

    image_urls = []
    for a in aas:
        if a['data-src']:
            if 'seals' not in a['data-src']:
                image_urls.append(a['data-src'])

    for url in image_urls:
        image_id += 1
        download_image(url, DIR, image_id)

