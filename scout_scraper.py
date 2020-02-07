from bs4 import BeautifulSoup
import time
import urllib.request as urllib2
import requests
import shutil
import os
import argparse
import csv

def download_image(image_url, dir, id):
    try:
        response = requests.get(image_url, stream=True)
        response.raw.decode_content = True
        file = open(f'{dir}/sc_{id}.jpg', 'wb')
        shutil.copyfileobj(response.raw, file)
    except requests.exceptions.MissingSchema:
        print('Corrupted image!')

def get_labels(filename):
    labels = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            labels.append(row[0])

    return labels

def get_image_elements(url):
    # With bad internet images might not get loaded, retry until we get images
    retries = 0
    while 1:
        response = requests.get(url)
        page = response.text
        soup = BeautifulSoup(page, 'html.parser')
        imgs = soup.find_all('img', class_='lazyload')
        if imgs:
            return imgs
        else:
            retries += 1
            print('We failed no imgs')
            if retries > 4:
                print('Ran out of pages!')
                return []

        time.sleep(3)

def get_image_urls(url):
    imgs = get_image_elements(url)
    image_urls = []
    for img in imgs:
        if img['data-src']:
            if 'seals' not in img['data-src']:
                image_urls.append(img['data-src'])
    return image_urls

def main():
    parser = argparse.ArgumentParser(description='Scrape car images from autoscout24 website')
    parser.add_argument('--num_pages', '-n', default=21, type=int, help='Num pages to load per car model')
    parser.add_argument('--label_file', '-f', required=True, help='Csv file that contains car models to scrape.')
    parser.add_argument('--data_dir', '-d', required=True, help='Which directory to download the images into')
    args = parser.parse_args()

    labels = get_labels(args.label_file)

    try:
        os.mkdir(args.data_dir)
    except FileExistsError:
        print(f'Directory {args.data_dir} already exists')

    for i, _ in enumerate(labels):
        image_id = 0
        car = labels[i].split()
        model = car[-1]
        brand = '-'.join(car[:-1])
        base_url = f"https://www.autoscout24.com/lst/{brand}/{model}"
        model_folder = '_'.join(car)
        print(f"Scraping images for {model_folder}")
        DIR = f'{args.data_dir}/{model_folder}'
        try:
            os.mkdir(DIR)
        except FileExistsError:
            print(f'Directory {DIR} already exists')

        for page_num in range(1, args.num_pages):
            print(f"Scraping page {page_num}/{args.num_pages-1}...")
            url = f"{base_url}?sort=age&desc=1&size=20&page={page_num}"
            image_urls = get_image_urls(url)
            if not image_urls:
                break
            for url in image_urls:
                image_id += 1
                download_image(url, DIR, image_id)

if __name__ == "__main__":
    main()
