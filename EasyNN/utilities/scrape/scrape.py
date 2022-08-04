from itertools import islice
from serpapi import GoogleSearch
import hashlib
from PIL import Image
import io
import os
import requests
import time

from prettyformatter import pprint


def persist_image(folder_path: str, url: str, index: int):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = os.path.join(folder_path, str(index) + ".jpg")
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def scrape_google(
    search_term: str,
    target_path: str = "./images",
    count: int = 5,
):
    """
    Used for scraping images from google.

    Example:
        >>> from EasyNN.scrape.scrape import scrape_google
        >>> scrape_google("dog")
        Opens browser and downloads 5 images of dogs and puts the into images folder.

        >>> scrape_google("dog", count=100)
        Opens browser and downloads 100 images of dog.

        >>> scrape_google("dog", target_path='./my_new_folder')
        Opens browser and downloads 5 images of dog and put it into a folder named "my_new_folder".

    Function parameters:

        search_term: The term you want to scrape google images for.

        count: Number of image to scrape from google.

        driver_path: Defaulted to the path where the python file is executed.

        target_path: Creates a directory with target_path name and puts all
        the images inside here.

    Return:
        Images folder with sub folder of the search term with images inside of there.


    """
    target_folder = os.path.join(
        target_path, "_".join(search_term.lower().split(" ")))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    params = {
        "q": search_term,
        "tbm": "isch",
        "ijn": "0",
        "api_key": "4435d3a85dec7cb053ae29a3e360d111ae85170fa24d8e108c702d3c9b56374c"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    images_results = results["images_results"]

    links: list = []

    for item in islice(images_results, count):
        links.append(item['original'])

    for index, url in enumerate(links):
        persist_image(target_folder, url, index)
