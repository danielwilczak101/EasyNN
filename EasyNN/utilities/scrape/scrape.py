import hashlib
from PIL import Image
import io, os
import requests
import time
from selenium import webdriver

"""
Code downloaded from:
    Medium article: https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d
    Author:Fabian Bosler
"""

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception as e:
                print(e)
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            #return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path:str,url:str,index:int):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path, str(index) + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

def scrape_google(search_term:str, driver_path='./chromedriver', target_path='./images', count=5):
    """
    Used for scraping images from google using the chromedriver.

    Requirement:
        chromedriver - 
            1. Install Google Chrome (skip if its already installed)
            2. Identify your Chrome version. In chrome go to  "About Google Chrome". I 
            currently have version 77.0.3865.90 (my main version is thus 77, the number
            before the first dot).
            
            https://chromedriver.chromium.org/downloads

            Download your corresponding ChromeDriver from the link provided for your main
            version and put the executable into the same folder as your python file.

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
    target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term, count, wd=wd, sleep_between_interactions=0.5)

    for index, elem in enumerate(res):
        persist_image(target_folder,elem,index)

