import io
import time
from pathlib import Path
import hashlib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from selenium import webdriver
import yaml
import math

SCRIPT_PATH = Path(__file__).parent.absolute()


def fetch_image_urls(
    query: str,
    max_links_to_fetch: int,
    wd: webdriver,
    sleep_between_interactions: int,
):
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

    for _ in range(math.ceil((max_links_to_fetch - image_count) // 100)):
        scroll_to_end(wd)
    time.sleep(5)

    while image_count < max_links_to_fetch:
        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(
            f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}"
        )

        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(0.1)
            except Exception:
                thumbnail_results.append(img)
                continue

            # extract image urls
            actual_images = wd.find_elements_by_css_selector("img.n3VNCb")
            for actual_image in actual_images:
                if actual_image.get_attribute(
                    "src"
                ) and "http" in actual_image.get_attribute("src"):
                    image_urls.add(actual_image.get_attribute("src"))

                image_count = len(image_urls)

                if len(image_urls) >= max_links_to_fetch:
                    print(f"Found: {len(image_urls)} image links, done!")
                    return image_urls

        print("Found:", len(image_urls), "image links, looking for more ...")
        scroll_to_end(wd)
        time.sleep(10)

        # move the result startpoint further down
        results_start = number_results


def persist_image(folder_path: str, url: str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = (
            Path(folder_path) / hashlib.sha1(image_content).hexdigest()[:10] / ".jpg"
        )
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


def search_and_download(
    search_term: str, driver_path: str, target_path="./images", number_images=5
):
    target_folder = Path(target_path) / " ".join(search_term.lower().split(" "))

    if not Path(target_folder).is_dir():
        Path.mkdir(target_folder, parents=True)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(
            search_term, number_images, wd=wd, sleep_between_interactions=0.5
        )

    for elem in res:
        persist_image(target_folder, elem)


if __name__ == "__main__":
    config = yaml.safe_load(open(SCRIPT_PATH / "download_images_cfg.yaml"))
    keywords = config["keywords"]
    max_num = config["max_num"]

    for keyword in keywords:
        search_and_download(
            search_term=f"star wars {keyword}",
            driver_path="F:/Software/chromedriver.exe",
            target_path="F:/Data Stuff/Star Wars/star_wars_who_am_i/project_data/images",
            number_images=max_num,
        )

    # testing
    # search_and_download(
    #     search_term="star wars",
    #     driver_path="F:/Software/chromedriver.exe",
    #     target_path="F:/Data Stuff/Star Wars/star_wars_who_am_i/project_data/trial",
    #     number_images=max_num,
    # )
