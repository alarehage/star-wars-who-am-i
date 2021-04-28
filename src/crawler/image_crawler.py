from icrawler.builtin import GoogleImageCrawler
import yaml
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.absolute()


def download_images(config: dict) -> None:
    """
    Function to download images from google

    Args:
        config (dict): params for downloading
    """

    for keyword in config["keywords"]:
        google_crawler = GoogleImageCrawler(
            feeder_threads=2,
            parser_threads=4,
            downloader_threads=10,
            storage={"root_dir": f"project_data/images/{keyword}"},
        )
        google_crawler.crawl(
            keyword=f"{keyword} star wars",
            max_num=config["max_num"],
            filters={"date": ((1990, 1, 1), (2021, 12, 31))},
        )


if __name__ == "__main__":
    config = yaml.safe_load(open(SCRIPT_PATH / "download_images_cfg.yaml"))
    download_images(config)
