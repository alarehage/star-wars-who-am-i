from pathlib import Path
from typing import Union, Tuple
import splitfolders
import yaml

SCRIPT_PATH = Path(__file__).parent.absolute()


def split(
    data_path: Union[Path, str], output_folder: str, ratio: Tuple[int, ...], seed: int
) -> None:
    """
    Splits images into train/val/test folders

    Args:
        data_path (Union[Path, str]): path to images
        output_folder (str): folder name to output split folders
        ratio (Tuple[int, ...]): train/val/test ratio
        seed (int): random seed
    """
    splitfolders.ratio(data_path, output_folder, ratio=ratio, seed=42)


if __name__ == "__main__":
    params = yaml.safe_load(open(SCRIPT_PATH / "utils.yaml"))["image_splitter"]
    DATA_PATH = params["data_path"]
    OUTPUT_FOLDER = params["output_folder"]
    RATIO = tuple(params["ratio"])
    SEED = params["seed"]

    split(DATA_PATH, OUTPUT_FOLDER, RATIO, SEED)
