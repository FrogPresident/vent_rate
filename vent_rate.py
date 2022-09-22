from pathlib import Path

import cv2
import argparse

import numpy as np
from tqdm import tqdm
from typing import List

vent_color_check = 150


def main():
    args = get_arg_parse().parse_args()
    source: Path = args.source
    source_base_dir: Path
    sources: List[Path]
    out: Path = args.out
    if source.is_dir():
        sources = sorted(source.glob("**/*.*"))
        source_base_dir = source
    elif source.is_file():
        sources = [source]
        source_base_dir = source.parent
    else:
        raise FileNotFoundError

    for file_path in tqdm(sources):
        out_file_path = out / file_path.relative_to(source_base_dir)
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        print(processing_img(img))


def processing_img(img: np.ndarray) -> int:
    x, y = img.shape
    cut_picture = img[int(x * 0.35):int(x * 0.75), int(y * 0.2):int(y * 0.8)]
    vent_area = np.sum(cut_picture > vent_color_check)
    return vent_area


def get_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    return parser


if __name__ == '__main__':
    main()
