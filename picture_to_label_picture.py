import argparse
import json
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np


def main():
    args = get_arg_parse().parse_args()
    img_source: Path = args.image
    txt_source: Path = args.txt
    img_sources: List[Path]
    txt_sources: List[Path]
    # out: Path = args.out

    if img_source.is_dir():
        img_sources = sorted(img_source.glob("**/*.*"))
        txt_sources = sorted(txt_source.glob("**/*.*"))
    elif img_source.is_file():
        img_sources = [img_source]
        txt_sources = [txt_source]
    else:
        raise FileNotFoundError
    for img_path, txt_path in zip(img_sources, txt_sources):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        all_yolo_contour_list = get_all_yolo_contour_in_label(txt_path)
        for contour in all_yolo_contour_list:
            print(contour)
            x1 = int(contour[1] - 0.5 * contour[3])
            y1 = int(contour[0] - 0.5 * contour[2])
            x4 = x1 + contour[3]
            y4 = y1 + contour[2]
            print(x1, y1, x4, y4)
            img123 = img[x1:x4, y1:y4]
            cv2.imshow("img", img123)
            cv2.waitKey()


def get_all_yolo_contour_in_label(label_txt: Path) -> list[np.ndarray]:
    with open(label_txt) as f:
        return [
            to_boundary_yolo_contour(
                *[float(v) for v in line.split(' ')[1: 5]]
            )
            for line in f.readlines()
        ]


def to_boundary_yolo_contour(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    label_list = [cx, cy, w, h]
    label_list = np.array(label_list, np.int32)
    return label_list


def get_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("txt", type=Path)
    # parser.add_argument("out", type=Path)
    return parser


if __name__ == '__main__':
    main()
