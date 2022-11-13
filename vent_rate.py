import json
from pathlib import Path

import cv2
import argparse

import numpy
import numpy as np
from tqdm import tqdm
from typing import List

vent_color_check = 150


def main():
    args = get_arg_parse().parse_args()
    source: Path = args.source
    # out: Path = args.out
    json_to_img(source)


def json_to_img(source: Path):
    with source.open("r") as f:
        json_data = json.load(f)
        for result in json_data:
            img = cv2.imread(str(result["img_path"]))
            for i in result["boundary_contours"]:
                vent_mask = np.zeros_like(img)
                shrimp_mask = np.zeros_like(img)
                vent_contours = numpy.array(i)
                shrimp_contours = vent_contours.copy()
                vent_contours = numpy.array(segment_img(list(vent_contours)))
                # Draw contours
                cv2.drawContours(vent_mask, [vent_contours], -1, (255, 255, 255), -1)
                cv2.drawContours(shrimp_mask, [shrimp_contours], -1, (255, 255, 255), -1)
                mixed_img = cv2.addWeighted(img, 0.9, shrimp_mask, 0.1, -1)
                cv2.imshow("img", mixed_img)
                cv2.waitKey()
                vent_rate = img[vent_mask > 0].mean()
                shrimp_rate = img[shrimp_mask > 0].mean()
                print(f"{vent_rate},vent rate")
                print(f"{shrimp_rate},shrimp rate")
                detect_vent(vent_rate, shrimp_rate)


# def processing_img(img: np.ndarray) -> int:
#     x, y = img.shape
#     cut_picture = img[int(x * 0.35):int(x * 0.75), int(y * 0.2):int(y * 0.8)]
#     vent_area = np.sum(cut_picture > vent_color_check)
#     return vent_area

def segment_img(vent_contours):
    x_vector = vent_contours[1][0] - vent_contours[2][0]
    y_vector = vent_contours[1][1] - vent_contours[2][1]
    # 上下壓縮
    vent_contours[0][0] = vent_contours[0][0] - x_vector * 0.2
    vent_contours[0][1] = vent_contours[0][1] - y_vector * 0.2
    vent_contours[1][0] = vent_contours[1][0] - x_vector * 0.2
    vent_contours[1][1] = vent_contours[1][1] - y_vector * 0.2
    vent_contours[2][0] = vent_contours[2][0] + x_vector * 0.3
    vent_contours[2][1] = vent_contours[2][1] + y_vector * 0.3
    vent_contours[3][0] = vent_contours[3][0] + x_vector * 0.3
    vent_contours[3][1] = vent_contours[3][1] + y_vector * 0.3
    # 左右壓縮
    y_x_vector = vent_contours[1][0] - vent_contours[0][0]
    y_y_vector = vent_contours[1][1] - vent_contours[0][1]
    vent_contours[0][0] = vent_contours[0][0] + y_x_vector * 0.3
    vent_contours[0][1] = vent_contours[0][1] + y_y_vector * 0.3
    vent_contours[1][0] = vent_contours[1][0] - y_x_vector * 0.3
    vent_contours[1][1] = vent_contours[1][1] - y_y_vector * 0.3
    vent_contours[2][0] = vent_contours[2][0] - y_x_vector * 0.3
    vent_contours[2][1] = vent_contours[2][1] - y_y_vector * 0.3
    vent_contours[3][0] = vent_contours[3][0] + y_x_vector * 0.3
    vent_contours[3][1] = vent_contours[3][1] + y_y_vector * 0.3
    return vent_contours


def detect_vent(vent_rate: float, shrimp_rate: float):
    if shrimp_rate * 1.1 < vent_rate:
        print("it's good")


def convert_contour(file_label: Path):
    with open(file_label, "r") as fp:
        print(fp.readlines()[0])


def get_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    return parser


if __name__ == '__main__':
    main()
