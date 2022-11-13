from pathlib import Path
import argparse

import cv2
import numpy as np


def main():
    args = get_args_parse().parse_args()
    source: Path = args.source
    img = cv2.imread(str(source))
    mask = np.zeros_like(img)
    contours = np.array([[248, 183],
                         [201, 160],
                         [120, 322],
                         [167, 346]
                         ])
    original = contours.copy()
    # 上下壓縮
    x_x_vector = contours[1][0] - contours[2][0]
    x_y_vector = contours[1][1] - contours[2][1]
    contours[0][0] = contours[0][0] - x_x_vector * 0.35
    contours[0][1] = contours[0][1] - x_y_vector * 0.35
    contours[1][0] = contours[1][0] - x_x_vector * 0.35
    contours[1][1] = contours[1][1] - x_y_vector * 0.35
    contours[2][0] = contours[2][0] + x_x_vector * 0.3
    contours[2][1] = contours[2][1] + x_y_vector * 0.3
    contours[3][0] = contours[3][0] + x_x_vector * 0.3
    contours[3][1] = contours[3][1] + x_y_vector * 0.3
    # 左右壓縮
    y_x_vector = contours[1][0]-contours[0][0]
    y_y_vector = contours[1][1]-contours[0][1]
    contours[0][0] = contours[0][0] + y_x_vector * 0.3
    contours[0][1] = contours[0][1] + y_y_vector * 0.3
    contours[1][0] = contours[1][0] - y_x_vector * 0.3
    contours[1][1] = contours[1][1] - y_y_vector * 0.3
    contours[2][0] = contours[2][0] - y_x_vector * 0.3
    contours[2][1] = contours[2][1] - y_y_vector * 0.3
    contours[3][0] = contours[3][0] + y_x_vector * 0.3
    contours[3][1] = contours[3][1] + y_y_vector * 0.3

    cv2.drawContours(mask, [contours], -1, (255, 255, 255), -1)
    mixed_img = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
    # print(img[mask > 0].mean())
    print(original, contours)
    cv2.imshow("img", mixed_img)
    cv2.waitKey()


def get_args_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    return parser


if __name__ == '__main__':
    main()
