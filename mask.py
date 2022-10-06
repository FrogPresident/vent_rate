import argparse
import json
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np


class ImgInfo:
    def __init__(self, h, w):
        self.h = h
        self.w = w


def main():
    args = get_arg_parse().parse_args()
    source: Path = args.polygon
    img_source: Path = args.image
    txt_source: Path = args.txt
    sources: List[Path]
    img_sources: List[Path]
    txt_sources: List[Path]
    source_base_dir: Path

    if source.is_dir():
        sources = sorted(source.glob("**/*.*"))
        img_sources = sorted(img_source.glob("**/*.*"))
        txt_sources = sorted(txt_source.glob("**/*.*"))
    elif source.is_file():
        sources = [source]
        img_sources = [img_source]
        txt_sources = [txt_source]
    else:
        raise FileNotFoundError

    out: Path = args.out

    data = []
    for file_path, img_path, txt_path in zip(sources, img_sources, txt_sources):
        results = []
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        all_vent = convertPolygonToMask(file_path)
        all_contour_list = get_all_contour_in_label(txt_path)
        for contour in all_contour_list:
            shrimp_mask = np.zeros_like(img)
            cv2.drawContours(shrimp_mask, [contour], -1, 255, -1)
            vent_mask = cv2.bitwise_and(shrimp_mask, all_vent)
            if not np.all(~(vent_mask > 0)):
                vent_mean = img[vent_mask > 0].mean()
                shrimp_mean = img[shrimp_mask > 0].mean()
                ratio = vent_mean / shrimp_mean
                print(vent_mean, shrimp_mean, ratio)
                results.append({"vent_mean": vent_mean, "shrimp_mean": shrimp_mean, "ratio": ratio})
            else:
                print("empty vent")
                results.append(None)

            if args.visualize:
                mixed_img = cv2.addWeighted(img, 0.7, shrimp_mask, 0.3, 0)
                mixed_img = cv2.addWeighted(mixed_img, 0.5, vent_mask, 0.5, 0)
                cv2.imshow("mask", mixed_img)
                if cv2.waitKey(0) & 255 == ord("q"):
                    exit()

        data.append({"filename": str(file_path), "results": results})

    if not args.visualize:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fp:
            json.dump(data, fp)

    cv2.destroyAllWindows()


def drawRotateLabel(label_txt: Path, img_info: ImgInfo):
    mask = np.zeros((img_info.h, img_info.w), np.uint8)
    cv2.drawContours(mask, get_all_contour_in_label(label_txt), -1, 255, -1)
    return mask


def get_all_contour_in_label(label_txt: Path) -> list[np.ndarray]:
    with open(label_txt) as f:
        return [
            to_boundary_contour(
                *[float(v) for v in line.split(' ')[0: 5]]
            )
            for line in f.readlines()
        ]


def to_boundary_contour(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """
    Get the boundary contour of a rotated box.

    :param cx: center x-coordinate of the rotated box
    :param cy: center y-coordinate of the rotated box
    :param w: width of the rotated box
    :param h: height of the rotated box
    :param angle: angle of the rotated box (in radian)
    :return: A Contour. (list of points of boundary of the rotated box)
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x0 = cx + 0.5 * w
    y0 = y1
    x2 = x1
    y2 = cy + 0.5 * h
    x3 = x0
    y3 = y2
    x0n = (x0 - cx) * cos_a - (y0 - cy) * sin_a + cx
    y0n = (x0 - cx) * sin_a + (y0 - cy) * cos_a + cy
    x1n = (x1 - cx) * cos_a - (y1 - cy) * sin_a + cx
    y1n = (x1 - cx) * sin_a + (y1 - cy) * cos_a + cy
    x2n = (x2 - cx) * cos_a - (y2 - cy) * sin_a + cx
    y2n = (x2 - cx) * sin_a + (y2 - cy) * cos_a + cy
    x3n = (x3 - cx) * cos_a - (y3 - cy) * sin_a + cx
    y3n = (x3 - cx) * sin_a + (y3 - cy) * cos_a + cy
    label_list = [[x0n, y0n], [x1n, y1n], [x2n, y2n], [x3n, y3n]]
    label_list = np.array(label_list, np.int32)
    return label_list


def convertPolygonToMask(poly_label_path: Path):
    with open(poly_label_path, "r", encoding='utf-8') as jsonf:
        json_data = json.load(jsonf)
        img_h = json_data["imageHeight"]
        img_w = json_data["imageWidth"]
        mask = np.zeros((img_h, img_w), np.uint8)
        num = 0
        for obj in json_data["shapes"]:
            polygon_points = obj["points"]
            polygon_points = np.array(polygon_points, np.int32)
            num += 1
            cv2.drawContours(mask, [polygon_points], -1, 255, -1)

    return mask


def get_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("polygon", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("txt", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("-v", "--visualize", action="store_true")
    return parser


if __name__ == '__main__':
    main()
