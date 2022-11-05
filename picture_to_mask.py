import argparse
import json
import math
from pathlib import Path
from typing import List, Iterable

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
    yolo_txt_source: Path = args.yolo_txt
    sources: List[Path]
    img_sources: List[Path]
    txt_sources: List[Path]
    source_base_dir: Path

    if source.is_dir():
        sources = sorted(source.glob("**/*.*"), key=str_path_to_int)
        img_sources = sorted(img_source.glob("**/*.*"), key=str_path_to_int)
        txt_sources = sorted(txt_source.glob("**/*.*"), key=str_path_to_int)
        yolo_txt_sources = sorted(yolo_txt_source.glob("**/*.*"), key=str_path_to_int)
    elif source.is_file():
        sources = [source]
        img_sources = [img_source]
        txt_sources = [txt_source]
        yolo_txt_sources = [yolo_txt_source]
    else:
        raise FileNotFoundError

    out_base = Path("shrimp_out")
    img_base_path = out_base / "imgs"
    mask_base_path = out_base / "masks"

    for p in [img_base_path, mask_base_path]:
        p.mkdir(parents=True, exist_ok=True)

    data = []
    for file_path, img_path, txt_path, yolo_txt_path in zip(sources, img_sources, txt_sources, yolo_txt_sources):
        print(f"{file_path=}")
        results = []
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        all_vent = convertPolygonToMask(file_path)
        all_contour_list = get_all_contour_in_label(txt_path)
        all_yolo_contour_list = get_all_yolo_contour_in_label(yolo_txt_path)
        for i, (contour, yolo_contour) in enumerate(zip(all_contour_list, all_yolo_contour_list)):
            print(f"{i=}")
            shrimp_mask = np.zeros_like(img)
            cv2.drawContours(shrimp_mask, [contour], -1, 1, -1)
            vent_mask = cv2.bitwise_and(shrimp_mask, all_vent)
            xc, yc, w, h = yolo_contour
            x1 = int(xc - 0.5 * w)
            y1 = int(yc - 0.5 * h)
            x2 = x1 + w
            y2 = y1 + h
            print([img.shape[1], img.shape[0]], end="|")
            print(x1, y1, x2, y2, end="|")
            x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, [img.shape[1], img.shape[0]] * 2)
            print(x1, y1, x2, y2)
            out_img = img[y1: y2, x1: x2]
            out_mask = vent_mask[y1: y2, x1: x2]
            cv2.imwrite(str(img_base_path / f"{txt_path.stem}_{i}.png"), out_img)
            cv2.imwrite(str(mask_base_path / f"{txt_path.stem}_{i}.png"), out_mask)
            if not np.all(~(vent_mask > 0)):
                vent_mean = img[vent_mask > 0].mean()
                shrimp_mean = img[shrimp_mask > 0].mean()
                ratio = vent_mean / shrimp_mean
                results.append({"vent_mean": vent_mean, "shrimp_mean": shrimp_mean, "ratio": ratio})
            else:
                results.append(None)

            if args.visualize:
                mixed_img = cv2.addWeighted(img, 0.7, shrimp_mask, 0.3, 0)
                mixed_img = cv2.addWeighted(mixed_img, 0.5, vent_mask, 0.5, 0)
                cv2.imshow("mask", mixed_img)
                if cv2.waitKey(0) & 255 == ord("q"):
                    exit()

        data.append({"filename": str(file_path), "results": results})

    cv2.destroyAllWindows()


def str_path_to_int(p: Path):
    return int(p.stem)


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
    parser.add_argument("yolo_txt", type=Path)
    parser.add_argument("-v", "--visualize", action="store_true")
    return parser


if __name__ == '__main__':
    main()
