import argparse
import json
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm


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
        source_base_dir = source
    elif source.is_file():
        sources = [source]
        img_sources = [img_source]
        txt_sources = [txt_source]
        source_base_dir = source.parent
    else:
        raise FileNotFoundError

    for file_path, img_path, txt_path in tqdm(zip(sources, img_sources, txt_sources)):
        mask = convertPolygonToMask(file_path)
        _, th = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mix_img = cv2.addWeighted(img, 0.8, th, 0.2, 0)
        drawRotateLabel(mix_img, txt_path)
        cv2.imshow("mask", mix_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def drawRotateLabel(img, txtPath: Path):
    with open(txtPath) as f:
        for line in f.readlines():
            s = line.split(' ')
            cx = float(s[0])
            cy = float(s[1])
            h = float(s[2])
            w = float(s[3])
            angle = float(s[4])
            angle = angle - 1.57
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x0 = cx + 0.5 * w
            y0 = y1
            x2 = x1
            y2 = cy + 0.5 * h
            x3 = x0
            y3 = y2
            x0n = (x0 - cx) * cosA - (y0 - cy) * sinA + cx
            y0n = (x0 - cx) * sinA + (y0 - cy) * cosA + cy
            x1n = (x1 - cx) * cosA - (y1 - cy) * sinA + cx
            y1n = (x1 - cx) * sinA + (y1 - cy) * cosA + cy
            x2n = (x2 - cx) * cosA - (y2 - cy) * sinA + cx
            y2n = (x2 - cx) * sinA + (y2 - cy) * cosA + cy
            x3n = (x3 - cx) * cosA - (y3 - cy) * sinA + cx
            y3n = (x3 - cx) * sinA + (y3 - cy) * cosA + cy
            label_list = [[x0n, y0n], [x1n, y1n], [x2n, y2n], [x3n, y3n]]
            label_list = np.array(label_list, np.int32)
            cv2.drawContours(img, [label_list], -1, 255, 1)


def convertPolygonToMask(jsonfilePath: Path):
    with open(jsonfilePath, "r", encoding='utf-8') as jsonf:
        jsonData = json.load(jsonf)
        img_h = jsonData["imageHeight"]
        img_w = jsonData["imageWidth"]
        mask = np.zeros((img_h, img_w), np.uint8)
        num = 0
        for obj in jsonData["shapes"]:
            label = obj["label"]
            polygonPoints = obj["points"]
            polygonPoints = np.array(polygonPoints, np.int32)
            # print("+" * 50, "\n", polygonPoints)
            # print(label)
            num += 1
            cv2.drawContours(mask, [polygonPoints], -1, 255, -1)

    return mask


def get_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("polygon", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("txt", type=Path)
    return parser


if __name__ == '__main__':
    main()
