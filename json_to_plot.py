import argparse
import json
from pathlib import Path
from matplotlib import pyplot as plt


def main():
    args = get_arg_parse().parse_args()
    source: Path = args.src
    add_up = json_to_plot(source)
    vent_mean = []
    shrimp_mean = []
    ratio = []
    num = []
    for i in add_up:
        vent_mean.append(i[0])
        shrimp_mean.append(i[1])
        ratio.append(i[2])
    for i in range(len(add_up)):
        num.append(i)

    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, 1)
    plt.title("Comparison chart num")
    plt.ylabel("vent mean")
    plt.ylim(0, 150)
    ax2 = fig.add_subplot(3, 2, 3)
    plt.ylabel("shrimp mean")
    plt.ylim(0, 150)
    ax3 = fig.add_subplot(3, 2, 5)
    plt.ylabel("ratio mean")
    ax4 = fig.add_subplot(3, 2, 2)
    plt.title("Quantity histogram")
    plt.ylabel("vent mean")
    ax5 = fig.add_subplot(3, 2, 4)
    plt.ylabel("shrimp mean")
    ax6 = fig.add_subplot(3, 2, 6)
    plt.ylabel("ratio mean")
    ax1.plot(num, vent_mean)
    ax2.plot(num, shrimp_mean)
    ax3.plot(num, ratio)
    ax4.hist(vent_mean, rwidth=0.8, bins=[60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120])
    ax5.hist(shrimp_mean, rwidth=0.8, bins=[50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76])
    ax6.hist(ratio, rwidth=0.8, bins=[0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    print(vent_mean)
    plt.show()


def json_to_plot(source: Path):
    with source.open("r") as f:
        add_up = []
        json_data = json.load(f)
        for result in json_data:
            for i in result["results"]:
                if i is not None:
                    add_up.append((i["vent_mean"], i["shrimp_mean"], i["ratio"]))
    return add_up


def get_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    return parser


if __name__ == '__main__':
    main()
