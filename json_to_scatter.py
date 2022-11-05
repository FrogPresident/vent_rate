import argparse
import json
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import svm


def main():
    args = get_arg_parse().parse_args()
    source: Path = args.src
    add_up = json_to_plot(source)
    vent_mean = []
    shrimp_mean = []
    ratio = []
    x_test = [70, 74, 75, 70]
    num = []
    for i in add_up:
        vent_mean.append(i[0])
        shrimp_mean.append(i[1])
        ratio.append(i[2])
    X, y = datasets.load_iris(return_X_y=True)
    print(X.shape, y.shape)
    # rbfModel = svm.SVR(C=6, kernel='rbf', gamma='auto')
    # rbfModel.fit(shrimp_mean, vent_mean)
    # predicted = rbfModel.predict(x_test)
    # print(predicted)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title("scatter")
    plt.xlabel("shrimp mean")
    plt.ylabel("ratio")
    plt.scatter(shrimp_mean, ratio)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.scatter(shrimp_mean, vent_mean)
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
