from argparse import ArgumentParser
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--input_files', '-i', nargs='+', help='json files with scores per question.')

    args = parser.parse_args()

    input_files = [Path(x) for x in args.input_files]

    all_scores = []
    for x in input_files:
        with x.open('r') as f:
            data = json.load(f)
            scores = [y for x in data['scores'] for y in x]
            all_scores.append(np.array(scores))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.violinplot(all_scores)

    plt.savefig('./tmp.png')
    plt.show()
