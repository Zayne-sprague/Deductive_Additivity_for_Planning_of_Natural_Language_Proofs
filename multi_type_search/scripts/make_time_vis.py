from multi_type_search.utils.paths import SEARCH_OUTPUT_FOLDER

from pathlib import Path
import json
from matplotlib import pyplot as plt

exp_folder = SEARCH_OUTPUT_FOLDER / 'timing'

exps = exp_folder.glob("*")
out_folder = Path('tmp/time_vis')
out_folder.mkdir(exist_ok=True, parents=True)

for exp in exps:
    try:
        exp_name = exp.name

        expansion_data = exp / 'visualizations/data/expansions.json'
        data = json.load(expansion_data.open('r'))
        times = list(sorted(data['Time to Proof']))

        total_solved = 0

        EXP_TIME = 30

        data = [sum([1 if y < x else 0 for y in times]) for x in range(EXP_TIME)]

        plt.plot(range(EXP_TIME), data)
        plt.title(exp_name)
        plt.ylim(0, 100)

        plt.savefig(str(out_folder / f'{exp_name}.png'))

        plt.show()

    except Exception:
        pass