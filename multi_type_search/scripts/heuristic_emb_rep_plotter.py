import os
import json
import numpy as np
import matplotlib.pyplot as plt


# Load json files
def load_json(model, dataset):
    file_path = f"../output/heuristic_benchmarks/4_14/rep/{model}/{dataset}/output/scores.json"
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


models = ["gpt3_raw", "gpt3_trained", "simcse"]
datasets = ["ebt2", "morals"]
keys = ["Random", "Partially Random", "Annotation", "Deduction", "Agreement"]
colors = {"gpt3_raw": "#007AFF", "gpt3_trained": "#FFC857", "simcse": "#FF6E40"}
labels = {"gpt3_raw": "GPT3", "gpt3_trained": "GPT3-tuned", "simcse": "SimCSE"}
key_labels = {"Random": "Random", "Partially Random": "Partial", "Annotation": "Gold", "Deduction": "Model", "Agreement": "G. to S."}

# Merge jsons together by the model
merged_data = {model: {key: [] for key in keys} for model in models}

for model in models:
    for dataset in datasets:
        data = load_json(model, dataset)
        for key in keys:
            merged_data[model][key].extend(data[key])

# Create violin plot using matplotlib
fig, ax = plt.subplots()

positions = list(range(len(keys)))
width = 0.3
offset = width

for idx, (model, color) in enumerate(colors.items()):
    plot_data = [merged_data[model][key] for key in keys]
    plot_positions = [pos + (-offset + offset * idx) for pos in positions]
    vp = ax.violinplot(plot_data, plot_positions, widths=width, showmedians=True, showextrema=False)

    for pc in vp['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(1.0)

    if 'cmedians' in vp:
        vp['cmedians'].set_color('black')

    ax.plot([], [], color=color, label=labels[model])

ax.set_xticks(positions)
ax.set_xticklabels(list(key_labels.values()))
ax.set_xlabel("Distribution Type")
ax.set_ylabel("Cosine Similarity Scores")
plt.legend(loc='lower right')
plt.title("Cosine Similarity Test For Additive Deduction")
# plt.figure(dpi=1200)
# plt.show()
fig1 = plt.gcf()
fig1.savefig('./_embdist.png', dpi=300)