import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# ===============================
# CONFIG
# ===============================
BASE_DIR = "."
TARGET_SETS = ["low-target", "mid-target", "high-target", "var-target"]
ALGORITHMS = ["wpp", "rand", "heur"]

ALGO_COLORS = {
    "wpp": "#2A9D8F",
    "rand": "#E76F51",
    "heur": "#264653"
}

# ===============================
# LOAD STATS
# ===============================
def load_stats(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    ue_devs = np.array(data["ue_throughputs"])
    jamming_power = np.array(data["jamming_powers"])

    avg_dev = ue_devs.mean(axis=0).mean()
    avg_power = jamming_power.mean() 

    return avg_dev, avg_power


# ===============================
# SCAN ALL FOLDERS
# ===============================
results = defaultdict(lambda: defaultdict(dict))

for folder in os.listdir(BASE_DIR):

    match = re.match(r"(\d+)ue_(\d+)prb", folder)
    if not match:
        continue

    ue_count = int(match.group(1))
    prb_count = int(match.group(2))
    folder_path = os.path.join(BASE_DIR, folder)

    for target in TARGET_SETS:
        for algo in ALGORITHMS:

            filename = f"average_{target}_{algo}.json"
            filepath = os.path.join(folder_path, filename)

            if not os.path.exists(filepath):
                continue

            dev, power = load_stats(filepath)
            results[target][algo][(ue_count, prb_count)] = (dev, power)


# ===============================
# COMMON PLOTTING FUNCTION
# ===============================
def plot_summary(x_values_dict, xlabel, filename):

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), sharex=False)
    axes = axes.flatten()

    first_ax2 = None

    for ax, target in zip(axes, TARGET_SETS):

        ax2 = ax.twinx()
        ax2.set_ylabel("Interference Power", fontsize=13.5)
        ax2.tick_params(axis='y', width=1.8)

        if first_ax2 is None:
            first_ax2 = ax2

        for algo in ALGORITHMS:

            if algo not in x_values_dict[target]:
                continue

            color = ALGO_COLORS[algo]

            x_vals = sorted(x_values_dict[target][algo].keys())
            devs = [x_values_dict[target][algo][x][0] for x in x_vals]
            powers = [x_values_dict[target][algo][x][1] for x in x_vals]

            # Throughput deviation (solid) on primary axis
            ax.plot(x_vals, devs,
                    linestyle='-',
                    color=color,
                    marker='o',
                    linewidth=2.5,
                    markersize=7,
                    label=f"{'RL' if algo == 'wpp' else algo.upper()} Dev")

            # Power (dashed) on secondary axis
            ax2.plot(x_vals, powers,
                    linestyle='--',
                    color=color,
                    marker='o',
                    linewidth=2.5,
                    markersize=7,
                    label=f"{'RL' if algo == 'wpp' else algo.upper()} Power")

        # Title styling
        ax.set_title(target.replace("-", " ").title(),
                     fontsize=12,
                     fontweight='bold')

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Throughput Deviation", fontsize=13.5)
        ax.grid(True, linestyle="--", alpha=0.6)

        for spine in ax.spines.values():
            spine.set_linewidth(2.3)

        ax.tick_params(axis='both', width=1.8)

    # Collect handles from both axes of first subplot for legend
    handles1, labels1 = axes[0].get_legend_handles_labels()  # RL Dev, RAND Dev, HEUR Dev
    handles2, labels2 = first_ax2.get_legend_handles_labels()  # RL Power, RAND Power, HEUR Power

    print("labels1:", labels1)
    print("labels2:", labels2)

    ordered_handles = [handles1[0], handles2[0],
                       handles1[1], handles2[1],
                       handles1[2], handles2[2]]
    ordered_labels  = [labels1[0],  labels2[0],
                       labels1[1],  labels2[1],
                       labels1[2],  labels2[2]]

    fig.legend(ordered_handles, ordered_labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.985),
               ncol=3,
               fontsize=11,
               handlelength=4)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename}")

# ===============================
# 1️⃣ UE SCALING (Fix PRB=52)
# ===============================
ue_scaling = defaultdict(lambda: defaultdict(dict))

for target in TARGET_SETS:
    for algo in ALGORITHMS:
        for (ue, prb), values in results[target][algo].items():
            if prb == 26:
                ue_scaling[target][algo][ue] = values

plot_summary(ue_scaling,
             xlabel="Number of Target UEs",
             filename="summary_ue_scaling.png")


# ===============================
# 2️⃣ PRB SCALING (Fix UE=5)
# ===============================
prb_scaling = defaultdict(lambda: defaultdict(dict))

for target in TARGET_SETS:
    for algo in ALGORITHMS:
        for (ue, prb), values in results[target][algo].items():
            if ue == 2:
                prb_scaling[target][algo][prb] = values

plot_summary(prb_scaling,
             xlabel="Number of PRBs",
             filename="summary_prb_scaling.png")


# ===============================
# 3️⃣ NETWORK SIZE SCALING
# ===============================
network_scaling = defaultdict(lambda: defaultdict(dict))

NETWORK_CONFIGS = [
    (2, 10),
    (5, 26),
    (10, 52),
]

for target in TARGET_SETS:
    for algo in ALGORITHMS:
        for (ue, prb), values in results[target][algo].items():

            if (ue, prb) in NETWORK_CONFIGS:

                # Define a meaningful size metric
                size_metric = ue * 10   # recommended

                network_scaling[target][algo][size_metric] = values

plot_summary(network_scaling,
             xlabel="Network Size Percentage (%)",
             filename="summary_network_percentage_scaling.png")

