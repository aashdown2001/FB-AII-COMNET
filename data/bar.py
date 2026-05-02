import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Formatter function: strip decimals
integer_formatter = FuncFormatter(lambda x, _: f"{int(x)}")

# -------- USER INPUT -------- #
json_files = [
    # ----- TRAINED (Row 1) -----
    "5ue_26prb/average_low-target_wpp.json",
    "5ue_26prb/average_mid-target_wpp.json",
    "5ue_26prb/average_high-target_wpp.json",
    "5ue_26prb/average_var-target_wpp.json",
    # ----- RANDOM (Row 2) -----
    "5ue_26prb/average_low-target_rand.json",
    "5ue_26prb/average_mid-target_rand.json",
    "5ue_26prb/average_high-target_rand.json",
    "5ue_26prb/average_var-target_rand.json",
    # ----- HEURISTIC (Row 3) -----
    "5ue_26prb/average_low-target_heur.json",
    "5ue_26prb/average_mid-target_heur.json",
    "5ue_26prb/average_high-target_heur.json",
    "5ue_26prb/average_var-target_heur.json",
]

subplot_titles = [
    # Row 1
    "Low Target RL", "Mid Target RL", "High Target RL", "Var Target RL",
    # Row 2
    "Low Target Random", "Mid Target Random", "High Target Random", "Var Target Random",
    # Row 3
    "Low Target Heuristic", "Mid Target Heuristic", "High Target Heuristic", "Var Target Heuristic",
]

# -------- COLORS (extendable palette) -------- #
BAR_COLOR_POOL = [
    "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
    "#A6A6A6", "#457B9D", "#A8DADC", "#1D3557", "#6D6875",
    "#B5838D", "#6B9080", "#F2CC8F", "#81B29A", "#F4F1DE",
]

# -------- FUNCTIONS -------- #
def load_stats(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    ue_devs = np.array(data["ue_throughputs"])
    jamming_power = np.array(data["jamming_powers"])
    avg_dev = ue_devs.mean(axis=0)
    std_dev = ue_devs.std(axis=0)
    avg_power = jamming_power.mean() 
    return avg_dev, std_dev, avg_power

# -------- PLOTTING -------- #
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'lines.linewidth': 2.5,
    'axes.linewidth': 2.0
})

# Load first file to determine number of UEs dynamically
_sample_avg, _, _ = load_stats(json_files[0])
n_ues = len(_sample_avg)

# Dynamically build labels, x positions, and colors
labels = [f"UE {i+1}" for i in range(n_ues)]
x = np.arange(n_ues)
bar_colors = BAR_COLOR_POOL[:n_ues]  # slice to however many UEs exist

fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=False, sharey=False)
plt.subplots_adjust(left=0.043, right=0.993, top=0.94, bottom=0.08,
                    hspace=0.35, wspace=0.13)
axes = axes.flatten()

for i, filepath in enumerate(json_files):
    avg_dev, std_dev, avg_power = load_stats(filepath)

    # Re-derive per subplot in case files ever have different UE counts
    n = len(avg_dev)
    xi = np.arange(n)
    colors = BAR_COLOR_POOL[:n]

    std_dev_filtered = np.where(std_dev < 0.02 * avg_dev, 0, std_dev)

    ax = axes[i]
    ax.bar(xi, avg_dev, yerr=std_dev_filtered, capsize=3,
           color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xticks(xi)
    ax.set_xticklabels(np.arange(1, n + 1))
    ax.grid(True, axis='both', which='major', linestyle='--', alpha=0.6)
    ax.set_title(subplot_titles[i])
    ax.yaxis.set_major_formatter(integer_formatter)
    ax.annotate(
        f"Avg Dev: {avg_dev.mean():.2f} Mbps\nTx Power: {avg_power:.2f} units",
        xy=(0.96, 0.3), xycoords='axes fraction',
        ha='right', va='top', fontsize=15, fontweight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='black', linewidth=2.0)
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

# Shared axis labels — UE count pulled dynamically
fig.text(0.523, 0.018,
         f"Target UE Index Number ({n_ues} Total Target UEs and 26 Total PRBs)",
         ha='center', fontsize=20)
fig.text(0.001, 0.5,
         "Average Target UE Throughput Deviation (Mbps)",
         va='center', rotation='vertical', fontsize=22)

fig.savefig("mt-fb-aii-bar.png", format="png", dpi=300, bbox_inches="tight")
plt.show()
