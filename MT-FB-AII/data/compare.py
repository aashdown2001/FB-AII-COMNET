#!/usr/bin/env python3

import os
import json
import numpy as np
import re
from collections import defaultdict

# ===============================
# CONFIG (MATCHES PLOT SCRIPT)
# ===============================
BASE_DIR = "."
TARGET_SETS = ["low-target", "mid-target", "high-target", "var-target"]
ALGORITHMS = ["wpp", "rand", "heur", "unif"]

NETWORK_CONFIGS = [
    (2, 10),
    (5, 26),
    (10, 52),
]

BUDGET_MULTIPLIER = 0.45

# ===============================
# LOAD STATS
# ===============================
def load_stats(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    ue_devs = np.array(data["ue_throughputs"])
    jamming_power = np.array(data["jamming_powers"])
    avg_jammed_sinr = np.array(data["avg_jammed_path_loss"])
    power_per_ue = np.array(data["power_per_ue"])
    avg_prb_concentration = np.array(data["avg_prb_concentration"])

    avg_dev = ue_devs.mean(axis=0).mean()
    avg_power = jamming_power.mean()
    avg_sinr = np.nanmean(avg_jammed_sinr)
    avg_ue_concentration = power_per_ue.std(axis=1).mean()
    avg_prb_conc = np.nanmean(avg_prb_concentration)

    return avg_dev, avg_power, avg_sinr, avg_ue_concentration, avg_prb_conc


# ===============================
# LOAD ALL RESULTS
# ===============================
results = defaultdict(lambda: defaultdict(dict))

for folder in os.listdir(BASE_DIR):
    match = re.match(r"(\d+)ue_(\d+)prb", folder)
    if not match:
        continue

    ue = int(match.group(1))
    prb = int(match.group(2))
    folder_path = os.path.join(BASE_DIR, folder)

    for target in TARGET_SETS:
        for algo in ALGORITHMS:
            filepath = os.path.join(folder_path, f"average_{target}_{algo}.json")
            if not os.path.exists(filepath):
                continue

            results[target][algo][(ue, prb)] = load_stats(filepath)


# ===============================
# HELPERS
# ===============================
def compute_budget(ue, prb):
    return BUDGET_MULTIPLIER * prb


def get_stats(points, key):
    devs      = [d  for d, p, s, uc, pc in points[key]]
    powers    = [p  for d, p, s, uc, pc in points[key]]
    sinrs     = [s  for d, p, s, uc, pc in points[key]]
    ue_concs  = [uc for d, p, s, uc, pc in points[key]]
    prb_concs = [pc for d, p, s, uc, pc in points[key]]

    return (
        np.mean(devs),
        np.mean(powers),
        np.nanmean(sinrs),
        np.mean(ue_concs),
        np.nanmean(prb_concs)
    )


def summarize(label, points, budgets):
    avg_budget = np.mean(budgets) if budgets else 0.0

    rl_dev,   rl_pow,   rl_sinr,   rl_uc,   rl_pc   = get_stats(points, "wpp")
    rand_dev, rand_pow, rand_sinr, rand_uc, rand_pc  = get_stats(points, "rand")
    heur_dev, heur_pow, heur_sinr, heur_uc, heur_pc  = get_stats(points, "heur")
    unif_dev, unif_pow, unif_sinr, unif_uc, unif_pc  = get_stats(points, "unif")

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'RL':>8} {'RAND':>8} {'HEUR':>8} {'UNIF':>8}")
    print(f"  {'-'*70}")

    print(f"  {'Mean Dev':<30} {rl_dev:>8.3f} {rand_dev:>8.3f} {heur_dev:>8.3f} {unif_dev:>8.3f}")
    print(f"  {'Mean Power':<30} {rl_pow:>8.3f} {rand_pow:>8.3f} {heur_pow:>8.3f} {unif_pow:>8.3f}")
    print(f"  {'Avg Power Budget':<30} {avg_budget:>8.3f}")

    print(f"  {'Budget Usage RL':<30} {(rl_pow/avg_budget)*100:>7.1f}%")
    print(f"  {'Budget Usage RAND':<30} {(rand_pow/avg_budget)*100:>7.1f}%")
    print(f"  {'Budget Usage HEUR':<30} {(heur_pow/avg_budget)*100:>7.1f}%")
    print(f"  {'Budget Usage UNIF':<30} {(unif_pow/avg_budget)*100:>7.1f}%")

    print(f"  {'Power-Weighted SINR':<30} {rl_sinr:>8.3f} {rand_sinr:>8.3f} {heur_sinr:>8.3f} {unif_sinr:>8.3f}")
    print(f"  {'UE Power Concentration':<30} {rl_uc:>8.3f} {rand_uc:>8.3f} {heur_uc:>8.3f} {unif_uc:>8.3f}")
    print(f"  {'PRB Power Concentration':<30} {rl_pc:>8.3f} {rand_pc:>8.3f} {heur_pc:>8.3f} {unif_pc:>8.3f}")

    print("\n  DEV COMPARISON:")
    print(f"    RL vs RAND → {((rand_dev - rl_dev)/rand_dev)*100:.1f}% lower")
    print(f"    RL vs HEUR → {((heur_dev - rl_dev)/heur_dev)*100:.1f}% lower")
    print(f"    RL vs UNIF → {((unif_dev - rl_dev)/unif_dev)*100:.1f}% lower")

    print("\n  POWER COMPARISON:")
    print(f"    RL vs RAND → {((rand_pow - rl_pow)/rand_pow)*100:.1f}% lower")
    print(f"    RL vs HEUR → {((heur_pow - rl_pow)/heur_pow)*100:.1f}% lower")
    print(f"    RL vs UNIF → {((unif_pow - rl_pow)/unif_pow)*100:.1f}% lower")

    print("\n  PRB CONCENTRATION:")
    print(f"    RL vs RAND → {((rl_pc - rand_pc)/rand_pc)*100:.1f}% more concentrated")
    print(f"    RL vs HEUR → {((rl_pc - heur_pc)/heur_pc)*100:.1f}% more concentrated")
    print(f"    RL vs UNIF → {((rl_pc - unif_pc)/unif_pc)*100:.1f}% more concentrated")

    print("\n  Counts:")
    for algo in ALGORITHMS:
        print(f"    {algo}: {len(points[algo])}")


# ===============================
# COLLECTORS (MATCH PLOTS)
# ===============================
def collect_ue(results):
    pts, budgets = defaultdict(list), []
    for t in TARGET_SETS:
        for a in ALGORITHMS:
            for (ue, prb), v in results[t][a].items():
                if prb == 26:
                    pts[a].append(v)
                    if a == "wpp":
                        budgets.append(compute_budget(ue, prb))
    return pts, budgets


def collect_prb(results):
    pts, budgets = defaultdict(list), []
    for t in TARGET_SETS:
        for a in ALGORITHMS:
            for (ue, prb), v in results[t][a].items():
                if ue == 2:
                    pts[a].append(v)
                    if a == "wpp":
                        budgets.append(compute_budget(ue, prb))
    return pts, budgets


def collect_net(results):
    pts, budgets = defaultdict(list), []
    for t in TARGET_SETS:
        for a in ALGORITHMS:
            for (ue, prb), v in results[t][a].items():
                if (ue, prb) in NETWORK_CONFIGS:
                    pts[a].append(v)
                    if a == "wpp":
                        budgets.append(compute_budget(ue, prb))
    return pts, budgets


# ===============================
# RUN INDIVIDUAL SUMMARIES
# ===============================
p_ue,  b_ue  = collect_ue(results)
p_prb, b_prb = collect_prb(results)
p_net, b_net = collect_net(results)

summarize("UE SCALING (PRB=26)",  p_ue,  b_ue)
summarize("PRB SCALING (UE=2)",   p_prb, b_prb)
summarize("NETWORK SCALING",      p_net, b_net)


# ===============================
# GRAND AVERAGE — each (target, algo, ue, prb) counted exactly once
# ===============================

# Determine which (ue, prb) configs appeared in ANY of the three plot scenarios
def plot_configs(results):
    """Return the set of (ue, prb) configs that appear in at least one plot."""
    configs = set()
    for t in TARGET_SETS:
        for a in ALGORITHMS:
            for (ue, prb) in results[t][a].keys():
                if prb == 26 or ue == 2 or (ue, prb) in NETWORK_CONFIGS:
                    configs.add((ue, prb))
    return configs

valid_configs = plot_configs(results)

grand_points  = defaultdict(list)
grand_budgets = []
seen_budget_configs = set()  # to avoid duplicate budget entries across algos

for t in TARGET_SETS:
    for a in ALGORITHMS:
        for (ue, prb), v in results[t][a].items():
            if (ue, prb) in valid_configs:
                grand_points[a].append(v)
                if (t, ue, prb) not in seen_budget_configs:
                    grand_budgets.append(compute_budget(ue, prb))
                    seen_budget_configs.add((t, ue, prb))

summarize("GRAND AVERAGE (all plot configs, no double-counting)", grand_points, grand_budgets)

