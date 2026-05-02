import json
import glob
import numpy as np
import os

BASE_DIR = ""
UE_DIRS = [
    "2ue_10prb",
    "2ue_52prb",
    "2ue_5prb",
    "2ue_24prb",
    "2ue_26prb",
    "2ue_38prb",
    "5ue_26prb",
    "10ue_52prb",
    "10ue_26prb",
    "5ue_5prb",
    "5ue_10prb",
    "5ue_24prb",
    "5ue_38prb",
    "5ue_52prb",
]
TARGET_GROUPS = [
    "low-target",
    "mid-target",
    "high-target",
    "var-target"
]
POLICIES = [
    "wpp",
    "rand",
    "heur",
    "unif",
]

for ue_dir in UE_DIRS:
    for target_group in TARGET_GROUPS:
        for policy in POLICIES:
            pattern = os.path.join(
                BASE_DIR,
                ue_dir,
                f"{target_group}-{policy}",
                "*.json"
            )
            files = glob.glob(pattern)
            if len(files) == 0:
                continue
            print(f"Averaging {pattern} ({len(files)} files)")

            ue_throughputs_list = []
            jamming_powers_list = []
            avg_jammed_sinr_list = []
            power_per_ue_list = []
            avg_prb_concentration_list = []

            for f in files:
                with open(f, "r") as infile:
                    data = json.load(infile)
                    ue_throughputs_list.append(np.array(data["ue_throughputs"]))
                    jamming_powers_list.append(np.array(data["jamming_powers"]))
                    avg_jammed_sinr_list.append(np.array(data["avg_jammed_path_loss"]))
                    power_per_ue_list.append(np.array(data["power_per_ue"]))
                    avg_prb_concentration_list.append(np.array(data["avg_prb_concentration"]))

            ue_throughputs_avg = np.mean(np.stack(ue_throughputs_list), axis=0).tolist()
            jamming_powers_avg = np.mean(np.stack(jamming_powers_list), axis=0).tolist()
            avg_jammed_sinr_avg = np.nanmean(np.stack(avg_jammed_sinr_list), axis=0).tolist()
            power_per_ue_avg = np.nanmean(np.stack(power_per_ue_list), axis=0).tolist()
            avg_prb_concentration_avg = np.nanmean(np.stack(avg_prb_concentration_list), axis=0).tolist()

            averaged_data = {
                "ue_throughputs": ue_throughputs_avg,
                "jamming_powers": jamming_powers_avg,
                "avg_jammed_path_loss": avg_jammed_sinr_avg,
                "power_per_ue": power_per_ue_avg,
                "avg_prb_concentration": avg_prb_concentration_avg
            }

            output_file = os.path.join(
                BASE_DIR,
                ue_dir,
                f"average_{target_group}_{policy}.json"
            )
            with open(output_file, "w") as outfile:
                json.dump(averaged_data, outfile, indent=2)

print("ALL AVERAGING COMPLETE")

