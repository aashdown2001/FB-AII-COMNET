import os
import json
import numpy as np
import random
import torch
from stable_baselines3 import PPO
from jamming_env import JammingEnv

# =======================
# GLOBAL SETTINGS
# =======================
MODEL_PATH = "ppo_jammer_model_randomized_120M-8FIXED.zip"
BASE_OUTPUT_DIR = "/workspace/FB-AII-DATA/4retrained/minor-rev-data/"
EPISODES = 100
SEED = 999

# =======================
# REPRODUCIBILITY
# =======================
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# =======================
# EXPERIMENT CONFIGURATIONS
# =======================
EXPERIMENT_CONFIGS = [
    {"dir": "2ue_10prb", "active_ues": 2, "usable_prbs": 10},
    {"dir": "2ue_52prb", "active_ues": 2, "usable_prbs": 52},
    {"dir": "2ue_5prb", "active_ues": 2, "usable_prbs": 5},
    {"dir": "2ue_24prb", "active_ues": 2, "usable_prbs": 24},
    {"dir": "2ue_26prb", "active_ues": 2, "usable_prbs": 26},
    {"dir": "5ue_26prb", "active_ues": 5, "usable_prbs": 26},
    {"dir": "10ue_26prb", "active_ues": 10, "usable_prbs": 26},
    {"dir": "2ue_38prb", "active_ues": 2, "usable_prbs": 38},
    {"dir": "10ue_52prb", "active_ues": 10, "usable_prbs": 52},
    {"dir": "5ue_5prb", "active_ues": 5, "usable_prbs": 5},
    {"dir": "5ue_10prb", "active_ues": 5, "usable_prbs": 10},
    {"dir": "5ue_24prb", "active_ues": 5, "usable_prbs": 24},
    {"dir": "5ue_38prb", "active_ues": 5, "usable_prbs": 38},
    {"dir": "5ue_52prb", "active_ues": 5, "usable_prbs": 52},
]

# ============================================================
# SEED GENERATION
# ============================================================
rng = np.random.RandomState(SEED)
EPISODE_SEEDS = rng.randint(0, 2**31 - 1, size=EPISODES).tolist()


def reset_all_rngs(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# HELPER: extract current PRB mask from observation
# ============================================================
def get_current_prb_mask(obs, num_ues, num_prbs):
    curr_prb_flat = obs[2*num_ues + num_prbs + num_ues*num_prbs:]
    return curr_prb_flat.reshape(num_ues, num_prbs)


# ============================================================
# HEURISTIC POLICIES
# ============================================================

class MomentumHeuristic:
    def __init__(self, active_ues, num_prbs, budget):
        self.active_ues = active_ues
        self.num_prbs = num_prbs
        self.budget = budget
        self.alpha = 0.3
        self.reset()

    def reset(self):
        self.effectiveness = np.ones(self.active_ues)
        self.prev_tput = None
        self.prev_power_per_ue = None

    def __call__(self, obs, env, target_throughputs):
        num_ues = env.num_ues
        num_prbs = env.num_prbs_total
        budget = self.budget

        current_throughputs = obs[num_ues:2*num_ues]
        last_action = obs[2*num_ues:2*num_ues + num_prbs]
        last_prb_flat = obs[2*num_ues + num_prbs:2*num_ues + num_prbs + num_ues*num_prbs]
        last_prb_mask = last_prb_flat.reshape(num_ues, num_prbs)

        # Read current PRB mask from observation
        prb_mask = get_current_prb_mask(obs, num_ues, num_prbs)

        targets = np.array(target_throughputs[:self.active_ues])
        last_prb_counts = np.array([last_prb_mask[ue].sum() for ue in range(self.active_ues)])
        curr_prb_counts = np.array([prb_mask[ue].sum() for ue in range(self.active_ues)])

        last_power_per_ue = np.array([
            last_action[np.where(last_prb_mask[ue] > 0)[0]].sum()
            if last_prb_mask[ue].sum() > 0 else 0.0
            for ue in range(self.active_ues)
        ])

        if self.prev_tput is not None and self.prev_power_per_ue is not None:
            for ue in range(self.active_ues):
                if self.prev_power_per_ue[ue] > 0.01:
                    tput_drop = max(0.0, self.prev_tput[ue] - current_throughputs[ue])
                    new_effectiveness = tput_drop / self.prev_power_per_ue[ue]
                    self.effectiveness[ue] = (1 - self.alpha) * self.effectiveness[ue] + self.alpha * new_effectiveness

        self.prev_tput = current_throughputs[:self.active_ues].copy()
        self.prev_power_per_ue = last_power_per_ue.copy()

        jam_scale = 1.0 + (last_power_per_ue / (budget / self.active_ues))
        estimated_natural_tput = current_throughputs[:self.active_ues] * jam_scale

        estimated_tput = np.zeros(self.active_ues)
        for ue in range(self.active_ues):
            if last_prb_counts[ue] > 0:
                estimated_tput[ue] = estimated_natural_tput[ue] * (curr_prb_counts[ue] / last_prb_counts[ue])
            else:
                estimated_tput[ue] = estimated_natural_tput[ue]

        excess = np.maximum(0.0, estimated_tput - targets)
        inv_effectiveness = 1.0 / (self.effectiveness + 1e-6)
        weighted_excess = excess * inv_effectiveness
        total_weighted_excess = np.sum(weighted_excess)

        action = np.zeros(num_prbs, dtype=np.float32)
        if total_weighted_excess == 0:
            return action

        n_active_with_excess = np.sum(excess > 0)
        blend_alpha = 0.5

        for ue in range(self.active_ues):
            if excess[ue] > 0:
                prop_frac = weighted_excess[ue] / total_weighted_excess
                uniform_frac = 1.0 / n_active_with_excess
                frac = blend_alpha * prop_frac + (1 - blend_alpha) * uniform_frac
                prb_indices = np.where(prb_mask[ue] > 0)[0]
                prb_count = len(prb_indices)
                if prb_count > 0:
                    power_per_prb = frac * budget / prb_count
                    action[prb_indices] += power_per_prb

        # Mask to only allocated PRBs (from obs) and clip to budget
        allocated_mask = (prb_mask[:self.active_ues].sum(axis=0) > 0).astype(np.float32)
        action = action * allocated_mask

        total_power = action.sum()
        if total_power > budget and total_power > 0:
            action *= budget / total_power

        return action


class UniformAdaptiveHeuristic:
    def __init__(self, usable_prbs, num_prbs, budget, step_size=0.1):
        self.usable_prbs = usable_prbs
        self.num_prbs = num_prbs
        self.budget = budget
        self.step_size = step_size
        self.reset()

    def reset(self):
        self.current_power_per_prb = self.budget / self.usable_prbs

    def __call__(self, obs, env, target_throughputs, active_ues):
        num_ues = env.num_ues
        num_prbs = env.num_prbs_total
        current_throughputs = obs[num_ues:2*num_ues][:active_ues]
        targets = np.array(target_throughputs[:active_ues])

        # Read current PRB mask from observation
        prb_mask = get_current_prb_mask(obs, num_ues, num_prbs)

        avg_tput = np.mean(current_throughputs)
        avg_target = np.mean(targets)

        if avg_tput > avg_target:
            self.current_power_per_prb *= (1 + self.step_size)
        else:
            self.current_power_per_prb *= (1 - self.step_size)

        max_per_prb = self.budget / self.usable_prbs * 2
        self.current_power_per_prb = np.clip(self.current_power_per_prb, 0.0, max_per_prb)

        allocated_prbs = np.where(prb_mask[:active_ues].sum(axis=0) > 0)[0]
        action = np.zeros(num_prbs, dtype=np.float32)
        if len(allocated_prbs) > 0:
            action[allocated_prbs] = self.current_power_per_prb
            total_power = action.sum()
            if total_power > self.budget and total_power > 0:
                action *= self.budget / total_power

        return action


def random_budget_action(obs, env, active_ues, budget):
    num_ues = env.num_ues
    num_prbs = env.num_prbs_total

    # Read current PRB mask from observation
    prb_mask = get_current_prb_mask(obs, num_ues, num_prbs)

    allocated_prbs = np.where(prb_mask[:active_ues].sum(axis=0) > 0)[0]
    action = np.zeros(num_prbs, dtype=np.float32)
    if len(allocated_prbs) > 0:
        raw = np.random.uniform(0, 1, size=len(allocated_prbs))
        action[allocated_prbs] = raw / raw.sum() * budget  # always sums to exactly budget
    return action


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_agent(env, action_fn, episodes, active_ues, target_throughputs, reset_fn=None):
    ue_throughputs_log = []
    jamming_powers_log = []
    avg_jammed_sinr_log = []
    power_per_ue_log = []
    avg_prb_concentration_log = []
    target_arr = np.array(target_throughputs[:active_ues])

    for ep in range(episodes):
        reset_all_rngs(EPISODE_SEEDS[ep])
        if reset_fn is not None:
            reset_fn()
        obs = env.reset()
        done = False
        total_jamming_power = 0.0
        episode_throughput = np.zeros(active_ues)
        episode_jammed_sinr = 0.0
        episode_power_per_ue = np.zeros(active_ues)
        episode_prb_concentration = 0.0
        sinr_step_count = 0
        prb_conc_step_count = 0
        step_count = 0

        while not done:
            action = action_fn(obs)
            obs, reward, done, info = env.step(action)

            total_jamming_power += info.get("jamming_power", 0.0)

            sinr_val = info.get("avg_jammed_path_loss", 0.0)
            if sinr_val > 0.0:
                episode_jammed_sinr += sinr_val
                sinr_step_count += 1

            prb_conc_val = info.get("avg_prb_concentration", 0.0)
            if prb_conc_val > 0.0:
                episode_prb_concentration += prb_conc_val
                prb_conc_step_count += 1

            ue_tput = np.array(info.get("ue_throughputs", [0.0]*env.num_ues))
            episode_throughput += ue_tput[:active_ues]

            power_ue = np.array(info.get("power_per_ue", [0.0]*active_ues))
            episode_power_per_ue += power_ue[:active_ues]

            step_count += 1

        episode_throughput /= step_count
        total_jamming_power /= step_count
        episode_jammed_sinr = episode_jammed_sinr / sinr_step_count if sinr_step_count > 0 else float('nan')
        episode_prb_concentration = episode_prb_concentration / prb_conc_step_count if prb_conc_step_count > 0 else float('nan')
        episode_power_per_ue /= step_count

        deviations = np.abs(episode_throughput - target_arr)

        ue_throughputs_log.append(deviations.tolist())
        jamming_powers_log.append(total_jamming_power)
        avg_jammed_sinr_log.append(episode_jammed_sinr)
        power_per_ue_log.append(episode_power_per_ue.tolist())
        avg_prb_concentration_log.append(episode_prb_concentration)

    return {
        "ue_throughputs": ue_throughputs_log,
        "jamming_powers": jamming_powers_log,
        "avg_jammed_path_loss": avg_jammed_sinr_log,
        "power_per_ue": power_per_ue_log,
        "avg_prb_concentration": avg_prb_concentration_log,
    }


# ============================================================
# LOAD MODEL
# ============================================================
model = PPO.load(MODEL_PATH)

# ============================================================
# MASTER EXPERIMENT LOOP
# ============================================================
TARGET_NAMES = ["low-target", "mid-target", "high-target", "var-target"]

for config in EXPERIMENT_CONFIGS:
    active_ues = config["active_ues"]
    usable_prbs = config["usable_prbs"]
    base_dir = os.path.join(BASE_OUTPUT_DIR, config["dir"])

    print(f"\n===== Running {config['dir']} =====")

    for target_name in TARGET_NAMES:
        print(f"--- Target group: {target_name} ---")

        for i in range(1, 11):
            target_rng = np.random.RandomState(SEED + i + hash(target_name) % 10000)

            prbs_per_ue = usable_prbs / active_ues
            max_tput = prbs_per_ue * 0.711

            if target_name == "low-target":
                target_set = target_rng.uniform(0.1 * max_tput, 0.3 * max_tput, size=active_ues).tolist()
            elif target_name == "mid-target":
                target_set = target_rng.uniform(0.3 * max_tput, 0.6 * max_tput, size=active_ues).tolist()
            elif target_name == "high-target":
                target_set = target_rng.uniform(0.6 * max_tput, 0.9 * max_tput, size=active_ues).tolist()
            elif target_name == "var-target":
                target_set = target_rng.uniform(0.1 * max_tput, 0.9 * max_tput, size=active_ues).tolist()

            env = JammingEnv(
                active_ues=active_ues,
                usable_prbs=usable_prbs,
                target_throughputs=target_set
            )

            budget = 0.45 * usable_prbs

            wpp_dir  = os.path.join(base_dir, f"{target_name}-wpp")
            rand_dir = os.path.join(base_dir, f"{target_name}-rand")
            heur_dir = os.path.join(base_dir, f"{target_name}-heur")
            unif_dir = os.path.join(base_dir, f"{target_name}-unif")

            os.makedirs(wpp_dir, exist_ok=True)
            os.makedirs(rand_dir, exist_ok=True)
            os.makedirs(heur_dir, exist_ok=True)
            os.makedirs(unif_dir, exist_ok=True)

            # WPP
            results_wpp = evaluate_agent(
                env,
                lambda obs: model.predict(obs, deterministic=True)[0],
                EPISODES,
                active_ues,
                target_set
            )
            with open(os.path.join(wpp_dir, f"{i}.json"), "w") as f:
                json.dump(results_wpp, f, indent=2)

            # Random
            results_rand = evaluate_agent(
                env,
                lambda obs: random_budget_action(obs, env, active_ues, budget),
                EPISODES,
                active_ues,
                target_set
            )
            with open(os.path.join(rand_dir, f"{i}.json"), "w") as f:
                json.dump(results_rand, f, indent=2)

            # Momentum Heuristic
            heuristic = MomentumHeuristic(active_ues, env.num_prbs_total, budget)
            results_heur = evaluate_agent(
                env,
                lambda obs, _h=heuristic, _env=env, _t=target_set: _h(obs, _env, _t),
                EPISODES,
                active_ues,
                target_set,
                reset_fn=heuristic.reset
            )
            with open(os.path.join(heur_dir, f"{i}.json"), "w") as f:
                json.dump(results_heur, f, indent=2)

            # Uniform Adaptive Heuristic
            uniform_heuristic = UniformAdaptiveHeuristic(usable_prbs, env.num_prbs_total, budget)
            results_unif = evaluate_agent(
                env,
                lambda obs, _h=uniform_heuristic, _env=env, _t=target_set, _a=active_ues: _h(obs, _env, _t, _a),
                EPISODES,
                active_ues,
                target_set,
                reset_fn=uniform_heuristic.reset
            )
            with open(os.path.join(unif_dir, f"{i}.json"), "w") as f:
                json.dump(results_unif, f, indent=2)

            env.close()
            print(f"Finished {config['dir']} | {target_name} | set {i}")

print("\nALL EXPERIMENTS COMPLETE.")

