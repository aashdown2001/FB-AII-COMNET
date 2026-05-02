import numpy as np
import gym
from gym import spaces

# Fixed uniform UE-gNB baseline SINR (same for all UEs and PRBs)
BASELINE_SINR_DB = 35.0

# Fixed jammer path loss vector — represents jammer-to-frequency channel per PRB
# Higher value = jammer has stronger path at that frequency = more effective jamming per unit power
# Shape (52,) — one value per PRB, not per UE, since the jammer's channel
# to a given frequency is independent of which UE happens to be using that PRB
np.random.seed(999)
GLOBAL_JAMMER_PATH_LOSS = np.random.uniform(20.0, 30.0, size=(52,)).astype(np.float32)
np.random.seed(None)  # restore true randomness for everything else

class JammingEnv(gym.Env):
    def __init__(self, num_ues=10, num_prbs_total=52, active_ues=10, usable_prbs=52, target_throughputs=None):
        super().__init__()
        self.num_ues = num_ues
        self.num_prbs_total = num_prbs_total
        self.active_ues = active_ues
        self.usable_prbs = usable_prbs
        self.max_steps = 50
        self.global_step = 0

        # obs: target_throughputs + last_ue_throughputs + last_action (prbs) + last_prb_mask + current_prb_mask
        obs_len = self.num_ues * 2 + self.num_prbs_total + self.num_ues * self.num_prbs_total * 2
        self.observation_space = spaces.Box(low=0.0, high=np.inf,
                                            shape=(obs_len,), dtype=np.float32)

        # Action is per-PRB only
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(self.num_prbs_total,),
                                       dtype=np.float32)

        self.target_throughputs = np.zeros(self.num_ues, dtype=np.float32)
        if target_throughputs is not None:
            self.target_throughputs[:self.active_ues] = target_throughputs[:self.active_ues]

        self.mcs_table = [
            (2, 120), (2, 193), (2, 308), (2, 449), (2, 602), (2, 378),
            (2, 490), (2, 616), (2, 466), (2, 567), (2, 666), (4, 466),
            (4, 567), (4, 666), (4, 772), (4, 873), (6, 711), (6, 797),
            (6, 885), (6, 948), (6, 1023), (6, 1122), (6, 1222), (6, 1321),
            (6, 1421), (6, 1520), (6, 1620), (6, 1720), (6, 1820)
        ]

        self.sinr10_bler = np.array([
            -5.0, -3.7, -2.4, -1.0,  0.6,  2.1,  3.6,  5.1,  6.4,  7.7,
             9.0, 10.2, 11.4, 12.6, 13.8, 15.0, 16.2, 17.4, 18.6, 19.8,
            21.0, 22.3, 23.5, 24.8, 26.1, 27.4, 28.7, 30.0, 31.3
        ], dtype=np.float32)
        self.bler_sigma = np.full_like(self.sinr10_bler, 1.6, dtype=np.float32)

        self.current_mcs = None
        self.prb_mask = None
        self.last_ue_throughputs = None
        self.last_jammer_action = None
        self.last_prb_mask = None
        self.steps = 0

        self.reset()

    def _allocate_prbs(self):
        self.prb_mask = np.zeros((self.num_ues, self.num_prbs_total), dtype=np.float32)

        usable_prbs = self.usable_prbs
        active_ues = self.active_ues
        block_sizes = np.random.multinomial(usable_prbs,
                                        np.ones(self.active_ues) / self.active_ues)

        free = np.zeros(self.num_prbs_total, dtype=bool)
        free[:usable_prbs] = True

        for ue in np.random.permutation(self.active_ues):
            size = block_sizes[ue]
            if size == 0:
                continue

            segments = []
            start = None
            for i in range(self.num_prbs_total + 1):
                if i < self.num_prbs_total and free[i]:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        run = i - start
                        segments.append((start, run))
                        start = None

            if not segments:
                continue

            fitting = [(s, l) for (s, l) in segments if l >= size]
            if fitting:
                seg_start, seg_len = fitting[np.random.randint(len(fitting))]
            else:
                seg_start, seg_len = max(segments, key=lambda t: t[1])

            size_to_use = min(size, seg_len)
            block_start = np.random.randint(seg_start, seg_start + seg_len - size_to_use + 1)

            self.prb_mask[ue, block_start:block_start + size_to_use] = 1.0
            free[block_start:block_start + size_to_use] = False

        # Assign remaining PRBs to the UE with the nearest existing block (preserves contiguity)
        remaining = np.where(free[:usable_prbs])[0]
        for prb in remaining:
            best_ue = 0
            best_dist = np.inf
            for ue in range(active_ues):
                ue_prbs = np.where(self.prb_mask[ue] > 0)[0]
                if len(ue_prbs) > 0:
                    dist = min(abs(prb - p) for p in ue_prbs)
                    if dist < best_dist:
                        best_dist = dist
                        best_ue = ue
            self.prb_mask[best_ue, prb] = 1.0
            free[prb] = False

    def _estimate_bler(self, sinr_db: float, mcs_idx: int) -> float:
        sigma = self.bler_sigma[mcs_idx]
        x0 = self.sinr10_bler[mcs_idx] - np.log(10.0) * sigma
        return float(1.0 / (1.0 + np.exp((sinr_db - x0) / sigma)))

    def compute_throughput(self, n_prbs, mcs_index, mcs_table, n_re_per_prb=168, slot_duration=0.000933):
        Qm, cr_x1024 = mcs_table[mcs_index]
        R = cr_x1024 / 1024.0
        N_info = n_prbs * n_re_per_prb * Qm * R

        if N_info <= 3824:
            tbs = 2 ** int(np.ceil(np.log2(N_info)))
        else:
            tbs = 8 * int(np.ceil((N_info - 24) / 8))

        return (tbs / 1e6) / slot_duration

    def reset(self):
        self.steps = 0
        self.current_mcs = np.full(self.num_ues, 9, dtype=np.int32)
        self.last_ue_throughputs = np.zeros(self.num_ues)
        self.last_jammer_action = np.zeros(self.num_prbs_total)
        self.last_prb_mask = np.zeros((self.num_ues, self.num_prbs_total))

        self.jammer_path_loss = GLOBAL_JAMMER_PATH_LOSS.copy()  # shape (52,)
        self._allocate_prbs()

        self.state = np.concatenate([
            self.target_throughputs,
            self.last_ue_throughputs,
            self.last_jammer_action,
            self.last_prb_mask.flatten(),
            self.prb_mask.flatten()
        ])
        return self.state

    def step(self, action):
        action = np.asarray(action).flatten()
        if action.size != self.num_prbs_total:
            action = np.zeros(self.num_prbs_total)

        # Mask out unusable PRBs
        if self.usable_prbs < self.num_prbs_total:
            action[self.usable_prbs:] = 0.0

        self.last_prb_mask = self.prb_mask.copy()
        self._allocate_prbs()

        # ADD THIS: zero out PRBs not allocated to any active UE
        allocated_mask = (self.prb_mask[:self.active_ues].sum(axis=0) > 0).astype(np.float32)
        action = action * allocated_mask

        # Build SINR matrix: uniform baseline for all UEs/PRBs
        # Jamming reduces SINR by action[prb] * jammer_path_loss[prb]
        # path loss is per-PRB only — the jammer's channel to a frequency
        # is independent of which UE is allocated that PRB
        noise = np.random.randn(self.num_ues, self.num_prbs_total)
        sinr_matrix = np.full((self.num_ues, self.num_prbs_total),
                               BASELINE_SINR_DB, dtype=np.float32) + 1.2 * noise
        for ue in range(self.num_ues):
            prb_indices = np.where(self.prb_mask[ue] > 0)[0]
            for prb in prb_indices:
                sinr_matrix[ue, prb] -= action[prb] * self.jammer_path_loss[prb]

        ue_throughputs, mcs_indices, allocated, blers = [], [], [], []
        for ue in range(self.num_ues):
            prb_indices = np.where(self.prb_mask[ue] > 0)[0]

            if len(prb_indices) == 0 or ue >= self.active_ues:
                ue_throughputs.append(0.0)
                mcs_indices.append(int(self.current_mcs[ue]))
                allocated.append(0)
                blers.append(0.0)
                continue

            prev_idx = int(self.current_mcs[ue])

            mean_sinr = sinr_matrix[ue][prb_indices].mean()
            mean_bler = self._estimate_bler(mean_sinr, prev_idx)

            if mean_bler < 0.05 and prev_idx < 28:
                idx = prev_idx + 1
            elif mean_bler > 0.15 and prev_idx > 0:
                idx = prev_idx - 1
            else:
                idx = prev_idx

            # Realistic TB-level decision: one BLER over all allocated PRBs
            mean_bler_for_log = self._estimate_bler(mean_sinr, idx)
            if mean_bler_for_log <= 0.15:
                tput = self.compute_throughput(len(prb_indices), idx, self.mcs_table)
            else:
                tput = 0.0

            ue_throughputs.append(tput)
            mcs_indices.append(idx)
            allocated.append(len(prb_indices))
            blers.append(mean_bler_for_log)

        ue_throughputs = np.array(ue_throughputs)
        self.current_mcs = np.array(mcs_indices)

        delta = ue_throughputs - self.target_throughputs
        baseline = float(max(self.target_throughputs[:self.active_ues].max(), 1.0))
        abs_err_norm = np.abs(delta) / baseline
        abs_err_norm = 10 * abs_err_norm * abs_err_norm
        mean_err = abs_err_norm[:self.active_ues].mean()

        jammer_power = action[:self.usable_prbs].sum()

        budget = 0.45 * self.usable_prbs
        excess_power = max(0.0, jammer_power - budget)
        reward = - (mean_err + 30.0 * (excess_power / budget)) / (1000.0)

        self.steps += 1
        self.global_step += 1
        done = self.steps >= self.max_steps

        self.last_ue_throughputs = ue_throughputs
        self.last_jammer_action = action.copy()

        self.state = np.concatenate([
            self.target_throughputs,
            self.last_ue_throughputs,
            self.last_jammer_action,
            self.last_prb_mask.flatten(),
            self.prb_mask.flatten()
        ])

        # Logging: power-weighted average jammer path loss of jammed PRBs
        # Higher = RL is targeting PRBs where jammer has stronger path
        jammed_path_losses = []
        jammed_powers = []
        for ue in range(self.active_ues):
            prb_indices = np.where(self.prb_mask[ue] > 0)[0]
            for prb in prb_indices:
                if action[prb] > 0.01:
                    jammed_path_losses.append(self.jammer_path_loss[prb])  # index by prb only
                    jammed_powers.append(action[prb])

        if jammed_path_losses:
            jammed_path_losses = np.array(jammed_path_losses)
            jammed_powers = np.array(jammed_powers)
            avg_jammed_path_loss = float(np.average(jammed_path_losses, weights=jammed_powers))
        else:
            avg_jammed_path_loss = 0.0

        # Power per UE: sum of action over each UE's allocated PRBs
        power_per_ue = []
        for ue in range(self.active_ues):
            prb_indices = np.where(self.prb_mask[ue] > 0)[0]
            power_per_ue.append(float(action[prb_indices].sum()) if len(prb_indices) > 0 else 0.0)

        # PRB concentration: std of action over allocated PRBs per UE
        prb_concentration_list = []
        for ue in range(self.active_ues):
            prb_indices = np.where(self.prb_mask[ue] > 0)[0]
            if len(prb_indices) > 0:
                ue_prb_powers = action[prb_indices]
                if ue_prb_powers.sum() > 0.01:
                    prb_concentration_list.append(float(ue_prb_powers.std()))

        avg_prb_concentration = float(np.mean(prb_concentration_list)) if prb_concentration_list else 0.0

        info = {
            "ue_throughputs": ue_throughputs.tolist(),
            "jamming_power": float(jammer_power),
            "blers": blers,
            "avg_jammed_path_loss": avg_jammed_path_loss,
            "power_per_ue": power_per_ue,
            "avg_prb_concentration": avg_prb_concentration
        }

        if self.global_step % 2000 == 0:
            prb_list = [list(np.where(self.prb_mask[ue] > 0)[0])
                        for ue in range(self.num_ues)]
            lines = [
                "=" * 60,
                f"[Step {self.steps}] Reward: {reward:.2f}",
                f"[Step {self.steps}] Total Jamming Power: {jammer_power:.2f}",
                f"[Step {self.steps}] UE Throughputs: "
                f"{np.array2string(ue_throughputs, precision=2, separator=', ', suppress_small=True)}",
                f"[Step {self.steps}] MCS Indices per UE: {mcs_indices}",
                f"[Step {self.steps}] BLERs per UE: {np.array2string(np.array(blers)*100, precision=3)}",
                f"[Step {self.steps}] PRBs per UE: {allocated}",
                f"[Step {self.steps}] Jammer Power per PRB: {action}"
            ] + [f"  UE {u}: PRBs = {prb_list[u]}" for u in range(self.num_ues)]

            for ln in lines:
                print(ln)
            with open("training_log.txt", "a") as f:
                f.write("\n".join(lines) + "\n")

        return self.state, reward, done, info
