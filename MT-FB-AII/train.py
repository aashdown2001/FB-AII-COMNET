import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from jamming_env import JammingEnv
from torch.utils.tensorboard import SummaryWriter

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
ACTIVE_UE_CHOICES = [2, 5, 10]
USABLE_PRB_CHOICES = [5, 10, 24, 26, 38, 52]
NUM_ENVS = 16
log_dir = "./ppo_tensorboard_randomized-8FIXED/"

# --------------------------------------------------------
# Custom tensorboard logging callback
# --------------------------------------------------------
class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "ue_throughputs" in info and "jamming_power" in info:
                for i, tput in enumerate(info["ue_throughputs"]):
                    self.writer.add_scalar(f"throughput/ue_{i}", tput, self.num_timesteps)
                self.writer.add_scalar("jamming/total_power", info["jamming_power"], self.num_timesteps)
        return True

# --------------------------------------------------------
# Create environment with randomized UEs and PRBs
# --------------------------------------------------------
def make_env():
    def _init():
        active_ues = int(np.random.choice(ACTIVE_UE_CHOICES))
        usable_prbs = int(np.random.choice(USABLE_PRB_CHOICES))

        regime = np.random.choice(["low", "mid", "high", "var"])
        prbs_per_ue = usable_prbs / active_ues
        max_tput = prbs_per_ue * 0.711  # approx max Mbps per UE

        if regime == "low":
            targets = np.random.uniform(0.1 * max_tput, 0.3 * max_tput, size=active_ues).tolist()
        elif regime == "mid":
            targets = np.random.uniform(0.3 * max_tput, 0.6 * max_tput, size=active_ues).tolist()
        elif regime == "high":
            targets = np.random.uniform(0.6 * max_tput, 0.9 * max_tput, size=active_ues).tolist()
        else:  # var
            targets = np.random.uniform(0.1 * max_tput, 0.9 * max_tput, size=active_ues).tolist()

        return Monitor(JammingEnv(
            num_ues=10,
            num_prbs_total=52,
            active_ues=active_ues,
            usable_prbs=usable_prbs,
            target_throughputs=targets
        ))
    return _init

# --------------------------------------------------------
# EVERYTHING below must be inside if __name__ == '__main__'
# --------------------------------------------------------
if __name__ == '__main__':
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    os.makedirs(log_dir, exist_ok=True)

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./ppo_checkpoints_randomized-8FIXED/",
        name_prefix="ppo_jammer_checkpoint-8FIXED"
    )
    logging_callback = TensorboardLoggingCallback()

#    model = PPO(
#        "MlpPolicy",
#        env,
#        verbose=1,
#        tensorboard_log=log_dir,
#        device="cuda",
#        learning_rate=2e-5,
#        clip_range=0.1,
#        target_kl=0.05,
#        n_steps=2048,
#        batch_size=4096,
#        n_epochs=20
#    )

    model = PPO.load(
        "ppo_jammer_model_randomized_135M-8FIXED",
        env=env,
        device="cuda",
        custom_objects={
            "learning_rate": 2e-5,
            "clip_range": 0.1,
            "n_steps": 2048,
            "batch_size": 4096,
            "n_epochs": 20,
            "target_kl": 0.05,
        }
    )

    model.learn(total_timesteps=10_000_000, callback=[checkpoint_callback, logging_callback], reset_num_timesteps=False)
    model.save("ppo_jammer_model_randomized_145M-8FIXED")
    print("✅ Training complete and model saved.")
