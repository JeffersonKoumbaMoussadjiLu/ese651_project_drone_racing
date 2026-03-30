#!/usr/bin/env python3
"""
TA-style evaluation matching Ed post #71 exactly:
  - Only 3 parameters are randomly sampled (from the full list in handout 3.1)
  - x_local and y_local are hard-coded (within [-3.0,-0.5] and [-1.0,1.0])
  - Sampled once, held constant for the entire run

Usage:
  python scripts/rsl_rl/test_like_ta.py \
      --task Isaac-Quadcopter-Race-v0 \
      --num_envs 1 \
      --load_run <your_run_folder> \
      --checkpoint best_model.pt \
      --headless \
      --num_trials 5
"""

import sys
import os

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)

from rsl_rl.utils import wandb_fix  # noqa: F401
import argparse
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="TA-style evaluation with randomized dynamics.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=800)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_trials", type=int, default=5,
                    help="Number of different 3-param combos to test.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import random

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import src.isaac_quad_sim2real.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ═══════════════════════════════════════════════════════════════
# All randomizable parameters and their ranges (from handout 3.1)
# ═══════════════════════════════════════════════════════════════
# Each entry: (name, setter_function, low_scale, high_scale)
# The setter_function takes (env, scale_value) and applies it.

def _set_twr(env, s):
    env._thrust_to_weight[:] = env.cfg.thrust_to_weight * s

def _set_aero_xy(env, s):
    env._K_aero[:, 0] = env.cfg.k_aero_xy * s
    env._K_aero[:, 1] = env.cfg.k_aero_xy * s

def _set_aero_z(env, s):
    env._K_aero[:, 2] = env.cfg.k_aero_z * s

def _set_kp_rp(env, s):
    env._kp_omega[:, 0] = env.cfg.kp_omega_rp * s
    env._kp_omega[:, 1] = env.cfg.kp_omega_rp * s

def _set_ki_rp(env, s):
    env._ki_omega[:, 0] = env.cfg.ki_omega_rp * s
    env._ki_omega[:, 1] = env.cfg.ki_omega_rp * s

def _set_kd_rp(env, s):
    env._kd_omega[:, 0] = env.cfg.kd_omega_rp * s
    env._kd_omega[:, 1] = env.cfg.kd_omega_rp * s

def _set_kp_y(env, s):
    env._kp_omega[:, 2] = env.cfg.kp_omega_y * s

def _set_ki_y(env, s):
    env._ki_omega[:, 2] = env.cfg.ki_omega_y * s

def _set_kd_y(env, s):
    env._kd_omega[:, 2] = env.cfg.kd_omega_y * s


ALL_PARAMS = [
    ("TWR",     _set_twr,     0.95, 1.05),
    ("aero_xy", _set_aero_xy, 0.50, 2.00),
    ("aero_z",  _set_aero_z,  0.50, 2.00),
    ("kp_rp",   _set_kp_rp,   0.85, 1.15),
    ("ki_rp",   _set_ki_rp,   0.85, 1.15),
    ("kd_rp",   _set_kd_rp,   0.70, 1.30),
    ("kp_y",    _set_kp_y,    0.85, 1.15),
    ("ki_y",    _set_ki_y,    0.85, 1.15),
    ("kd_y",    _set_kd_y,    0.70, 1.30),
]


def set_nominal_physics(env):
    """Reset ALL physics to nominal (config) values."""
    env._thrust_to_weight[:] = env.cfg.thrust_to_weight
    env._K_aero[:, :2] = env.cfg.k_aero_xy
    env._K_aero[:, 2] = env.cfg.k_aero_z
    env._kp_omega[:, :2] = env.cfg.kp_omega_rp
    env._ki_omega[:, :2] = env.cfg.ki_omega_rp
    env._kd_omega[:, :2] = env.cfg.kd_omega_rp
    env._kp_omega[:, 2] = env.cfg.kp_omega_y
    env._ki_omega[:, 2] = env.cfg.ki_omega_y
    env._kd_omega[:, 2] = env.cfg.kd_omega_y


def randomize_3_params(env):
    """TA-style: pick 3 random parameters, sample each once, leave rest nominal."""
    # Reset everything to nominal first
    set_nominal_physics(env)

    # Pick 3 unique parameters
    chosen = random.sample(ALL_PARAMS, 3)

    changed = {}
    for name, setter, lo, hi in chosen:
        scale = np.random.uniform(lo, hi)
        setter(env, scale)
        changed[name] = f"{scale:.3f}"

    return changed


def get_obs_tensor(obs):
    """Extract raw tensor from obs (handles TensorDict, dict, or plain tensor)."""
    try:
        return obs["policy"]
    except (KeyError, TypeError, IndexError):
        return obs


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run TA-style evaluation."""

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.is_train = False
    env_cfg.rewards = {}

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading model from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env_rsl = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env_rsl, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    raw_env = env.unwrapped

    dt = float(raw_env.cfg.sim.dt * raw_env.cfg.decimation)
    num_gates = raw_env._waypoints.shape[0]
    max_steps = int(raw_env.max_episode_length)

    print("\n" + "=" * 80)
    print("TA-STYLE EVALUATION (Ed post #71)")
    print(f"  - 3 random params per trial (rest nominal)")
    print(f"  - Trials: {args_cli.num_trials}")
    print(f"  - Gates/lap: {num_gates}, Laps required: {raw_env.cfg.max_n_laps}")
    print(f"  - Max episode time: {max_steps * dt:.1f}s")
    print("=" * 80 + "\n")

    results = []

    for trial in range(args_cli.num_trials):
        # ── Force reset between trials ──
        if trial > 0:
            raw_env._crashed[:] = 10000
            zero_act = torch.zeros(raw_env.num_envs, raw_env.cfg.action_space, device=raw_env.device)
            env_rsl.step(zero_act)

        # Get fresh obs (env has auto-reset)
        obs = get_obs_tensor(env_rsl.get_observations())

        # ── TA randomization: 3 params, applied AFTER reset ──
        changed = randomize_3_params(raw_env)

        gates_passed = 0
        total_steps = 0
        finished = False
        done = False

        for step in range(max_steps):
            actions = policy(obs)
            obs, _, dones, extras = env_rsl.step(actions)
            obs = get_obs_tensor(obs)
            total_steps += 1

            current_gates = raw_env._n_gates_passed[0].item()
            if current_gates > gates_passed:
                gates_passed = current_gates

            if gates_passed > 0 and (gates_passed - 1) // num_gates >= raw_env.cfg.max_n_laps:
                finished = True
                break

            if dones.any():
                done = True
                break

        elapsed_time = total_steps * dt
        laps = max(0, (gates_passed - 1)) // num_gates if gates_passed > 0 else 0

        status = "FINISHED" if finished else ("DIED" if done else "TIMEOUT")

        results.append({
            "trial": trial + 1,
            "status": status,
            "time": elapsed_time,
            "gates": gates_passed,
            "laps": laps,
            "changed": changed,
        })

        changed_str = ", ".join(f"{k}={v}" for k, v in changed.items())
        print(f"Trial {trial+1:2d}/{args_cli.num_trials}: "
              f"{status:8s} | Time: {elapsed_time:6.2f}s | "
              f"Gates: {gates_passed:3d} | Laps: {laps} | "
              f"Changed: [{changed_str}]")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    finished_trials = [r for r in results if r["status"] == "FINISHED"]
    died_trials = [r for r in results if r["status"] == "DIED"]
    timeout_trials = [r for r in results if r["status"] == "TIMEOUT"]

    print(f"  Finished: {len(finished_trials)}/{len(results)}")
    print(f"  Died:     {len(died_trials)}/{len(results)}")
    print(f"  Timeout:  {len(timeout_trials)}/{len(results)}")

    if finished_trials:
        times = [r["time"] for r in finished_trials]
        print(f"\n  Finish times:")
        print(f"    Best:   {min(times):.2f}s")
        print(f"    Worst:  {max(times):.2f}s")
        print(f"    Mean:   {np.mean(times):.2f}s")
        print(f"    Median: {np.median(times):.2f}s")
    else:
        all_gates = [r["gates"] for r in results]
        print(f"\n  No trials finished 3 laps.")
        print(f"  Max gates passed: {max(all_gates)}")
        print(f"  Mean gates passed: {np.mean(all_gates):.1f}")

    print("=" * 80)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
