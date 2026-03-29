#!/usr/bin/env python3
"""
Comprehensive TA stress-test for drone racing policies.

Tests your policy under multiple scenarios:
  1. Nominal (no physics changes) — baseline speed
  2. TA-style: 3 random params per trial (what the TAs actually do)
  3. Worst-case extremes: hand-picked nasty combos
  4. All-9 params randomized simultaneously (harder than TA eval)

Usage:
  python scripts/rsl_rl/stress_test.py \
      --task Isaac-Quadcopter-Race-v0 \
      --num_envs 1 \
      --load_run <your_run_folder> \
      --checkpoint best_model.pt \
      --headless
"""

import sys
import os
import random

local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)

from rsl_rl.utils import wandb_fix  # noqa: F401
import argparse
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Comprehensive TA stress-test.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=800)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
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

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import src.isaac_quad_sim2real.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ═══════════════════════════════════════════════════════════════
# All randomizable parameters and their TA ranges
# ═══════════════════════════════════════════════════════════════

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


def set_nominal(env):
    """Reset ALL physics to config defaults."""
    env._thrust_to_weight[:] = env.cfg.thrust_to_weight
    env._K_aero[:, :2] = env.cfg.k_aero_xy
    env._K_aero[:, 2] = env.cfg.k_aero_z
    env._kp_omega[:, :2] = env.cfg.kp_omega_rp
    env._ki_omega[:, :2] = env.cfg.ki_omega_rp
    env._kd_omega[:, :2] = env.cfg.kd_omega_rp
    env._kp_omega[:, 2] = env.cfg.kp_omega_y
    env._ki_omega[:, 2] = env.cfg.ki_omega_y
    env._kd_omega[:, 2] = env.cfg.kd_omega_y


def apply_params(env, param_dict):
    """Apply a dict of {name: scale} to the env."""
    set_nominal(env)
    lookup = {p[0]: p[1] for p in ALL_PARAMS}
    for name, scale in param_dict.items():
        lookup[name](env, scale)


def random_3_params():
    """TA-style: pick 3 random params, sample each within TA bounds."""
    chosen = random.sample(ALL_PARAMS, 3)
    return {name: np.random.uniform(lo, hi) for name, _, lo, hi in chosen}


def random_all_params():
    """All 9 params randomized (harder than TA eval)."""
    return {name: np.random.uniform(lo, hi) for name, _, lo, hi in ALL_PARAMS}


def get_obs_tensor(obs):
    try:
        return obs["policy"]
    except (KeyError, TypeError, IndexError):
        return obs


# ═══════════════════════════════════════════════════════════════
# Worst-case scenarios — hand-picked nasty combos
# ═══════════════════════════════════════════════════════════════
WORST_CASES = [
    {"name": "Max drag + min TWR",
     "params": {"TWR": 0.95, "aero_xy": 2.0, "aero_z": 2.0}},
    {"name": "Min drag + max TWR (overshoot risk)",
     "params": {"TWR": 1.05, "aero_xy": 0.5, "aero_z": 0.5}},
    {"name": "Sluggish PID (max kd, min kp)",
     "params": {"kp_rp": 0.85, "kd_rp": 1.30, "kp_y": 0.85}},
    {"name": "Twitchy PID (min kd, max kp)",
     "params": {"kp_rp": 1.15, "kd_rp": 0.70, "ki_rp": 1.15}},
    {"name": "Heavy + sluggish (worst combo)",
     "params": {"TWR": 0.95, "aero_xy": 2.0, "kd_rp": 1.30}},
]


def run_trial(env_rsl, raw_env, policy, params, max_steps, dt, num_gates):
    """Run a single trial with given physics params. Returns (status, time, gates, laps)."""

    # Force reset
    raw_env._crashed[:] = 10000
    zero_act = torch.zeros(raw_env.num_envs, raw_env.cfg.action_space, device=raw_env.device)
    env_rsl.step(zero_act)

    obs = get_obs_tensor(env_rsl.get_observations())

    # Apply physics AFTER reset
    apply_params(raw_env, params)

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

    elapsed = total_steps * dt
    laps = max(0, (gates_passed - 1)) // num_gates if gates_passed > 0 else 0
    status = "FINISHED" if finished else ("DIED" if done else "TIMEOUT")
    return status, elapsed, gates_passed, laps


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):

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

    all_results = []

    # ══════════════════════════════════════════════════════════
    # TEST 1: Nominal (no physics changes)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TEST 1: NOMINAL (no physics changes) — 3 trials")
    print("=" * 80)
    for i in range(3):
        status, t, g, l = run_trial(env_rsl, raw_env, policy, {}, max_steps, dt, num_gates)
        print(f"  Trial {i+1}: {status:8s} | {t:6.2f}s | Gates: {g} | Laps: {l}")
        all_results.append({"group": "Nominal", "status": status, "time": t, "gates": g})

    # ══════════════════════════════════════════════════════════
    # TEST 2: TA-style (3 random params) — 10 trials
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TEST 2: TA-STYLE (3 random params) — 10 trials")
    print("=" * 80)
    for i in range(10):
        params = random_3_params()
        status, t, g, l = run_trial(env_rsl, raw_env, policy, params, max_steps, dt, num_gates)
        changed = ", ".join(f"{k}={v:.3f}" for k, v in params.items())
        print(f"  Trial {i+1:2d}: {status:8s} | {t:6.2f}s | Gates: {g} | [{changed}]")
        all_results.append({"group": "TA-style", "status": status, "time": t, "gates": g})

    # ══════════════════════════════════════════════════════════
    # TEST 3: Worst-case extremes — hand-picked nasty combos
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TEST 3: WORST-CASE EXTREMES — 5 hand-picked scenarios")
    print("=" * 80)
    for wc in WORST_CASES:
        status, t, g, l = run_trial(env_rsl, raw_env, policy, wc["params"], max_steps, dt, num_gates)
        print(f"  {wc['name']:40s} | {status:8s} | {t:6.2f}s | Gates: {g}")
        all_results.append({"group": "Worst-case", "status": status, "time": t, "gates": g})

    # ══════════════════════════════════════════════════════════
    # TEST 4: All 9 params randomized — 5 trials (harder than TA)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("TEST 4: ALL-9 PARAMS RANDOMIZED — 5 trials (harder than TA eval)")
    print("=" * 80)
    for i in range(5):
        params = random_all_params()
        status, t, g, l = run_trial(env_rsl, raw_env, policy, params, max_steps, dt, num_gates)
        print(f"  Trial {i+1}: {status:8s} | {t:6.2f}s | Gates: {g}")
        all_results.append({"group": "All-9", "status": status, "time": t, "gates": g})

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    for group in ["Nominal", "TA-style", "Worst-case", "All-9"]:
        grp = [r for r in all_results if r["group"] == group]
        finished = [r for r in grp if r["status"] == "FINISHED"]
        n_total = len(grp)
        n_fin = len(finished)

        if finished:
            times = [r["time"] for r in finished]
            print(f"  {group:12s}: {n_fin}/{n_total} finished | "
                  f"Best: {min(times):.2f}s | Mean: {np.mean(times):.2f}s | "
                  f"Worst: {max(times):.2f}s")
        else:
            max_g = max(r["gates"] for r in grp) if grp else 0
            print(f"  {group:12s}: {n_fin}/{n_total} finished | "
                  f"Max gates: {max_g}")

    total_fin = sum(1 for r in all_results if r["status"] == "FINISHED")
    total = len(all_results)
    print(f"\n  TOTAL: {total_fin}/{total} trials finished")

    if total_fin >= 20:
        print("  ✓ EXCELLENT — policy is extremely robust")
    elif total_fin >= 15:
        print("  ✓ GOOD — safe to submit")
    elif total_fin >= 10:
        print("  ~ OKAY — TA eval will probably pass but not guaranteed")
    else:
        print("  ✗ RISKY — consider more DR training")

    print("=" * 80)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
