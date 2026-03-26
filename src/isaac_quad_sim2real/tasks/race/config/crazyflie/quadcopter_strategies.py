# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Previous actions stored for observation smoothing
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Create _finished flag if the env doesn't have it (compatibility with original env)
        if not hasattr(self.env, '_finished'):
            self.env._finished = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

        # Gate geometry (square opening in gate frame y/z)
        self._gate_half_side = float(self.cfg.gate_model.gate_side) / 2.0

        # --------------------------
        # Curriculums / constants
        # --------------------------
        # Gate margin curriculum (start stricter for stability, allow tighter racing line later)
        self._gate_margin_start = 0.05
        self._gate_margin_end = 0.02

        # Speed curriculum (ramp aggressiveness)
        self._speed_phase_start_iter = 600
        self._speed_phase_end_iter = 2400

        # Lap curriculum (1 lap -> 2 laps -> 3 laps over training)
        self._lap_1_end_iter = 800
        self._lap_2_end_iter = 1600

        # Velocity reward cap (m/s)
        self._v_cap = 15.0

    # ------------------------------------------------------------------
    # Domain randomization helpers
    # ------------------------------------------------------------------
    def _get_dr_strength(self) -> float:
        """Get current DR strength in [0, 1]. Ramps linearly over first 1500 iterations."""
        it = int(getattr(self.env, "iteration", 0))
        dr_ramp_end = 1500
        if dr_ramp_end <= 0:
            return 1.0
        return float(np.clip(it / float(dr_ramp_end), 0.0, 1.0))

    def _randomize_physics(self, env_ids: torch.Tensor):
        """Randomize physics for reset environments to match TA evaluation ranges.

        Each time an environment resets, we sample new physics parameters:
          - TWR:         nominal * [0.95, 1.05]
          - Aero drag:   nominal * [0.5, 2.0]
          - PID kp/ki:   nominal * [0.85, 1.15]
          - PID kd:      nominal * [0.70, 1.30]

        Strength ramps 0% to 100% over 1500 iters so the policy adapts gradually.
        """
        s = self._get_dr_strength()
        if s < 1e-6:
            return  # No randomization yet

        n = len(env_ids)
        device = self.device

        def _rand_scale(low, high, size):
            """Sample random multiplier, interpolated toward nominal by (1-s)."""
            raw = torch.empty(size, device=device).uniform_(low, high)
            return 1.0 + s * (raw - 1.0)

        # Thrust to weight ratio: +/- 5%
        self.env._thrust_to_weight[env_ids] = self.env._twr_value * _rand_scale(0.95, 1.05, n)

        # Aerodynamic drag: 0.5x to 2.0x
        aero_xy = _rand_scale(0.5, 2.0, n)
        aero_z = _rand_scale(0.5, 2.0, n)
        self.env._K_aero[env_ids, 0] = self.env._k_aero_xy_value * aero_xy
        self.env._K_aero[env_ids, 1] = self.env._k_aero_xy_value * aero_xy
        self.env._K_aero[env_ids, 2] = self.env._k_aero_z_value * aero_z

        # PID gains for roll/pitch: kp/ki +/-15%, kd +/-30%
        kp_rp = _rand_scale(0.85, 1.15, n)
        ki_rp = _rand_scale(0.85, 1.15, n)
        kd_rp = _rand_scale(0.70, 1.30, n)
        self.env._kp_omega[env_ids, 0] = self.env._kp_omega_rp_value * kp_rp
        self.env._kp_omega[env_ids, 1] = self.env._kp_omega_rp_value * kp_rp
        self.env._ki_omega[env_ids, 0] = self.env._ki_omega_rp_value * ki_rp
        self.env._ki_omega[env_ids, 1] = self.env._ki_omega_rp_value * ki_rp
        self.env._kd_omega[env_ids, 0] = self.env._kd_omega_rp_value * kd_rp
        self.env._kd_omega[env_ids, 1] = self.env._kd_omega_rp_value * kd_rp

        # PID gains for yaw: kp/ki +/-15%, kd +/-30%
        kp_y = _rand_scale(0.85, 1.15, n)
        ki_y = _rand_scale(0.85, 1.15, n)
        kd_y = _rand_scale(0.70, 1.30, n)
        self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value * kp_y
        self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value * ki_y
        self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value * kd_y

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure

        # Speed curriculum: linearly ramp speed-related rewards from iter 600 to 2400
        it = int(getattr(self.env, "iteration", 0))
        if self._speed_phase_end_iter <= self._speed_phase_start_iter:
            speed_phase = 1.0
        else:
            speed_phase = float(np.clip(
                (it - self._speed_phase_start_iter) /
                float(self._speed_phase_end_iter - self._speed_phase_start_iter),
                0.0, 1.0))

        # Curriculum multipliers that shift training from "learn to pass gates" to "go fast"
        center_mult = 1.0 - 0.8 * speed_phase   # 1.0 -> 0.2 (care less about centering)
        action_mult = 1.0 - 0.7 * speed_phase   # 1.0 -> 0.3 (allow aggressive actions)
        vel_mult    = speed_phase                 # 0.0 -> 1.0 (velocity rewards ramp in)
        time_mult   = 1.0 + 5.0 * speed_phase   # 1.0 -> 6.0 (heavier time pressure)
        death_mult  = 1.0 + 3.0 * speed_phase   # 1.0 -> 4.0 (crash harder when going fast)

        # Gate margin shrinks over curriculum to allow tighter racing lines
        margin = float(self._gate_margin_start +
                       (self._gate_margin_end - self._gate_margin_start) * speed_phase)

        # Lap curriculum: start with 1 lap, ramp to 3 laps over training
        laps_target = int(self.cfg.max_n_laps)
        if self.cfg.is_train:
            if it < self._lap_1_end_iter:
                laps_target = 1
            elif it < self._lap_2_end_iter:
                laps_target = min(2, laps_target)

        # Drone position in current gate frame (computed by env in _get_dones)
        pose_gate = self.env._pose_drone_wrt_gate
        xg, yg, zg = pose_gate[:, 0], pose_gate[:, 1], pose_gate[:, 2]

        # Potential-based progress shaping: reward getting closer to the gate
        dist = torch.linalg.norm(pose_gate, dim=1)
        prev_dist = self.env._last_distance_to_goal
        progress = torch.clamp(prev_dist - dist, -0.50, 0.50)

        # Gate pass detection: drone crosses the gate plane from the correct direction
        # xg > 0 means in front of gate, xg <= 0 means behind/through gate
        prev_xg = self.env._prev_x_drone_wrt_gate
        crossed_plane = (prev_xg > 0.0) & (xg <= 0.0)

        # Check if the drone is inside the gate opening when crossing
        inside_opening = ((yg.abs() <= (self._gate_half_side - margin)) &
                          (zg.abs() <= (self._gate_half_side - margin)))
        gate_pass = crossed_plane & inside_opening
        gate_miss = crossed_plane & (~inside_opening)

        # Crash detection using contact sensor (ignore first 0.2s since drone starts on ground)
        contact_forces = self.env._contact_sensor.data.net_forces_w
        contact_mag = torch.linalg.norm(contact_forces, dim=-1).squeeze(1)
        contact = contact_mag > 1e-6
        contact_mask = self.env.episode_length_buf > 10
        crashed = (contact & contact_mask).float()
        self.env._crashed = self.env._crashed + (crashed > 0).int()

        # Penalize jerky actions (sum of squared action values)
        action_l2 = torch.sum(self.env._actions ** 2, dim=1)

        # Velocity shaping: project drone velocity toward current and next gate
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b

        idx = self.env._idx_wp
        gate_pos_w = self.env._waypoints[idx, :3]
        next_idx = (idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_idx, :3]

        # Transform gate positions into body frame
        gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, gate_pos_w)
        next_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_pos_w)

        # Unit direction vectors toward each gate
        gate_dir_b = gate_pos_b / (torch.linalg.norm(gate_pos_b, dim=1, keepdim=True) + 1e-6)
        next_gate_dir_b = next_gate_pos_b / (torch.linalg.norm(next_gate_pos_b, dim=1, keepdim=True) + 1e-6)

        # Project velocity onto direction toward each gate
        vel_proj_gate = torch.sum(drone_lin_vel_b * gate_dir_b, dim=1)
        vel_proj_next = torch.sum(drone_lin_vel_b * next_gate_dir_b, dim=1)

        # Next-gate velocity reward only activates near the current gate plane
        plane_focus = torch.exp(-xg.abs() / 0.5)

        dt = float(self.env.cfg.sim.dt * self.env.cfg.decimation)
        vel_to_gate = torch.clamp(vel_proj_gate, 0.0, self._v_cap) * dt
        vel_to_next_gate = plane_focus * torch.clamp(vel_proj_next, 0.0, self._v_cap) * dt

        # Centering bonus: only paid on the gate-pass step (Gaussian based on offset from center)
        center_err2 = yg ** 2 + zg ** 2
        center_bonus = torch.exp(-center_err2 / (0.25 ** 2))
        center_at_pass = gate_pass.float() * center_bonus

        # Finish detection: check if drone completed enough laps this timestep
        num_gates = self.env._waypoints.shape[0]
        n_before = self.env._n_gates_passed
        finish_before = ((n_before - 1) // num_gates >= laps_target)
        n_after = n_before + gate_pass.int()
        finish_after = ((n_after - 1) // num_gates >= laps_target)
        finished_this_step = finish_after & (~finish_before)

        # Track finish flag (self-contained, works with original env)
        self.env._finished = self.env._finished | finished_this_step

        # Force episode termination on finish via crash counter
        # (original env checks _crashed > 100 in _get_dones)
        if finished_this_step.any():
            finish_ids = torch.where(finished_this_step)[0]
            self.env._crashed[finish_ids] = 10000

        # "Finish earlier = bigger bonus" quadratic scaling
        max_steps = float(self.env.max_episode_length)
        finish_time_frac = torch.clamp(self.env.episode_length_buf.float() / max_steps, 0.0, 1.0)
        finish_bonus_scale = (1.0 - finish_time_frac) ** 2

        # Waypoint bookkeeping: advance to next gate on successful pass
        if gate_pass.any():
            ids = torch.where(gate_pass)[0]
            self.env._n_gates_passed[ids] += 1
            new_idx = (self.env._idx_wp[ids] + 1) % num_gates
            self.env._idx_wp[ids] = new_idx
            self.env._desired_pos_w[ids] = self.env._waypoints[new_idx, :3]

            # Update distance/pose buffers immediately for the new target gate
            new_pose_gate, _ = subtract_frame_transforms(
                self.env._waypoints[new_idx, :3],
                self.env._waypoints_quat[new_idx, :],
                self.env._robot.data.root_link_state_w[ids, :3])
            self.env._prev_x_drone_wrt_gate[ids] = new_pose_gate[:, 0]
            self.env._last_distance_to_goal[ids] = torch.linalg.norm(new_pose_gate, dim=1)
            self.env._pose_drone_wrt_gate[ids] = new_pose_gate

        # Gate miss = instant termination during training (sample efficiency)
        if self.cfg.is_train and gate_miss.any():
            miss_ids = torch.where(gate_miss)[0]
            self.env._crashed[miss_ids] = 1000

        # Update distance buffers for environments that did NOT pass a gate
        not_pass = ~gate_pass
        self.env._prev_x_drone_wrt_gate[not_pass] = xg[not_pass]
        self.env._last_distance_to_goal[not_pass] = dist[not_pass]
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress":        progress * self.env.rew["progress_reward_scale"],
                "gate_pass":       gate_pass.float() * self.env.rew["gate_pass_reward_scale"],
                "gate_miss":       gate_miss.float() * self.env.rew["gate_miss_reward_scale"],
                "center_at_pass":  center_at_pass * (self.env.rew["center_at_pass_reward_scale"] * center_mult),
                "vel_to_gate":     vel_to_gate * (self.env.rew["vel_to_gate_reward_scale"] * vel_mult),
                "vel_to_next_gate": vel_to_next_gate * (self.env.rew["vel_to_next_gate_reward_scale"] * vel_mult),
                "finish":          finished_this_step.float() * self.env.rew["finish_reward_scale"] * finish_bonus_scale,
                "time_penalty":    torch.ones_like(dist) * (self.env.rew["time_penalty_reward_scale"] * time_mult),
                "action_l2":       action_l2 * (self.env.rew["action_l2_reward_scale"] * action_mult),
                "crash":           crashed * self.env.rew["crash_reward_scale"],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            # Apply death cost only on failures, NOT on successful finishes
            death_cost = self.env.rew["death_cost"] * death_mult
            failure_term = self.env.reset_terminated & (~self.env._finished)
            reward = torch.where(failure_term, torch.ones_like(reward) * death_cost, reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        # Basic drone states
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b

        # Current and next gate positions in body frame
        idx = self.env._idx_wp
        gate_pos_w = self.env._waypoints[idx, :3]
        next_idx = (idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_idx, :3]

        gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, gate_pos_w)
        next_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_pos_w)

        # Relative position to current gate in gate frame
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Number of gates passed (race progress)
        gates_passed = self.env._n_gates_passed.unsqueeze(1).float()
        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                gate_pos_b,              # current gate in body frame (3)
                next_gate_pos_b,         # next gate in body frame (3)
                drone_pos_gate_frame,    # drone position in gate frame (3)
                drone_lin_vel_b,         # linear velocity in body frame (3)
                drone_ang_vel_b,         # angular velocity in body frame (3)
                drone_quat_w,            # orientation quaternion (4)
                self._prev_actions,      # previous actions for smoothing (4)
                gates_passed,            # race progress counter (1)
            ],
            # TODO ----- END -----
            dim=-1,
        )  # Total: 24 dimensions

        # Store current actions for next step's observation
        self._prev_actions = self.env._actions.detach().clone()

        observations = {"policy": obs}
        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            extras["Episode_Termination/finished"] = torch.count_nonzero(self.env._finished[env_ids]).item()

            # Log average finish time for successful episodes
            finished_mask = self.env._finished[env_ids]
            if torch.any(finished_mask):
                dt_step = float(self.env.cfg.sim.dt * self.env.cfg.decimation)
                mean_ft = (self.env.episode_length_buf[env_ids][finished_mask].float() * dt_step).mean()
                extras["Episode_Finish/finish_time_s"] = float(mean_ft.item())
            else:
                extras["Episode_Finish/finish_time_s"] = 0.0

            # Log domain randomization strength
            extras["Curriculum/dr_strength"] = self._get_dr_strength()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 7:] = 0.0

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # This example code initializes the drone 2m behind the first gate. You should delete it or heavily
        # modify it once you begin the racing task.

        if self.cfg.is_train:
            # 80% of resets start at gate 0 (matching TA eval), 20% at random gates after iter 500
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            if getattr(self.env, "iteration", 0) >= 500:
                rand_mask = torch.rand(n_reset, device=self.device) < 0.20
                if rand_mask.any():
                    waypoint_indices[rand_mask] = torch.randint(
                        low=0, high=self.env._waypoints.shape[0],
                        size=(int(rand_mask.sum().item()),),
                        device=self.device, dtype=self.env._idx_wp.dtype)

            # Get gate positions and headings
            x0_wp = self.env._waypoints[waypoint_indices][:, 0]
            y0_wp = self.env._waypoints[waypoint_indices][:, 1]
            z_wp = self.env._waypoints[waypoint_indices][:, 2]
            theta = self.env._waypoints[waypoint_indices][:, -1]

            # Start-position curriculum: ramp to full TA eval ranges by iter 800
            it = int(getattr(self.env, "iteration", 0))
            phase = float(np.clip(it / 800.0, 0.0, 1.0))
            x_min = -2.0 - 1.0 * phase    # -2.0 -> -3.0
            y_lim = 0.5 + 0.5 * phase     #  0.5 ->  1.0

            # Sample random start positions within TA eval ranges
            x_local = torch.empty(n_reset, device=self.device).uniform_(x_min, -0.5)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-y_lim, y_lim)

            # Rotate local position to world frame based on gate heading
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            x_rot = cos_t * x_local - sin_t * y_local
            y_rot = sin_t * x_local + cos_t * y_local
            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot

            # Gate 0 starts on the ground (z=0.05), other gates start at gate height
            z0 = torch.where(waypoint_indices == 0,
                             torch.full((n_reset,), 0.05, device=self.device), z_wp)

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = z0

            # Point drone towards the target gate with small yaw noise
            yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
            yaw = yaw + torch.empty(n_reset, device=self.device).uniform_(-0.20, 0.20)
            quat = quat_from_euler_xyz(
                torch.zeros(n_reset, device=self.device),
                torch.zeros(n_reset, device=self.device), yaw)
            default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled within TA eval bounds
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # Rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # Point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0
            default_root_state[:, 7:] = 0.0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0)
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids] = self.env._waypoints[waypoint_indices, :3].clone()
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0
        self.env._crashed[env_ids] = 0
        self.env._finished[env_ids] = False

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3])
        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0].clone()
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._pose_drone_wrt_gate[env_ids], dim=1)

        # Domain randomization: randomize physics for each reset environment
        # This trains the policy to handle the TA's physics perturbations
        if self.cfg.is_train:
            self._randomize_physics(env_ids)