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
        self._v_cap = 18.0

        # How early next-gate lookahead shaping turns on around the current gate plane.
        self._plane_focus_width_start = 0.65
        self._plane_focus_width_end = 1.20

        # Reset curriculum. Most episodes still start at gate 0, but later in training
        # we also sample local maneuver segments around the reused / shared gate.
        self._global_random_reset_start_iter = 500
        self._focus_reset_start_iter = 1000
        self._velocity_reset_start_iter = 800
        self._velocity_reset_end_iter = 2500

        # Fine-tune schedule for resuming from the 5k-iteration robust checkpoint.
        # Stage 1: reduced DR + speed-focused local practice.
        # Stage 2: ramp DR back to full TA range while preserving the faster line.
        self._fine_tune_start_iter = 5000
        self._fine_tune_polish_end_iter = 8500
        self._fine_tune_restore_end_iter = 11000

        # Physical-gate aliases: some waypoints reuse the same gate object with
        # a different desired traversal direction (e.g. powerloop/shared gate).
        # We group waypoints by gate position so we can allow the CURRENT target
        # gate but penalize flying through any other physical gate out of sequence.
        gate_pos = self.env._waypoints[:, :3]
        self._num_gates = gate_pos.shape[0]
        self._same_physical_gate = torch.cdist(gate_pos, gate_pos) < 1.0e-4
        self._prev_all_x = torch.zeros(self.num_envs, self._num_gates, device=self.device)

        # Gates involved in the powerloop / shared-gate maneuver. We bias some resets
        # toward these gates so the policy repeatedly practices the hardest segment.
        shared_gate_indices = torch.where(self._same_physical_gate.sum(dim=1) > 1)[0]
        if shared_gate_indices.numel() > 0:
            focus_gate_indices = torch.unique(
                torch.cat(
                    [
                        (shared_gate_indices - 2) % self._num_gates,
                        (shared_gate_indices - 1) % self._num_gates,
                        shared_gate_indices,
                        (shared_gate_indices + 1) % self._num_gates,
                    ]
                )
            )
            # Keep gate-0 starts dominant. If gate 0 only appears because of wrap-around,
            # exclude it from the maneuver-focused reset pool.
            if torch.any(focus_gate_indices == 0) and not torch.any(shared_gate_indices == 0):
                focus_gate_indices = focus_gate_indices[focus_gate_indices != 0]
            self._focus_gate_indices = focus_gate_indices.to(dtype=torch.long)
        else:
            self._focus_gate_indices = torch.arange(self._num_gates, device=self.device, dtype=torch.long)

        self._focus_gate_mask = torch.zeros(self._num_gates, device=self.device, dtype=torch.bool)
        self._focus_gate_mask[self._focus_gate_indices] = True

    # ------------------------------------------------------------------
    # Domain randomization helpers
    # ------------------------------------------------------------------
    def _get_dr_strength(self) -> float:
        """Domain-randomization schedule.

        Scratch training still ramps DR from 0 -> 1 over the first 1500 iterations.
        For fine-tuning from an existing ~5k-iteration checkpoint, we temporarily drop to a
        moderate DR level to polish the raceline, then ramp back to full TA-style DR.
        """
        it = int(getattr(self.env, "iteration", 0))
        dr_ramp_end = 1500
        if it < self._fine_tune_start_iter:
            if dr_ramp_end <= 0:
                return 1.0
            return float(np.clip(it / float(dr_ramp_end), 0.0, 1.0))
        if it < self._fine_tune_polish_end_iter:
            return 0.70
        if it < self._fine_tune_restore_end_iter:
            alpha = float(
                np.clip(
                    (it - self._fine_tune_polish_end_iter)
                    / float(self._fine_tune_restore_end_iter - self._fine_tune_polish_end_iter),
                    0.0,
                    1.0,
                )
            )
            return 0.70 + 0.30 * alpha
        return 1.0

    def _get_fine_tune_phase(self) -> float:
        """Fine-tune progress in [0, 1] for the speed-polish stage after iteration 5000."""
        it = int(getattr(self.env, "iteration", 0))
        if it < self._fine_tune_start_iter:
            return 0.0
        if self._fine_tune_polish_end_iter <= self._fine_tune_start_iter:
            return 1.0
        return float(
            np.clip(
                (it - self._fine_tune_start_iter)
                / float(self._fine_tune_polish_end_iter - self._fine_tune_start_iter),
                0.0,
                1.0,
            )
        )

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

        # Change k_eta
        k_eta = 2.3 * torch.pow(10, _rand_scale(6, 10, n))
        self.env._k_eta[env_ids] = k_eta

    def _get_all_gate_relative_positions(self, drone_pos_w: torch.Tensor) -> torch.Tensor:
        """Return drone position expressed in every gate frame.

        Output shape is [num_envs, num_gates, 3]. The x-axis of each gate frame is
        the signed traversal direction used for pass detection.
        """
        gate_pos_w = self.env._waypoints[:, :3]                    # [G, 3]
        gate_rot_w = matrix_from_quat(self.env._waypoints_quat)    # [G, 3, 3]
        rel_world = drone_pos_w[:, None, :] - gate_pos_w[None, :, :]
        # Convert world-frame displacement into each gate's local frame: R^T * dx
        rel_gate = torch.einsum("gji,egj->egi", gate_rot_w, rel_world)
        return rel_gate


    def get_rewards(self) -> torch.Tensor:
        """Compute racing rewards with legal gate-order enforcement and powerloop lookahead shaping."""

        # Speed curriculum: linearly ramp speed-related rewards from iter 600 to 2400
        it = int(getattr(self.env, "iteration", 0))
        fine_tune_phase = self._get_fine_tune_phase()
        if self._speed_phase_end_iter <= self._speed_phase_start_iter:
            speed_phase = 1.0
        else:
            speed_phase = float(
                np.clip(
                    (it - self._speed_phase_start_iter)
                    / float(self._speed_phase_end_iter - self._speed_phase_start_iter),
                    0.0,
                    1.0,
                )
            )

        # Curriculum multipliers that shift training from
        # "learn to legally pass gates" to "carry speed through the line".
        center_mult = 1.0 - 0.70 * speed_phase
        action_mult = 1.0 - 0.55 * speed_phase
        vel_mult = 0.25 + 0.75 * speed_phase
        time_mult = 1.0 + 5.5 * speed_phase
        death_mult = 1.0 + 3.0 * speed_phase

        # Gate margin shrinks over curriculum to allow tighter racing lines
        margin = float(self._gate_margin_start + (self._gate_margin_end - self._gate_margin_start) * speed_phase)
        plane_focus_width = float(
            self._plane_focus_width_start
            + (self._plane_focus_width_end - self._plane_focus_width_start) * speed_phase
        ) + 0.25 * fine_tune_phase

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

        # Potential-based progress shaping: reward getting closer to the gate.
        # We keep this, but the lookahead rewards below carry more of the speed objective.
        dist = torch.linalg.norm(pose_gate, dim=1)
        prev_dist = self.env._last_distance_to_goal
        progress = torch.clamp(prev_dist - dist, -0.50, 0.50)

        # Gate pass detection: drone crosses the gate plane from the correct direction.
        # xg > 0 means in front of the gate, xg <= 0 means behind / through the gate.
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

        # State needed for lookahead shaping
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_rot_w = matrix_from_quat(drone_quat_w)

        # Detect illegal gate traversals. A legal crossing must be through the
        # CURRENT target gate, in the desired direction, while inside the opening.
        # Crossing any other physical gate (including shared gates that are not the
        # current target) is treated as cheating and terminated during training.
        all_pose_gate = self._get_all_gate_relative_positions(drone_pos_w)
        all_x = all_pose_gate[..., 0]
        all_y = all_pose_gate[..., 1]
        all_z = all_pose_gate[..., 2]
        inside_any = ((all_y.abs() <= (self._gate_half_side - margin)) &
                      (all_z.abs() <= (self._gate_half_side - margin)))
        any_cross = (
            ((self._prev_all_x > 0.0) & (all_x <= 0.0)) |
            ((self._prev_all_x < 0.0) & (all_x >= 0.0))
        ) & inside_any

        idx = self.env._idx_wp
        allowed_same_physical_gate = self._same_physical_gate[idx]  # [E, G]
        illegal_other_gate = torch.any(any_cross & (~allowed_same_physical_gate), dim=1)
        wrong_way_current = (prev_xg < 0.0) & (xg >= 0.0) & inside_opening
        illegal_cross = wrong_way_current | illegal_other_gate

        gate_pos_w = self.env._waypoints[idx, :3]
        next_idx = (idx + 1) % self.env._waypoints.shape[0]
        next2_idx = (idx + 2) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_idx, :3]
        next2_gate_pos_w = self.env._waypoints[next2_idx, :3]

        critical_segment = self._focus_gate_mask[idx] | self._focus_gate_mask[next_idx]

        # Transform gate positions into body frame
        gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, gate_pos_w)
        next_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_pos_w)
        next2_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next2_gate_pos_w)

        # Unit direction vectors toward current / next / next2 gates
        gate_dir_b = gate_pos_b / (torch.linalg.norm(gate_pos_b, dim=1, keepdim=True) + 1e-6)
        next_gate_dir_b = next_gate_pos_b / (torch.linalg.norm(next_gate_pos_b, dim=1, keepdim=True) + 1e-6)
        next2_gate_dir_b = next2_gate_pos_b / (torch.linalg.norm(next2_gate_pos_b, dim=1, keepdim=True) + 1e-6)

        # Project body-frame velocity onto those lookahead directions
        vel_proj_gate = torch.sum(drone_lin_vel_b * gate_dir_b, dim=1)
        vel_proj_next = torch.sum(drone_lin_vel_b * next_gate_dir_b, dim=1)
        vel_proj_next2 = torch.sum(drone_lin_vel_b * next2_gate_dir_b, dim=1)

        # Lookahead rewards become active before the drone reaches the current gate plane.
        plane_focus = torch.exp(-xg.abs() / plane_focus_width)
        next2_focus = plane_focus * torch.exp(-torch.linalg.norm(next_gate_pos_b, dim=1) / 5.0)

        dt = float(self.env.cfg.sim.dt * self.env.cfg.decimation)
        vel_to_gate = torch.clamp(vel_proj_gate, 0.0, self._v_cap) * dt
        vel_to_next_gate = plane_focus * torch.clamp(vel_proj_next, 0.0, self._v_cap) * dt
        vel_to_next2 = next2_focus * torch.clamp(vel_proj_next2, 0.0, self._v_cap) * dt

        # Reward carrying speed through the CURRENT gate plane in the legal direction.
        gate_normal_w = self.env._normal_vectors[idx]
        gate_normal_b = torch.einsum("eji,ej->ei", drone_rot_w, gate_normal_w)
        center_err2 = yg ** 2 + zg ** 2
        through_focus = torch.exp(-((xg / 0.40) ** 2)) * torch.exp(-(center_err2 / (0.55 ** 2)))
        through_plane_speed = torch.clamp(-torch.sum(drone_lin_vel_b * gate_normal_b, dim=1), 0.0, self._v_cap) * dt
        through_gate_speed = through_focus * through_plane_speed

        # Centering bonus: only paid on the gate-pass step (Gaussian based on offset from center)
        center_bonus = torch.exp(-center_err2 / (0.25 ** 2))
        center_at_pass = gate_pass.float() * center_bonus

        # Fine-tune the slow shared-gate segment toward a tighter, faster legal line.
        if fine_tune_phase > 0.0:
            vel_to_next_gate = torch.where(
                critical_segment,
                vel_to_next_gate * (1.0 + 0.20 * fine_tune_phase),
                vel_to_next_gate,
            )
            vel_to_next2 = torch.where(
                critical_segment,
                vel_to_next2 * (1.0 + 0.30 * fine_tune_phase),
                vel_to_next2,
            )
            through_gate_speed = torch.where(
                critical_segment,
                through_gate_speed * (1.0 + 0.25 * fine_tune_phase),
                through_gate_speed,
            )
            center_at_pass = torch.where(
                critical_segment,
                center_at_pass * (1.0 - 0.35 * fine_tune_phase),
                center_at_pass,
            )
            progress = torch.where(
                critical_segment,
                progress * (1.0 - 0.10 * fine_tune_phase),
                progress,
            )

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

        # "Finish earlier = bigger bonus"
        max_steps = float(self.env.max_episode_length)
        finish_time_frac = torch.clamp(self.env.episode_length_buf.float() / max_steps, 0.0, 1.0)
        finish_bonus_scale = 0.25 + (1.0 - finish_time_frac) ** 2

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
                self.env._robot.data.root_link_state_w[ids, :3],
            )
            self.env._prev_x_drone_wrt_gate[ids] = new_pose_gate[:, 0]
            self.env._last_distance_to_goal[ids] = torch.linalg.norm(new_pose_gate, dim=1)
            self.env._pose_drone_wrt_gate[ids] = new_pose_gate

        # Gate miss = instant termination during training (sample efficiency)
        if self.cfg.is_train and gate_miss.any():
            miss_ids = torch.where(gate_miss)[0]
            self.env._crashed[miss_ids] = 1000

        # Wrong-way / out-of-sequence gate traversal = instant termination during training
        if self.cfg.is_train and illegal_cross.any():
            bad_ids = torch.where(illegal_cross)[0]
            self.env._crashed[bad_ids] = 1000

        # Update distance buffers for environments that did NOT pass a gate
        not_pass = ~gate_pass
        self.env._prev_x_drone_wrt_gate[not_pass] = xg[not_pass]
        self.env._last_distance_to_goal[not_pass] = dist[not_pass]
        self._prev_all_x[:] = all_x

        if self.cfg.is_train:
            rewards = {
                "gate_pass": gate_pass.float() * self.env.rew["gate_pass_reward_scale"],
                "gate_miss": gate_miss.float() * self.env.rew["gate_miss_reward_scale"],
                "wrong_way": wrong_way_current.float() * self.env.rew["wrong_way_reward_scale"],
                "illegal_gate": illegal_other_gate.float() * self.env.rew["illegal_gate_reward_scale"],
                "finish": finished_this_step.float() * self.env.rew["finish_reward_scale"] * finish_bonus_scale,
                "time_penalty": torch.ones_like(dist) * (self.env.rew["time_penalty_reward_scale"] * time_mult),
                "action_l2": action_l2 * (self.env.rew["action_l2_reward_scale"] * action_mult),
                "crash": crashed * self.env.rew["crash_reward_scale"],
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

        return reward


    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations including waypoint positions and drone state."""
        curr_idx = self.env._idx_wp % self.env._waypoints.shape[0]
        next_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]

        wp_curr_pos = self.env._waypoints[curr_idx, :3]
        wp_next_pos = self.env._waypoints[next_idx, :3]
        quat_curr = self.env._waypoints_quat[curr_idx]
        quat_next = self.env._waypoints_quat[next_idx]

        rot_curr = matrix_from_quat(quat_curr)
        rot_next = matrix_from_quat(quat_next)

        verts_curr = torch.bmm(self.env._local_square, rot_curr.transpose(1, 2)) + wp_curr_pos.unsqueeze(1) + self.env._terrain.env_origins.unsqueeze(1)
        verts_next = torch.bmm(self.env._local_square, rot_next.transpose(1, 2)) + wp_next_pos.unsqueeze(1) + self.env._terrain.env_origins.unsqueeze(1)

        waypoint_pos_b_curr, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_curr.view(-1, 3)
        )
        waypoint_pos_b_next, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_next.view(-1, 3)
        )

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(self.num_envs, 4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(self.num_envs, 4, 3)

        quat_w = self.env._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        obs = torch.cat(
            [
                self.env._robot.data.root_com_lin_vel_b,			# 3 dim (linear vel in body frame)
                attitude_mat.view(attitude_mat.shape[0], -1),			# 9 dim (drone rotation matrix)
                waypoint_pos_b_curr.view(waypoint_pos_b_curr.shape[0], -1),	# 12 dim (corners of current gate)
                waypoint_pos_b_next.view(waypoint_pos_b_next.shape[0], -1),	# 12 dim (corners of next gate)
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        # Update yaw tracking
        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self.env._previous_yaw
        self.env._previous_yaw = yaw_w
        self.env._yaw_n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.env._yaw_n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.env.unwrapped_yaw = yaw_w + 2 * np.pi * self.env._yaw_n_laps

        self.env._previous_actions = self.env._actions.clone()

        return observations


    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, "_episode_sums"):
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

            # Log domain randomization strength and fine-tune phase
            extras["Curriculum/dr_strength"] = self._get_dr_strength()
            extras["Curriculum/fine_tune_phase"] = self._get_fine_tune_phase()
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

        # IMPORTANT: do NOT randomize episode_length_buf on full resets.
        # That hurts clean time-to-finish credit assignment.
        self.env.episode_length_buf[env_ids] = 0

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

        if self.cfg.is_train:
            it = int(getattr(self.env, "iteration", 0))
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

            # Reset mix:
            #   - majority gate-0 starts (to match evaluation)
            #   - some random-gate starts
            #   - later, some maneuver-focused starts around the slow shared-gate segment
            u = torch.rand(n_reset, device=self.device)
            if it >= self._fine_tune_start_iter and self._focus_gate_indices.numel() > 0:
                global_random_prob = 0.10
                focus_phase = self._get_fine_tune_phase()
                focus_prob = 0.25 + 0.10 * focus_phase
            else:
                global_random_prob = 0.15 if it >= self._global_random_reset_start_iter else 0.0
                if it >= self._focus_reset_start_iter and self._focus_gate_indices.numel() > 0:
                    focus_phase = float(np.clip((it - self._focus_reset_start_iter) / 1200.0, 0.0, 1.0))
                    focus_prob = 0.10 + 0.15 * focus_phase
                else:
                    focus_prob = 0.0

            focus_start_mask = u < focus_prob
            random_start_mask = (~focus_start_mask) & (u < (focus_prob + global_random_prob))

            if random_start_mask.any():
                waypoint_indices[random_start_mask] = torch.randint(
                    low=0,
                    high=self.env._waypoints.shape[0],
                    size=(int(random_start_mask.sum().item()),),
                    device=self.device,
                    dtype=self.env._idx_wp.dtype,
                )

            if focus_start_mask.any():
                choice_ids = torch.randint(
                    low=0,
                    high=int(self._focus_gate_indices.numel()),
                    size=(int(focus_start_mask.sum().item()),),
                    device=self.device,
                )
                waypoint_indices[focus_start_mask] = self._focus_gate_indices[choice_ids].to(
                    dtype=self.env._idx_wp.dtype
                )

            # Gate positions and headings
            x0_wp = self.env._waypoints[waypoint_indices][:, 0]
            y0_wp = self.env._waypoints[waypoint_indices][:, 1]
            z_wp = self.env._waypoints[waypoint_indices][:, 2]
            theta = self.env._waypoints[waypoint_indices][:, -1]

            # Start-position curriculum: ramp to full TA eval ranges by iter 800
            spawn_phase = float(np.clip(it / 800.0, 0.0, 1.0))
            x_min_default = -2.0 - 1.0 * spawn_phase     # -2.0 -> -3.0
            y_lim_default = 0.50 + 0.50 * spawn_phase    #  0.5 ->  1.0

            x_local = torch.empty(n_reset, device=self.device).uniform_(x_min_default, -0.5)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-y_lim_default, y_lim_default)

            # Local maneuver starts are slightly closer and tighter around the shared gate segment.
            # In the 10k fine-tune, make these starts even more targeted so the agent repeatedly
            # practices the time-losing section of the track.
            if focus_start_mask.any():
                n_focus = int(focus_start_mask.sum().item())
                if it >= self._fine_tune_start_iter:
                    x_local[focus_start_mask] = torch.empty(n_focus, device=self.device).uniform_(-1.35, -0.30)
                    y_local[focus_start_mask] = torch.empty(n_focus, device=self.device).uniform_(-0.55, 0.55)
                else:
                    x_local[focus_start_mask] = torch.empty(n_focus, device=self.device).uniform_(
                        -1.75 - 0.35 * spawn_phase, -0.35
                    )
                    y_local[focus_start_mask] = torch.empty(n_focus, device=self.device).uniform_(-0.85, 0.85)

            # Rotate local position to world frame based on gate heading
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            x_rot = cos_t * x_local - sin_t * y_local
            y_rot = sin_t * x_local + cos_t * y_local
            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y

            # Gate 0 starts on the ground (matching evaluation), others start near gate height.
            # For focused fine-tune resets, reduce height noise to make segment practice cleaner.
            z0 = z_wp + torch.empty(n_reset, device=self.device).uniform_(-0.15, 0.15)
            gate0_mask = waypoint_indices == 0
            if it >= self._fine_tune_start_iter and focus_start_mask.any():
                z0[focus_start_mask] = z_wp[focus_start_mask] + torch.empty(int(focus_start_mask.sum().item()), device=self.device).uniform_(-0.08, 0.08)
            z0 = torch.clamp(z0, min=0.35)
            z0[gate0_mask] = 0.05
            default_root_state[:, 2] = z0

            # Point drone toward the target gate with small yaw noise and mild air-start tilt noise
            yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
            yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.20, 0.20)
            if it >= self._fine_tune_start_iter and focus_start_mask.any():
                yaw_noise[focus_start_mask] = torch.empty(int(focus_start_mask.sum().item()), device=self.device).uniform_(-0.12, 0.12)
            yaw = yaw + yaw_noise

            roll = torch.zeros(n_reset, device=self.device)
            pitch = torch.zeros(n_reset, device=self.device)
            air_mask = ~gate0_mask
            tilt_phase = float(np.clip((it - 1000) / 1500.0, 0.0, 1.0))
            if tilt_phase > 0.0 and air_mask.any():
                n_air = int(air_mask.sum().item())
                roll[air_mask] = torch.empty(n_air, device=self.device).uniform_(-0.08, 0.08) * tilt_phase
                pitch[air_mask] = torch.empty(n_air, device=self.device).uniform_(-0.10, 0.10) * tilt_phase

            quat = quat_from_euler_xyz(roll, pitch, yaw)
            default_root_state[:, 3:7] = quat

            # Velocity-reset curriculum: teach the policy to recover and carry speed mid-maneuver.
            vel_phase = float(
                np.clip(
                    (it - self._velocity_reset_start_iter)
                    / float(self._velocity_reset_end_iter - self._velocity_reset_start_iter),
                    0.0,
                    1.0,
                )
            )
            if vel_phase > 0.0:
                gate_pos_w = self.env._waypoints[waypoint_indices, :3]
                next_idx = (waypoint_indices + 1) % self._num_gates
                next_gate_pos_w = self.env._waypoints[next_idx, :3]

                dir_to_gate = gate_pos_w - default_root_state[:, :3]
                dir_to_gate = dir_to_gate / (torch.linalg.norm(dir_to_gate, dim=1, keepdim=True) + 1e-6)

                dir_after_gate = next_gate_pos_w - gate_pos_w
                dir_after_gate = dir_after_gate / (torch.linalg.norm(dir_after_gate, dim=1, keepdim=True) + 1e-6)

                travel_dir = 0.75 * dir_to_gate + 0.25 * dir_after_gate
                travel_dir = travel_dir / (torch.linalg.norm(travel_dir, dim=1, keepdim=True) + 1e-6)

                speeds = torch.zeros(n_reset, device=self.device)

                if gate0_mask.any():
                    n_ground = int(gate0_mask.sum().item())
                    speeds[gate0_mask] = torch.empty(n_ground, device=self.device).uniform_(0.0, 1.2)

                air_generic_mask = air_mask & (~focus_start_mask)
                if air_generic_mask.any():
                    n_air_generic = int(air_generic_mask.sum().item())
                    speeds[air_generic_mask] = torch.empty(n_air_generic, device=self.device).uniform_(0.5, 2.8)

                if focus_start_mask.any():
                    n_focus = int(focus_start_mask.sum().item())
                    if it >= self._fine_tune_start_iter:
                        speeds[focus_start_mask] = torch.empty(n_focus, device=self.device).uniform_(2.2, 4.8)
                    else:
                        speeds[focus_start_mask] = torch.empty(n_focus, device=self.device).uniform_(1.5, 4.2)

                default_root_state[:, 7:10] = travel_dir * (speeds * vel_phase).unsqueeze(1)

        else:
            # Play/eval mode initial position
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # Rotate local position to world frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # Point drone toward the initial gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0).clone()
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0
            default_root_state[:, 7:] = 0.0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0,
            )
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
            self.env._robot.data.root_link_state_w[env_ids, :3],
        )
        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0].clone()
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._pose_drone_wrt_gate[env_ids], dim=1
        )

        # Initialize per-gate signed distances for wrong-way / illegal-gate detection
        all_pose_gate = self._get_all_gate_relative_positions(default_root_state[:, :3])
        self._prev_all_x[env_ids] = all_pose_gate[:, :, 0]

        # Domain randomization: randomize physics for each reset environment
        if self.cfg.is_train:
            self._randomize_physics(env_ids)
