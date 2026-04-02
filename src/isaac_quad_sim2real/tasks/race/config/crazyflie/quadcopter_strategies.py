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

        # Previous speeds for reward shaping
        self._prev_speed = torch.zeros(self.num_envs, device=self.device)

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

        # Initialize fixed parameters once (DR will overwrite per-env in reset_idx)
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants and thrust to weight ratio
        self.env._tau_m[:] = self.env._tau_m_value
        self.env._thrust_to_weight[:] = self.env._twr_value

        # Gate geometry (square opening in gate frame y/z)
        self._gate_half_side = float(self.cfg.gate_model.gate_side) / 2.0

        # --------------------------
        # Powerloop virtual checkpoint chain
        # --------------------------
        # Gate 2: (-0.625, 0, 0.75) yaw=pi/2 → Gate 3: (0.625, 0, 0.75) yaw=pi/2
        # Both face +Y direction. The powerloop goes UP and OVER instead of around.
        # These checkpoints define the arc. When targeting gate 3, progress reward
        # points at the NEXT CHECKPOINT, not at gate 3 directly.
        self._powerloop_checkpoints = torch.tensor([
            [-0.50,  -0.30, 1.20],   # CP0: climb from -Y (shallower)
            [-0.15,  -0.20, 1.80],   # CP1: mid-climb
            [ 0.15,   0.00, 2.00],   # CP2: apex (lower = safer)
            [ 0.45,   0.20, 1.40],   # CP3: descent
            [ 0.625,  0.35, 0.85],   # CP4: gate 3 approach from +Y
        ], device=self.device)
        self._num_checkpoints = self._powerloop_checkpoints.shape[0]
        self._checkpoint_radius = 0.7  # larger tolerance for path variation under DR
        self._loop_center = torch.tensor([0.0, 0.0, 1.5], device=self.device)
        self._loop_radius = 0.5

        # Per-env tracking: which checkpoint each env is targeting
        self._checkpoint_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._prev_dist_to_checkpoint = torch.full((self.num_envs,), 5.0, device=self.device)
        self._prev_angle = torch.full((self.num_envs,), 0.0, device=self.device)
        # Altitude validator: track max altitude while targeting gate 3
        self._max_z_during_gate3 = torch.zeros(self.num_envs, device=self.device)

        # --------------------------
        # Curriculums / constants
        # --------------------------
        self._gate_margin_start = 0.05
        self._gate_margin_end = 0.02
        self._speed_phase_start_iter = 600
        self._speed_phase_end_iter = 2400
        self._lap_1_end_iter = 800
        self._lap_2_end_iter = 1600
        self._v_cap = 20.0

    # ------------------------------------------------------------------
    # Domain randomization helpers
    # ------------------------------------------------------------------
    def _get_dr_strength(self) -> float:
        """Get current DR strength in [0, 1]. Ramps linearly over first 1500 iterations."""
        # it = int(getattr(self.env, "iteration", 0))
        # dr_ramp_end = 1500
        # if dr_ramp_end <= 0:
        #     return 1.0
        # return float(np.clip(it / float(dr_ramp_end), 0.0, 1.0))
        return 0.0

    def _randomize_physics(self, env_ids: torch.Tensor):
        """Randomize physics for reset environments to match TA evaluation ranges."""
        s = self._get_dr_strength()
        if s < 1e-6:
            return

        n = len(env_ids)
        device = self.device

        def _rand_scale(low, high, size):
            raw = torch.empty(size, device=device).uniform_(low, high)
            return 1.0 + s * (raw - 1.0)

        self.env._thrust_to_weight[env_ids] = self.env._twr_value * _rand_scale(0.95, 1.05, n)

        aero_xy = _rand_scale(0.5, 2.0, n)
        aero_z = _rand_scale(0.5, 2.0, n)
        self.env._K_aero[env_ids, 0] = self.env._k_aero_xy_value * aero_xy
        self.env._K_aero[env_ids, 1] = self.env._k_aero_xy_value * aero_xy
        self.env._K_aero[env_ids, 2] = self.env._k_aero_z_value * aero_z

        kp_rp = _rand_scale(0.85, 1.15, n)
        ki_rp = _rand_scale(0.85, 1.15, n)
        kd_rp = _rand_scale(0.70, 1.30, n)
        self.env._kp_omega[env_ids, 0] = self.env._kp_omega_rp_value * kp_rp
        self.env._kp_omega[env_ids, 1] = self.env._kp_omega_rp_value * kp_rp
        self.env._ki_omega[env_ids, 0] = self.env._ki_omega_rp_value * ki_rp
        self.env._ki_omega[env_ids, 1] = self.env._ki_omega_rp_value * ki_rp
        self.env._kd_omega[env_ids, 0] = self.env._kd_omega_rp_value * kd_rp
        self.env._kd_omega[env_ids, 1] = self.env._kd_omega_rp_value * kd_rp

        kp_y = _rand_scale(0.85, 1.15, n)
        ki_y = _rand_scale(0.85, 1.15, n)
        kd_y = _rand_scale(0.70, 1.30, n)
        self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value * kp_y
        self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value * ki_y
        self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value * kd_y

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure

        # Speed curriculum
        it = int(getattr(self.env, "iteration", 0))
        if self._speed_phase_end_iter <= self._speed_phase_start_iter:
            speed_phase = 1.0
        else:
            speed_phase = float(np.clip(
                (it - self._speed_phase_start_iter) /
                float(self._speed_phase_end_iter - self._speed_phase_start_iter),
                0.0, 1.0))

        center_mult = 1.0 - 0.9 * speed_phase
        action_mult = 1.0 - 0.8 * speed_phase
        vel_mult    = speed_phase
        time_mult   = 1.0 + 10.0 * speed_phase
        death_mult  = 1.0 + 3.0 * speed_phase

        margin = float(self._gate_margin_start +
                       (self._gate_margin_end - self._gate_margin_start) * speed_phase)

        # Lap curriculum
        laps_target = int(self.cfg.max_n_laps)
        if self.cfg.is_train:
            if it < self._lap_1_end_iter:
                laps_target = 1
            elif it < self._lap_2_end_iter:
                laps_target = min(2, laps_target)

        # Get drone state
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        _, pitch, _ = euler_xyz_from_quat(drone_quat_w)
        pitch = wrap_to_pi(pitch)
        idx = self.env._idx_wp

        # Drone position in current gate frame
        pose_gate = self.env._pose_drone_wrt_gate
        xg, yg, zg = pose_gate[:, 0], pose_gate[:, 1], pose_gate[:, 2]

        # ============================================================
        # PROGRESS REWARD — different for gate 3 (powerloop) vs others
        # ============================================================
        dist = torch.linalg.norm(pose_gate, dim=1)
        prev_dist = self.env._last_distance_to_goal

        # Standard progress: distance reduction to current gate
        standard_progress = torch.clamp(prev_dist - dist, -0.50, 0.50)

        # Powerloop progress: distance reduction to CURRENT CHECKPOINT (not gate 3)
        is_gate3 = (idx == 3)

        if is_gate3.any() and self.cfg.is_train:
            # Get target checkpoint position for each env
            cp_idx = torch.clamp(self._checkpoint_idx, 0, self._num_checkpoints - 1)
            cp_target = self._powerloop_checkpoints[cp_idx]  # (num_envs, 3)

            # Distance to current checkpoint
            dist_to_cp = torch.linalg.norm(drone_pos_w - cp_target, dim=1)

            # Progress = got closer to checkpoint
            cp_progress = torch.clamp(
                self._prev_dist_to_checkpoint - dist_to_cp, -0.50, 0.50)

            # Check if checkpoint reached — advance to next
            reached = (dist_to_cp < self._checkpoint_radius) & is_gate3
            if reached.any():
                reached_ids = torch.where(reached)[0]
                self._checkpoint_idx[reached_ids] = torch.clamp(
                    self._checkpoint_idx[reached_ids] + 1, max=self._num_checkpoints - 1)
                # Reset distance to new target
                new_cp = self._powerloop_checkpoints[self._checkpoint_idx[reached_ids]]
                dist_to_cp[reached_ids] = torch.linalg.norm(
                    drone_pos_w[reached_ids] - new_cp, dim=1)

            self._prev_dist_to_checkpoint = torch.where(
                is_gate3, dist_to_cp, self._prev_dist_to_checkpoint)

            loop_error = drone_pos_w - self._loop_center
            radius = torch.linalg.norm(loop_error, dim=1)
            circle_reward = torch.exp(-(radius - self._loop_radius) ** 2 / 0.5)

            # angle = torch.atan2(-loop_error[:, 1], loop_error[:, 2])
            # angle_delta = wrap_to_pi(angle - self._prev_angle)
            # self._prev_angle = torch.where(is_gate3, angle, self._prev_angle)
            # angle_progress = torch.clamp(angle_delta, -0.5, 0.5)

            at_top = drone_pos_w[:,2] > self._loop_center[2] + self._loop_radius * 0.75
            flip_alignment = torch.cos(pitch)
            flip_reward = at_top * torch.clamp(-flip_alignment, -1.0, 1.0)

            tangent = torch.stack([-loop_error[:, 2], loop_error[:, 0], torch.zeros_like(loop_error[:, 0])], dim=1)
            tangent = tangent / (torch.linalg.norm(tangent, dim=1, keepdim=True) + 1e-6)

            tangent_vel = torch.sum(drone_lin_vel_b * tangent, dim=1)

            flow_reward = torch.clamp(tangent_vel, 0.0, self._v_cap)

            is_below = is_gate3 & (drone_pos_w[:, 2] < self._loop_center[2] - self._loop_radius * 0.5)
            entry_reward = (is_below & cp_idx == 0) * torch.clamp(pitch, 0.0, 1.0)
            exit_reward = (is_below & cp_idx == self._num_checkpoints - 1) * torch.exp(-pitch**2)

            # Update altitude tracker
            self._max_z_during_gate3 = torch.where(
                is_gate3,
                torch.max(self._max_z_during_gate3, drone_pos_w[:, 2]),
                self._max_z_during_gate3)

            # REPLACE progress for gate 3 envs with checkpoint progress
            progress = torch.where(is_gate3, cp_progress, standard_progress)
            circle_reward = is_gate3 * circle_reward
            flip_reward = is_gate3 * flip_reward
        else:
            progress = standard_progress
            circle_reward = torch.zeros_like(standard_progress)
            flip_reward = torch.zeros_like(standard_progress)
            flow_reward = torch.zeros_like(standard_progress)
            entry_reward = torch.zeros_like(standard_progress)
            exit_reward = torch.zeros_like(standard_progress)

        # Gate pass detection
        prev_xg = self.env._prev_x_drone_wrt_gate
        crossed_plane = (prev_xg > 0.0) & (xg <= 0.0)
        meaningful_cross = (prev_xg - xg) > 0.01
        inside_opening = ((yg.abs() <= (self._gate_half_side - margin)) &
                          (zg.abs() <= (self._gate_half_side - margin)))
        gate_pass = crossed_plane & inside_opening & meaningful_cross
        gate_miss = crossed_plane & (~inside_opening)

        # Altitude gate validation for gate 3:
        # Only count gate 3 pass if drone reached altitude > 1.8m (powerloop verified)
        if self.cfg.is_train:
            gate3_pass = gate_pass & (idx == 3)
            gate3_no_loop = gate3_pass & (self._max_z_during_gate3 < 1.4)
            if gate3_no_loop.any():
                # Drone passed gate 3 without doing the powerloop → terminate
                cheat_ids = torch.where(gate3_no_loop)[0]
                self.env._crashed[cheat_ids] = 1000
                gate_pass = gate_pass & (~gate3_no_loop)

        # Crash detection
        contact_forces = self.env._contact_sensor.data.net_forces_w
        contact_mag = torch.linalg.norm(contact_forces, dim=-1).squeeze(1)
        contact = contact_mag > 1e-6
        contact_mask = self.env.episode_length_buf > 10
        crashed = (contact & contact_mask).float()
        self.env._crashed = self.env._crashed + (crashed > 0).int()

        # Action smoothness penalty
        action_l2 = torch.sum(self.env._actions ** 2, dim=1)

        # Velocity shaping
        gate_pos_w = self.env._waypoints[idx, :3]
        next_idx = (idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_idx, :3]

        gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, gate_pos_w)
        next_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_pos_w)

        gate_dir_b = gate_pos_b / (torch.linalg.norm(gate_pos_b, dim=1, keepdim=True) + 1e-6)
        next_gate_dir_b = next_gate_pos_b / (torch.linalg.norm(next_gate_pos_b, dim=1, keepdim=True) + 1e-6)

        vel_proj_gate = torch.sum(drone_lin_vel_b * gate_dir_b, dim=1)
        vel_proj_next = torch.sum(drone_lin_vel_b * next_gate_dir_b, dim=1)

        plane_focus = torch.exp(-xg.abs() / 0.5)
        dt = float(self.env.cfg.sim.dt * self.env.cfg.decimation)
        vel_to_gate = torch.clamp(vel_proj_gate, 0.0, self._v_cap) * dt
        vel_to_next_gate = plane_focus * torch.clamp(vel_proj_next, 0.0, self._v_cap) * dt

        # For gate 3: velocity toward CHECKPOINT instead of gate
        if is_gate3.any() and self.cfg.is_train:
            cp_target = self._powerloop_checkpoints[
                torch.clamp(self._checkpoint_idx, 0, self._num_checkpoints - 1)]
            cp_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, cp_target)
            cp_dir_b = cp_pos_b / (torch.linalg.norm(cp_pos_b, dim=1, keepdim=True) + 1e-6)
            vel_proj_cp = torch.sum(drone_lin_vel_b * cp_dir_b, dim=1)
            vel_to_cp = torch.clamp(vel_proj_cp, 0.0, self._v_cap) * dt
            # Replace vel_to_gate for gate 3 envs with velocity toward checkpoint
            vel_to_gate = torch.where(is_gate3, vel_to_cp, vel_to_gate)

        # Speed bonus
        speed = torch.linalg.norm(drone_lin_vel_b, dim=1)
        speed_bonus = torch.clamp(speed / self._v_cap, 0.0, 1.0) * dt

        # Centering bonus
        center_err2 = yg ** 2 + zg ** 2
        center_bonus = torch.exp(-center_err2 / (0.25 ** 2))
        center_at_pass = gate_pass.float() * center_bonus

        # Finish detection
        num_gates = self.env._waypoints.shape[0]
        n_before = self.env._n_gates_passed
        finish_before = ((n_before - 1) // num_gates >= laps_target)
        n_after = n_before + gate_pass.int()
        finish_after = ((n_after - 1) // num_gates >= laps_target)
        finished_this_step = finish_after & (~finish_before)

        self.env._finished = self.env._finished | finished_this_step

        if finished_this_step.any():
            finish_ids = torch.where(finished_this_step)[0]
            self.env._crashed[finish_ids] = 10000

        max_steps = float(self.env.max_episode_length)
        finish_time_frac = torch.clamp(self.env.episode_length_buf.float() / max_steps, 0.0, 1.0)
        finish_bonus_scale = (1.0 - finish_time_frac) ** 2

        # Waypoint bookkeeping
        if gate_pass.any():
            ids = torch.where(gate_pass)[0]
            self.env._n_gates_passed[ids] += 1
            new_idx = (self.env._idx_wp[ids] + 1) % num_gates
            self.env._idx_wp[ids] = new_idx
            self.env._desired_pos_w[ids] = self.env._waypoints[new_idx, :3]

            new_pose_gate, _ = subtract_frame_transforms(
                self.env._waypoints[new_idx, :3],
                self.env._waypoints_quat[new_idx, :],
                self.env._robot.data.root_link_state_w[ids, :3])
            self.env._prev_x_drone_wrt_gate[ids] = new_pose_gate[:, 0]
            self.env._last_distance_to_goal[ids] = torch.linalg.norm(new_pose_gate, dim=1)
            self.env._pose_drone_wrt_gate[ids] = new_pose_gate

            # Reset powerloop tracking when entering gate 3 segment
            entering_gate3 = (new_idx == 3)
            if entering_gate3.any():
                g3_ids = ids[entering_gate3]
                self._checkpoint_idx[g3_ids] = 0
                cp0 = self._powerloop_checkpoints[0]
                self._prev_dist_to_checkpoint[g3_ids] = torch.linalg.norm(
                    drone_pos_w[g3_ids] - cp0.unsqueeze(0), dim=1)
                self._max_z_during_gate3[g3_ids] = drone_pos_w[g3_ids, 2]

        # Gate miss = instant termination during training
        if self.cfg.is_train and gate_miss.any():
            miss_ids = torch.where(gate_miss)[0]
            self.env._crashed[miss_ids] = 1000

        # Update distance buffers for non-passed envs
        not_pass = ~gate_pass
        self.env._prev_x_drone_wrt_gate[not_pass] = xg[not_pass]
        self.env._last_distance_to_goal[not_pass] = dist[not_pass]
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards
            rewards = {
                "progress":             progress * self.env.rew["progress_reward_scale"],
                "gate_pass":            gate_pass.float() * self.env.rew["gate_pass_reward_scale"],
                "gate_miss":            gate_miss.float() * self.env.rew["gate_miss_reward_scale"],
                "center_at_pass":       center_at_pass * (self.env.rew["center_at_pass_reward_scale"] * center_mult),
                "vel_to_gate":          vel_to_gate * (self.env.rew["vel_to_gate_reward_scale"] * vel_mult),
                "vel_to_next_gate":     vel_to_next_gate * (self.env.rew["vel_to_next_gate_reward_scale"] * vel_mult),
                "speed_bonus":          speed_bonus * (self.env.rew["speed_bonus_reward_scale"] * vel_mult),
                "circle_alignment":     circle_reward * self.env.rew["circle_alignment_reward_scale"],
                "flip_alignment":       flip_reward * self.env.rew["flip_alignment_reward_scale"],
                "flow_alignment":       flow_reward * (self.env.rew["flow_alignment_reward_scale"] * vel_mult),
                "entry_alignment":      entry_reward * self.env.rew["entry_alignment_reward_scale"],
                "exit_alignment":       exit_reward * self.env.rew["exit_alignment_reward_scale"],
                "finish":               finished_this_step.float() * self.env.rew["finish_reward_scale"] * finish_bonus_scale,
                "time_penalty":         torch.ones_like(dist) * (self.env.rew["time_penalty_reward_scale"] * time_mult),
                "action_l2":            action_l2 * (self.env.rew["action_l2_reward_scale"] * action_mult),
                "crash":                crashed * self.env.rew["crash_reward_scale"],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            death_cost = self.env.rew["death_cost"] * death_mult
            failure_term = self.env.reset_terminated & (~self.env._finished)
            reward = torch.where(failure_term, torch.ones_like(reward) * death_cost, reward)

            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations."""

        # TODO ----- START -----
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_quat_w = self.env._robot.data.root_quat_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b

        idx = self.env._idx_wp
        gate_pos_w = self.env._waypoints[idx, :3]
        next_idx = (idx + 1) % self.env._waypoints.shape[0]
        next_gate_pos_w = self.env._waypoints[next_idx, :3]

        gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, gate_pos_w)
        next_gate_pos_b, _ = subtract_frame_transforms(drone_pos_w, drone_quat_w, next_gate_pos_w)

        drone_pos_gate_frame = self.env._pose_drone_wrt_gate
        gates_passed = self.env._n_gates_passed.unsqueeze(1).float()
        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START -----
            [
                gate_pos_b,
                next_gate_pos_b,
                drone_pos_gate_frame,
                drone_lin_vel_b,
                drone_ang_vel_b,
                drone_quat_w,
                self._prev_actions,
                gates_passed,
            ],
            # TODO ----- END -----
            dim=-1,
        )  # Total: 24 dimensions

        self._prev_actions = self.env._actions.detach().clone()
        observations = {"policy": obs}
        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging
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

            finished_mask = self.env._finished[env_ids]
            if torch.any(finished_mask):
                dt_step = float(self.env.cfg.sim.dt * self.env.cfg.decimation)
                mean_ft = (self.env.episode_length_buf[env_ids][finished_mask].float() * dt_step).mean()
                extras["Episode_Finish/finish_time_s"] = float(mean_ft.item())
            else:
                extras["Episode_Finish/finish_time_s"] = 0.0

            extras["Curriculum/dr_strength"] = self._get_dr_strength()
            self.env.extras["log"].update(extras)

        # Call robot reset
        self.env._robot.reset(env_ids)

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

        # Reset buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 7:] = 0.0

        # TODO ----- START -----
        if self.cfg.is_train:
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            if getattr(self.env, "iteration", 0) >= 500:
                rand_mask = torch.rand(n_reset, device=self.device) < 0.20
                if rand_mask.any():
                    waypoint_indices[rand_mask] = torch.randint(
                        low=0, high=self.env._waypoints.shape[0],
                        size=(int(rand_mask.sum().item()),),
                        device=self.device, dtype=self.env._idx_wp.dtype)

            x0_wp = self.env._waypoints[waypoint_indices][:, 0]
            y0_wp = self.env._waypoints[waypoint_indices][:, 1]
            z_wp = self.env._waypoints[waypoint_indices][:, 2]
            theta = self.env._waypoints[waypoint_indices][:, -1]

            it = int(getattr(self.env, "iteration", 0))
            phase = float(np.clip(it / 800.0, 0.0, 1.0))
            x_min = -2.0 - 1.0 * phase
            y_lim = 0.5 + 0.5 * phase

            x_local = torch.empty(n_reset, device=self.device).uniform_(x_min, -0.5)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-y_lim, y_lim)

            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            x_rot = cos_t * x_local - sin_t * y_local
            y_rot = sin_t * x_local + cos_t * y_local
            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot

            z0 = torch.where(waypoint_indices == 0,
                             torch.full((n_reset,), 0.05, device=self.device), z_wp)

            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = z0

            yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
            yaw = yaw + torch.empty(n_reset, device=self.device).uniform_(-0.20, 0.20)
            quat = quat_from_euler_xyz(
                torch.zeros(n_reset, device=self.device),
                torch.zeros(n_reset, device=self.device), yaw)
            default_root_state[:, 3:7] = quat

            # Powerloop spawn curriculum: after iter 500, 30% spawn at powerloop apex
            if getattr(self.env, "iteration", 0) >= 500:
                apex_mask = torch.rand(n_reset, device=self.device) < 0.30
                if apex_mask.any():
                    n_apex = int(apex_mask.sum().item())
                    waypoint_indices[apex_mask] = 3
                    # Spawn near checkpoint 2 (apex area)
                    default_root_state[apex_mask, 0] = torch.empty(n_apex, device=self.device).uniform_(-0.2, 0.2)
                    default_root_state[apex_mask, 1] = torch.empty(n_apex, device=self.device).uniform_(-0.1, 0.3)
                    default_root_state[apex_mask, 2] = torch.empty(n_apex, device=self.device).uniform_(1.4, 2.2)
                    g3_pos = self.env._waypoints[3, :2]
                    yaw_apex = torch.atan2(
                        g3_pos[1] - default_root_state[apex_mask, 1],
                        g3_pos[0] - default_root_state[apex_mask, 0])
                    quat_apex = quat_from_euler_xyz(
                        torch.zeros(n_apex, device=self.device),
                        torch.zeros(n_apex, device=self.device), yaw_apex)
                    default_root_state[apex_mask, 3:7] = quat_apex
        # TODO ----- END -----

        # Handle play mode
        if not self.cfg.is_train:
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

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

        # Apply state
        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids] = self.env._waypoints[waypoint_indices, :3].clone()
        self.env._n_gates_passed[env_ids] = 0

        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._yaw_n_laps[env_ids] = 0
        self.env._crashed[env_ids] = 0
        self.env._finished[env_ids] = False

        # Reset powerloop tracking
        self._checkpoint_idx[env_ids] = 0
        self._prev_dist_to_checkpoint[env_ids] = 5.0
        self._max_z_during_gate3[env_ids] = 0.0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3])
        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0].clone()
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._pose_drone_wrt_gate[env_ids], dim=1)

        # # Domain randomization
        # if self.cfg.is_train:
        #     self._randomize_physics(env_ids)