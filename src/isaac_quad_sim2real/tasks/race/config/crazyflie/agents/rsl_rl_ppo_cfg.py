# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Longer rollouts help the policy optimize multi-gate maneuvers such as the powerloop.
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 50
    experiment_name = "quadcopter_direct_powerloop_speed"
    empirical_normalization = False
    wandb_project = "ese651_quadcopter"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128, 128],
        activation="elu",
        min_std=0.02,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
