#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os
import pickle

from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import MultiCallbacks

from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer
from utils import PathUtils, TrainingUtils

ON_MAC = True

train_batch_size = 60000 if not ON_MAC else 200  # Jan 32768
num_workers = 5 if not ON_MAC else 0  # jan 4
num_envs_per_worker = 50 if not ON_MAC else 1  # Jan 32
rollout_fragment_length = (
    train_batch_size
    if ON_MAC
    else train_batch_size // (num_workers * num_envs_per_worker)
)
scenario_name = "joint_passage"
model_name = "GPPO"


def train(
    share_observations,
    centralised_critic,
    restore,
    heterogeneous,
    max_episode_steps,
    use_mlp,
    aggr,
    topology_type,
    add_agent_index,
    continuous_actions,
    seed,
    notes,
    share_action_value,
    # cohet
    alignment_type,
    comm_radius,
    dyn_model_hidden_units,
    dyn_model_layer_num,
    intr_rew_beta,
    intr_rew_weighting,
    intr_beta_type,
    # cohet end
):
    checkpoint_rel_path = "ray_results/joint/GIPPO/MultiPPOTrainer_joint_9b1c9_00000_0_2022-09-01_10-08-12/checkpoint_000099/checkpoint-99"
    checkpoint_path = PathUtils.scratch_dir / checkpoint_rel_path
    params_path = checkpoint_path.parent.parent / "params.pkl"

    if centralised_critic and not use_mlp:
        if share_observations:
            group_name = "GAPPO"
        else:
            group_name = "MAPPO"
    elif use_mlp:
        group_name = "CPPO"
    elif share_observations:
        group_name = "GPPO"
    else:
        group_name = "IPPO"

    group_name = f"{'Het' if heterogeneous else ''}{group_name}"

    if restore:
        with open(params_path, "rb") as f:
            config = pickle.load(f)

    trainer = MultiPPOTrainer
    trainer_name = "MultiPPOTrainer" if trainer is MultiPPOTrainer else "PPOTrainer"
    tune.run(
        trainer,
        name=group_name if model_name == "GPPO" else model_name,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        max_failures=0,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        callbacks=[
            WandbLoggerCallback(
                project=f"{scenario_name}{'_test' if ON_MAC else ''}",
                api_key_file=str(PathUtils.scratch_dir / "wandb_api_key_file"),
                group=group_name,
                notes=notes,
            )
        ],
        local_dir=str(PathUtils.scratch_dir / "ray_results" / scenario_name),
        stop={"training_iteration": 600},
        restore=str(checkpoint_path) if restore else None,
        config={
            "seed": seed,
            "framework": "torch",
            "env": scenario_name,
            "kl_coeff": 0.01,
            "kl_target": 0.01,
            "lambda": 0.9,
            "clip_param": 0.2,  # 0.3
            "vf_loss_coeff": 1,  # Jan 0.001
            "vf_clip_param": float("inf"),
            "entropy_coeff": 0,  # 0.01,
            "train_batch_size": train_batch_size,
            "rollout_fragment_length": rollout_fragment_length,
            "sgd_minibatch_size": 4096 if not ON_MAC else 100,  # jan 2048
            "num_sgd_iter": 40,  # Jan 30
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": num_workers,
            "num_envs_per_worker": num_envs_per_worker,
            "lr": 5e-5,
            "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "grad_clip": 40,
            "batch_mode": "complete_episodes",
            "model": {
                "vf_share_layers": share_action_value,
                "custom_model": model_name,
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "activation_fn": "tanh",
                    "share_observations": share_observations,
                    "gnn_type": "MatPosConv",
                    "centralised_critic": centralised_critic,
                    "heterogeneous": heterogeneous,
                    "use_beta": False,
                    "aggr": aggr,
                    "topology_type": topology_type,
                    "use_mlp": use_mlp,
                    "add_agent_index": add_agent_index,
                    "pos_start": 0,
                    "pos_dim": 2,
                    "vel_start": 2,
                    "vel_dim": 2,
                    "share_action_value": share_action_value,
                    "trainer": trainer_name,
                },
            },
            "env_config": {
                "device": "cpu",
                "num_envs": num_envs_per_worker,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_episode_steps,
                # Env specific
                # Scenario configured
                "scenario_config": {
                    "fixed_passage": True,
                    "n_passages": 1,
                    "random_start_angle": True,
                    "random_goal_angle": True,
                    "pos_shaping_factor": 1,
                    "rot_shaping_factor": 1,
                    "collision_reward": -0.06,  # -0.06
                    "energy_reward_coeff": 0,
                    "joint_length": 0.5,
                    "observe_joint_angle": True,
                    "joint_angle_obs_noise": 0.0,
                    "asym_package": True,
                    "mass_ratio": 5,
                    "mass_position": 0.75,
                    "max_speed_1": None,  # 0.05
                    "all_passed_rot": True,
                    "obs_noise": 0.0001,
                    "use_controller": False,
                    # cohet
                    "alignment_type": alignment_type,
                    "comm_radius": comm_radius,
                    "dyn_model_hidden_units": dyn_model_hidden_units,
                    "dyn_model_layer_num": dyn_model_layer_num,
                    "intr_rew_beta": intr_rew_beta,
                    "intr_beta_type": intr_beta_type,
                    "intr_rew_weighting": intr_rew_weighting,
                    # cohet end
                },
            },
            "evaluation_interval": 20,
            "evaluation_duration": 3,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": False,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
                "callbacks": MultiCallbacks(
                    [
                        TrainingUtils.RenderingCallbacks,
                        TrainingUtils.EvaluationCallbacks,
                        TrainingUtils.HeterogeneityMeasureCallbacks,
                    ]
                ),
            },
            "callbacks": MultiCallbacks(
                [
                    TrainingUtils.EvaluationCallbacks,
                ]
            ),
        }
        if not restore
        else config,
    )


if __name__ == "__main__":
    TrainingUtils.init_ray(scenario_name=scenario_name, local_mode=True)

    for seed in [0, 1, 2]:
        train(
            seed=seed,
            restore=False,
            notes="",
            # Model important
            share_observations=True,
            heterogeneous=True,
            # Other model
            share_action_value=False,
            centralised_critic=False,
            use_mlp=False,
            add_agent_index=False,
            aggr="add",
            topology_type=None,
            # cohet
            alignment_type="team",
            comm_radius=0.75,
            dyn_model_hidden_units=128,
            dyn_model_layer_num=2,
            intr_rew_beta=20,
            intr_beta_type="percent",
            intr_rew_weighting="distance",
            # cohet end
            # Env
            max_episode_steps=200,
            continuous_actions=True,
        )
