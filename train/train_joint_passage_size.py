#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os
import pickle

from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import MultiCallbacks, DefaultCallbacks

from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer
from utils import PathUtils, TrainingUtils

ON_MAC = True

train_batch_size = 60000 if not ON_MAC else 200  # Jan 32768
num_workers = 10 if not ON_MAC else 0  # jan 4
num_envs_per_worker = 20 if not ON_MAC else 1  # Jan 32
rollout_fragment_length = (
    train_batch_size
    if ON_MAC
    else train_batch_size // (num_workers * num_envs_per_worker)
)
scenario_name = "joint_passage_size"
model_name = "GPPO"


class CurriculumReward(DefaultCallbacks):
    def on_train_result(self, algorithm, result, **kwargs):
        def set_n_passages(env, val, pos_factor, collision_reward):
            env.scenario.set_n_passages(val)
            if pos_factor is not None:
                env.scenario.pos_shaping_factor = pos_factor
            if collision_reward is not None:
                env.scenario.collision_reward = collision_reward

        def set_new_trainer_passage(val, pos_factor=None, collision_reward=None):
            algorithm.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: set_n_passages(env, val, pos_factor, collision_reward)
                )
            )
            algorithm.evaluation_workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: set_n_passages(env, val, pos_factor, collision_reward)
                )
            )

        if result["training_iteration"] == 1:
            set_new_trainer_passage(val=4)
        elif result["training_iteration"] == 100:
            set_new_trainer_passage(val=3, pos_factor=0, collision_reward=-0.1)
            algorithm.get_policy().entropy_coeff = 0.01
        elif result["training_iteration"] == 200:
            set_new_trainer_passage(val=3, pos_factor=1, collision_reward=0)
            algorithm.get_policy().entropy_coeff = 0
        elif result["training_iteration"] == 700:
            set_new_trainer_passage(val=4)
        elif result["training_iteration"] == 900:
            set_new_trainer_passage(val=3)


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
    curriculum,
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
        callbacks=[
            # WandbLoggerCallback(
            #     project=f"{scenario_name}{'_test' if ON_MAC else ''}",
            #     api_key_file=str(PathUtils.scratch_dir / "wandb_api_key_file"),
            #     group=group_name,
            #     notes=notes,
            # )
        ],
        local_dir=str(PathUtils.scratch_dir / "ray_results" / scenario_name),
        stop={"training_iteration": 1200},
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
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
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
                    "curriculum": curriculum,
                },
            },
            "env_config": {
                "device": "cpu",
                "num_envs": num_envs_per_worker,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_episode_steps,
                # Env specific
                "scenario_config": {
                    "fixed_passage": False,
                    "random_start_angle": False,
                    "random_goal_angle": False,
                    "pos_shaping_factor": 1,
                    "rot_shaping_factor": 1,
                    "collision_reward": 0,  # -0.1,
                    "energy_reward_coeff": 0,
                    "observe_joint_angle": False,
                    "joint_angle_obs_noise": 0.0,
                    "asym_package": False,
                    "mass_ratio": 1,
                    "mass_position": 0.75,
                    "max_speed_1": None,  # 0.05
                    "obs_noise": 0.0,
                    "n_passages": 4,
                    "middle_angle_180": True,
                },
            },
            "evaluation_interval": 30,
            "evaluation_duration": 3,
            "evaluation_sample_timeout_s": 360,
            "evaluation_num_workers": 2,
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
                [TrainingUtils.EvaluationCallbacks]
                + ([CurriculumReward] if curriculum else [])
            ),
        }
        if not restore
        else config,
    )


if __name__ == "__main__":
    TrainingUtils.init_ray(scenario_name=scenario_name, local_mode=True)
    for seed in [6]:
        train(
            seed=seed,
            restore=False,
            notes="",
            curriculum=True,
            # Model important
            share_observations=True,
            heterogeneous=True,
            # Other model
            share_action_value=True,
            centralised_critic=False,
            use_mlp=False,
            add_agent_index=False,
            aggr="add",
            topology_type="full",
            # Env
            max_episode_steps=300,
            continuous_actions=True,
        )
