#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os
import pickle

from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.rllib.models import MODEL_DEFAULTS
from utils import PathUtils, TrainingUtils
from ray.rllib.algorithms.maddpg import MADDPG, MADDPGConfig

ON_MAC = True
train_batch_size = 60000 if not ON_MAC else 200  # Jan 32768
num_workers = 5 if not ON_MAC else 0  # jan 4
num_envs_per_worker = 60 if not ON_MAC else 1  # Jan 32
rollout_fragment_length = (
    train_batch_size
    if ON_MAC
    else train_batch_size // (num_workers * num_envs_per_worker)
)

scenario_name = "navigation"
model_name = "MADDPGDefault"


def train(
    max_episode_steps,
    comm_radius,
    continuous_actions,
    seed,
):
    trainer = MADDPG
    trainer_name = "MADDPG"
    tune.run(
        trainer,
        name="MADDPG",
        callbacks=[
            # WandbLoggerCallback(
            #     project=f"{scenario_name}{'_test' if ON_MAC else ''}",
            #     api_key_file=str(PathUtils.scratch_dir / "wandb_api_key_file"),
            #     group=group_name,
            #     notes=notes,
            # )
        ],
        local_dir=str(PathUtils.scratch_dir / "ray_results" / scenario_name),
        stop={"training_iteration": 1500},
        config={
            "seed": seed,
            "framework": "torch",
            "env": scenario_name,
            "train_batch_size": train_batch_size,
            "rollout_fragment_length": rollout_fragment_length,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": num_workers,
            "num_envs_per_worker": num_envs_per_worker,
            # "batch_mode": "complete_episodes",
            "actor_hiddens": [64, 64],
            "actor_hidden_activation": "relu",
            "critic_hiddens": [64, 64],
            "critic_hidden_activation": "relu",
            "env_config": {
                "device": "cpu",
                "num_envs": num_envs_per_worker,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_episode_steps,
                # Env specific
                # scenario config checked
                "scenario_config": {
                    "n_agents": 4,
                    "lidar_range": comm_radius,
                    "agent_radius": 0.1,
                    "shared_rew": False,
                    "pos_shaping_factor": 1,
                    "final_reward": 0.1,
                    "comm_range": comm_radius,
                    "agent_collision_penalty": -.5,
                    "observe_all_goals": False,
                },
            },
            "evaluation_interval": 20,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            "evaluation_config": {
                "num_envs_per_worker": 1,
                "env_config": {
                    "num_envs": 1,
                },
                "callbacks": MultiCallbacks(
                    [
                        TrainingUtils.RenderingCallbacks,
                    ]
                ),
            },
            "callbacks": MultiCallbacks([TrainingUtils.EvaluationCallbacks]),
        }
    )


if __name__ == "__main__":
    TrainingUtils.init_ray(scenario_name=scenario_name, local_mode=ON_MAC)
    # config = MADDPGConfig()
    # print(f"config.replay_buffer_config: {config.replay_buffer_config}")
    # replay_config = config.replay_buffer_config.update(  
    # {
    #     "capacity": 100000,
    #     "prioritized_replay_alpha": 0.8,
    #     "prioritized_replay_beta": 0.45,
    #     "prioritized_replay_eps": 2e-6,
    # }
    # )
    # config.training(replay_buffer_config=replay_config)   
    # config = config.resources(num_gpus=0)   
    # config = config.rollouts(num_rollout_workers=4)   
    # config = config.environment("CartPole-v1")   
    # algo = config.build()  
    # algo.train() 

    for seed in [2]:
        train(
            seed=seed,
            # Env
            max_episode_steps=200,
            continuous_actions=True,
            comm_radius=0.55,
        )

