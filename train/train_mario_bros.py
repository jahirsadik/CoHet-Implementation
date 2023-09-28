#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import os
import pickle
from pathlib import Path

import ray
from ray import tune
# from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.rllib.models import MODEL_DEFAULTS

from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from rllib_differentiable_comms.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)

from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.atari import mario_bros_v3

from models.gppo import GPPO

ON_MAC = True
SCRATCH_DIR = Path('/Users/jahirsadikmonon/Codes/scratch/')

train_batch_size = 60000 if not ON_MAC else 200  # Jan 32768
num_workers = 5 if not ON_MAC else 0  # jan 4
num_envs_per_worker = 60 if not ON_MAC else 1  # Jan 32
rollout_fragment_length = (
    train_batch_size
    if ON_MAC
    else train_batch_size // (num_workers * num_envs_per_worker)
)
scenario_name = "mario_bros_v3"
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
    add_agent_index,
    continuous_actions,
    seed,
    notes,
):

    fcnet_model_config = MODEL_DEFAULTS.copy()
    fcnet_model_config.update({"vf_share_layers": False})

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

    trainer = MultiPPOTrainer
    trainer_name = "MultiPPOTrainer" if trainer is MultiPPOTrainer else "PPOTrainer"
    tune.run(
        trainer,
        name=group_name if model_name.startswith("GPPO") else model_name,
        callbacks=[
            # WandbLoggerCallback(
            #     project=f"{scenario_name}{'_test' if ON_MAC else ''}",
            #     api_key_file=str(PathUtils.scratch_dir / "wandb_api_key_file"),
            #     group=group_name,
            #     notes=notes,
            # )
        ],
        local_dir=str(SCRATCH_DIR / "ray_results" / scenario_name),
        stop={"training_iteration": 5000},
        restore=None,
        config={
            "framework": "torch",
            "env": scenario_name,
            "num_workers": 1,
            "num_gpus": 0,
            "train_batch_size": 5000,
            "lr": 0.0003,
            "gamma": 0.99,
            "_enable_rl_module_api": False,
            "_enable_learner_api": False,
            "model": {
                "custom_model": "GPPO",
                "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "activation_fn": "relu",
                    "share_observations": share_observations,
                    "gnn_type": "MatPosConv",
                    "centralised_critic": centralised_critic,
                    "heterogeneous": heterogeneous,
                    "use_beta": False,
                    "aggr": aggr,
                    "topology_type": None,
                    "use_mlp": use_mlp,
                    "add_agent_index": add_agent_index,
                    "pos_start": 0,
                    "pos_dim": 2,
                    "vel_start": 2,
                    "vel_dim": 2,
                    "goal_rel_start": 4,
                    "goal_rel_dim": 2,
                    "trainer": trainer_name,
                    "share_action_value": share_action_value,
                    # cohet
                    # "alignment_type": alignment_type,
                    "comm_radius": comm_radius,
                    # "dyn_model_hidden_units": dyn_model_hidden_units,
                    # "dyn_model_layer_num": dyn_model_layer_num,
                    # "intr_rew_beta": intr_rew_beta,
                    # "intr_beta_type": intr_beta_type,
                    # "intr_rew_weighting": intr_rew_weighting,
                    # cohet end
                }
                if model_name == "GPPO"
                else fcnet_model_config,
            },
            # "env_config": {
            #     "device": "cpu",
            #     "num_envs": num_envs_per_worker,
            #     "scenario_name": scenario_name,
            #     "continuous_actions": continuous_actions,
            #     "max_steps": max_episode_steps,
            # }
        }
    )


def make_env():
    env = mario_bros_v3.env()
    # env.observation_space = spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
    return PettingZooEnv(env)


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(
            _temp_dir=str(SCRATCH_DIR / "ray"),
            local_mode=ON_MAC,
        )
        print("Ray init!")
        register_env(scenario_name, lambda config: make_env())
        ModelCatalog.register_custom_model("GPPO", GPPO)
        ModelCatalog.register_custom_action_dist(
            "hom_multi_action", TorchHomogeneousMultiActionDistribution
        )

    for seed in [2]:
        train(
            seed=seed,
            restore=False,
            notes="",
            # Model important
            share_observations=False,
            share_action_value=False,
            heterogeneous=True,
            # Other model
            centralised_critic=False,
            use_mlp=False,
            add_agent_index=False,
            aggr="add",
            topology_type=None,
            # cohet
            alignment_type=None,
            comm_radius=0.45,
            dyn_model_hidden_units=128,
            dyn_model_layer_num=2,
            intr_rew_beta=20,
            intr_beta_type="percent",
            intr_rew_weighting="distance",
            # cohet end
            # Env
            max_episode_steps=200,
            continuous_actions=False,
        )

