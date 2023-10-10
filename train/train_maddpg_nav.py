import os
import ray
import vmas
from ray import tune
from typing import Dict
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import MultiCallbacks
from utils import PathUtils, TrainingUtils
from ray.tune import run_experiments
from ray.tune.registry import register_trainable, register_env
import ray.rllib.algorithms.maddpg.maddpg as maddpg
import argparse
from ray.rllib.models import ModelCatalog
from vmas import make_env
from models.fcnet import MyFullyConnectedNetwork
from vmas.simulator.environment import Environment
from rllib_differentiable_comms.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)

ON_MAC = True
scenario_name = "navigation"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
train_batch_size = 60000 if not ON_MAC else 200  # Jan 32768
num_workers = 5 if not ON_MAC else 0  # jan 4
num_envs_per_worker = 60 if not ON_MAC else 1  # Jan 32
rollout_fragment_length = (
    train_batch_size
    if ON_MAC
    else train_batch_size // (num_workers * num_envs_per_worker)
)


class CustomStdOut(object):
    def _log_result(self, result):
        if result["training_iteration"] % 50 == 0:
            try:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    result["timesteps_total"],
                    result["episodes_total"],
                    result["episode_reward_mean"],
                    result["policy_reward_mean"],
                    round(result["time_total_s"] - self.cur_time, 3)
                ))
            except:
                pass

            self.cur_time = result["time_total_s"]


def parse_args():
    parser = argparse.ArgumentParser("MADDPG with OpenAI MPE")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple",
                        choices=['simple', 'simple_speaker_listener',
                                 'simple_crypto', 'simple_push',
                                 'simple_tag', 'simple_spread', 'simple_adversary'],
                        help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25,
                        help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000,
                        help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0,
                        help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg",
                        help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg",
                        help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    # NOTE: 1 iteration = sample_batch_size * num_workers timesteps * num_envs_per_worker
    parser.add_argument("--sample-batch-size", type=int, default=25,
                        help="number of data points sampled /update /worker")
    parser.add_argument("--train-batch-size", type=int, default=1024,
                        help="number of data points /update")
    parser.add_argument("--n-step", type=int, default=1,
                        help="length of multistep value backup")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the mlp")
    parser.add_argument("--replay-buffer", type=int, default=1000000,
                        help="size of replay buffer in training")

    # Checkpoint
    parser.add_argument("--checkpoint-freq", type=int, default=7500,
                        help="save model once every time this many iterations are completed")
    parser.add_argument("--local-dir", type=str, default="./ray_results",
                        help="path to save checkpoints")
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")

    # Parallelism
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-envs-per-worker", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)

    return parser.parse_args()

def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=vmas.Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # Scenario specific
        **config["scenario_config"],
    )
    return env

def init_ray(scenario_name: str, local_mode: bool = False):

    register_env(scenario_name, lambda config: env_creator(config))
    ModelCatalog.register_custom_model(
        "MyFullyConnectedNetwork", MyFullyConnectedNetwork
    )
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )



def main(args):
    config = {
                "device": "cpu",
                "num_envs": num_envs_per_worker,
                "scenario_name": scenario_name,
                "continuous_actions": True,
                "max_steps": 200,
                # Env specific
                # scenario config checked
                "scenario_config": {
                    "n_agents": 4,
                    "lidar_range": 0.55,
                    "agent_radius": 0.1,
                    "shared_rew": False,
                    "pos_shaping_factor": 1,
                    "final_reward": 0.1,
                    "comm_range": 0.55,
                    "agent_collision_penalty": -.5,
                    "observe_all_goals": False,
                },
    }
    ray.init(_temp_dir=str(PathUtils.scratch_dir / "ray"), local_mode=ON_MAC)
    MADDPGAgent = maddpg.MADDPGTrainer.with_updates(
        mixins=[CustomStdOut]
    )
    register_trainable("MADDPG", MADDPGAgent)
    print("Ray init!")
    register_env(scenario_name, env_creator(config))
    env = env_creator(config)
    def gen_policy(i):
        use_local_critic = [
            args.adv_policy == "ddpg" if i < args.num_adversaries else
            args.good_policy == "ddpg" for i in range(env.num_agents)
        ]
        return (
            None,
            env.observation_space_dict[i],
            env.action_space_dict[i],
            {
                "agent_id": i,
                "use_local_critic": use_local_critic[i],
                "obs_space_dict": env.observation_space_dict,
                "act_space_dict": env.action_space_dict,
            }
        )

    policies = {"policy_%d" %i: gen_policy(i) for i in range(len(env.observation_space_dict))}
    policy_ids = list(policies.keys())

    tune.run({
        "MADDPG_RLLib": {
            "run": "contrib/MADDPG",
            "env": "vmas",
            "stop": {
                "episodes_total": args.num_episodes,
            },
            "checkpoint_freq": args.checkpoint_freq,
            "local_dir": args.local_dir,
            "restore": args.restore,
            "config": {
                # === Log ===
                "log_level": "ERROR",

                # === Environment ===
                "env_config": {
                    "scenario_name": args.scenario,
                },
                "num_envs_per_worker": args.num_envs_per_worker,
                "horizon": args.max_episode_len,

                # === Policy Config ===
                # --- Model ---
                "good_policy": args.good_policy,
                "adv_policy": args.adv_policy,
                "actor_hiddens": [args.num_units] * 2,
                "actor_hidden_activation": "relu",
                "critic_hiddens": [args.num_units] * 2,
                "critic_hidden_activation": "relu",
                "n_step": args.n_step,
                "gamma": args.gamma,

                # --- Exploration ---
                "tau": 0.01,

                # --- Replay buffer ---
                "buffer_size": args.replay_buffer,

                # --- Optimization ---
                "actor_lr": args.lr,
                "critic_lr": args.lr,
                "learning_starts": args.train_batch_size * args.max_episode_len,
                "sample_batch_size": args.sample_batch_size,
                "train_batch_size": args.train_batch_size,
                "batch_mode": "truncate_episodes",

                # --- Parallelism ---
                "num_workers": args.num_workers,
                "num_gpus": args.num_gpus,
                "num_gpus_per_worker": 0,

                # === Multi-agent setting ===
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": ray.tune.function(
                        lambda i: policy_ids[i]
                    )
                },
            },
        },
    }, verbose=0)


if __name__ == '__main__':
    # from ray.rllib.algorithms.maddpg.maddpg import MADDPGConfig
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
    args = parse_args()
    main(args)