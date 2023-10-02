"""
PyTorch's policy class used for PPO.
"""
#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import logging
from abc import ABC
from typing import Dict
from typing import List, Optional, Union, Tuple, Iterable
from typing import Type
import torch.nn as nn
import gym
from torch.optim import Adam
from gym.spaces import Discrete, Box
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.execution import synchronous_parallel_sample
from ray.rllib.execution.common import (
    _check_sample_batch_type,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.models import ModelV2, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
    SampleBatch,
    DEFAULT_POLICY_ID,
    concat_samples,
)
from ray.rllib.policy.torch_mixins import (
    LearningRateSchedule,
    KLCoeffMixin,
    EntropyCoeffSchedule,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2, torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
)
from ray.rllib.utils.torch_utils import (
    warn_if_infinite_kl_divergence,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import AgentID, TensorType, ResultDict, TrainerConfigDict
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from rllib_differentiable_comms.utils import to_torch

from models.dynamics import WorldModel

nn_functional = nn.functional
torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class InvalidActionSpace(Exception):
    """Raised when the action space is invalid"""

    pass


def standardized(array: np.ndarray):
    """Normalize the values in an array.

    Args:
        array (np.ndarray): Array of values to normalize.

    Returns:
        array with zero mean and unit standard deviation.
    """
    return (array - array.mean(axis=0, keepdims=True)) / array.std(
        axis=0, keepdims=True
    ).clip(min=1e-4)


def standardize_fields(samples: SampleBatchType, fields: List[str]) -> SampleBatchType:
    """Standardize fields of the given SampleBatch"""
    _check_sample_batch_type(samples)
    wrapped = False

    if isinstance(samples, SampleBatch):
        samples = samples.as_multi_agent()
        wrapped = True

    for policy_id in samples.policy_batches:
        batch = samples.policy_batches[policy_id]
        for field in fields:
            if field in batch:
                batch[field] = standardized(batch[field])

    if wrapped:
        samples = samples.policy_batches[DEFAULT_POLICY_ID]

    return samples


def compute_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
    other_agent_batches: Optional[Dict[AgentID, SampleBatch]] = None,
    episode: Optional[Episode] = None,
) -> SampleBatch:
    """Adds GAE (generalized advantage estimations) to a trajectory.
    The trajectory contains only data from one episode and from one agent.
    - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
    contain a truncated (at-the-end) episode, in case the
    `config.rollout_fragment_length` was reached by the sampler.
    - If `config.batch_mode=complete_episodes`, sample_batch will contain
    exactly one episode (no matter how long).
    New columns can be added to sample_batch and existing ones may be altered.
    Args:
        policy (Policy): The Policy used to generate the trajectory
            (`sample_batch`)
        sample_batch (SampleBatch): The SampleBatch to postprocess.
        other_agent_batches (Optional[Dict[PolicyID, SampleBatch]]): Optional
            dict of AgentIDs mapping to other agents' trajectory data (from the
            same episode). NOTE: The other agents use the same policy.
        episode (Optional[MultiAgentEpisode]): Optional multi-agent episode
            object in which the agents operated.
    Returns:
        SampleBatch: The postprocessed, modified SampleBatch (or a new one).
    """
    n_agents = len(policy.action_space)

    if sample_batch[SampleBatch.INFOS].dtype == "float32":
        # The trajectory view API will pass populate the info dict with a np.zeros((ROLLOUT_SIZE,))
        # array in the first call, in that case the dtype will be float32, and we
        # ignore it by assignining it to all agents
        samplebatch_infos_rewards = concat_samples(
            [
                SampleBatch(
                    {
                        str(i): sample_batch[SampleBatch.REWARDS].copy()
                        for i in range(n_agents)
                    }
                )
            ]
        )
    else:
        #  For regular calls, we extract the rewards from the info
        #  dict into the samplebatch_infos_rewards dict, which now holds the rewards
        #  for all agents as dict.
        # sample_batch[SampleBatch.INFOS] = list of len ROLLOUT_SIZE of which every element is
        # {'rewards': {0: -0.077463925, 1: -0.0029145998, 2: -0.08233316}} if there are 3 agents
        samplebatch_infos_rewards = concat_samples(
            [
                SampleBatch({str(k): [np.float32(v)] for k, v in s["rewards"].items()})
                for s in sample_batch[SampleBatch.INFOS]
                # s = {'rewards': {0: -0.077463925, 1: -0.0029145998, 2: -0.08233316}} if there are 3 agents
            ]
        )

        # samplebatch_infos_rewards = SampleBatch(ROLLOUT_SIZE: ['0', '1', '2']) if there are 3 agents
        # (i.e. it has ROLLOUT_SIZE entries with keys '0','1','2')

    if not isinstance(policy.action_space, gym.spaces.tuple.Tuple):
        raise InvalidActionSpace("Expect tuple action space")

    keys_to_overwirte = [
        SampleBatch.REWARDS,
        SampleBatch.VF_PREDS,
        Postprocessing.ADVANTAGES,
        Postprocessing.VALUE_TARGETS,
    ]

    original_batch = sample_batch.copy()

    # We prepare the sample batch to contain the agent batches
    for k in keys_to_overwirte:
        sample_batch[k] = np.zeros((len(original_batch), n_agents), dtype=np.float32)

    if original_batch[SampleBatch.DONES][-1]:
        all_values = None
    else:
        input_dict = original_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last"
        )
        all_values = policy._value(**input_dict)

    # Create the sample_batch for each agent
    for key in samplebatch_infos_rewards.keys():
        agent_index = int(key)
        sample_batch_agent = original_batch.copy()
        sample_batch_agent[SampleBatch.REWARDS] = samplebatch_infos_rewards[key]
        sample_batch_agent[SampleBatch.VF_PREDS] = original_batch[SampleBatch.VF_PREDS][
            :, agent_index
        ]

        if all_values is None:
            last_r = 0.0
        # Trajectory has been truncated -> last r=VF estimate of last obs.
        else:
            last_r = (
                all_values[agent_index].item()
                if policy.config["use_gae"]
                else all_values
            )

        # Adds the policy logits, VF preds, and advantages to the batch,
        # using GAE ("generalized advantage estimation") or not.
        sample_batch_agent = compute_advantages(
            sample_batch_agent,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=policy.config["use_gae"],
            use_critic=policy.config.get("use_critic", True),
        )

        for k in keys_to_overwirte:
            sample_batch[k][:, agent_index] = sample_batch_agent[k]

    return sample_batch


def ppo_surrogate_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[ActionDistribution],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]): The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    # print(f"ppo_surrogate_loss model: {model}")
    # print(f"SAMPLE BATCH INFOS: {train_batch[SampleBatch.INFOS]}")
    # print(f"batch agent index shape: {train_batch.columns([SampleBatch.AGENT_INDEX])}")
    # print(f"batch actions shape: {train_batch.columns([SampleBatch.ACTIONS])[0].shape}")
    from datetime import datetime
    # print(f"Calling model forward() function from ppo_surrogate_loss with batch size {len(train_batch)} at {datetime.now().strftime('%H:%M:%S')}")
    # print(f"train_batch size: {len(train_batch)}")
    # print(f"Train batch example: {train_batch[SampleBatch.OBS][:1]}")
    logits, state = model(train_batch)
    # logits has shape (BATCH, num_agents * num_outputs_per_agent)
    curr_action_dist = dist_class(logits, model)
    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major(),
        )
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    # train_batch[SampleBatch.ACTIONS] has shape (BATCH, num_agents * action_size)
    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
        - train_batch[SampleBatch.ACTION_LOGP]
    )

    use_kl = policy.config["kl_coeff"] > 0.0
    if use_kl:
        action_kl = prev_action_dist.kl(curr_action_dist)
    else:
        action_kl = torch.tensor(0.0, device=logp_ratio.device)

    curr_entropies = curr_action_dist.entropy()

    # Compute a value function loss.
    if policy.config["use_critic"]:
        # print(f"Calling model value_function()")
        value_fn_out = model.value_function()
    else:
        value_fn_out = torch.tensor(0.0, device=logp_ratio.device)

    loss_data = []
    n_agents = len(policy.action_space)
        
    for i in range(n_agents):
        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES][..., i] * logp_ratio[..., i],
            train_batch[Postprocessing.ADVANTAGES][..., i]
            * torch.clamp(
                logp_ratio[..., i],
                1 - policy.config["clip_param"],
                1 + policy.config["clip_param"],
            ),
        )

        # Compute a value function loss.
        if policy.config["use_critic"]:
            agent_value_fn_out = value_fn_out[..., i]
            vf_loss = torch.pow(
                agent_value_fn_out - train_batch[Postprocessing.VALUE_TARGETS][..., i],
                2.0,
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, policy.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            agent_value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = (
            -surrogate_loss
            + policy.config["vf_loss_coeff"] * vf_loss_clipped
            - policy.entropy_coeff * curr_entropies[..., i]
        )

        # Add mean_kl_loss if necessary.
        if use_kl:
            mean_kl_loss = reduce_mean_valid(action_kl[..., i])
            total_loss += policy.kl_coeff * mean_kl_loss
            # TODO smorad: should we do anything besides warn? Could discard KL term
            # for this update
            warn_if_infinite_kl_divergence(policy, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        total_loss = reduce_mean_valid(total_loss)
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)
        mean_entropy = reduce_mean_valid(curr_entropies[..., i])
        vf_explained_var = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS][..., i], agent_value_fn_out
        )

        # Store stats in policy for stats_fn.
        loss_data.append(
            {
                "total_loss": total_loss,
                "mean_policy_loss": mean_policy_loss,
                "mean_vf_loss": mean_vf_loss,
                "mean_entropy": mean_entropy,
                "mean_kl": mean_kl_loss,
                "vf_explained_var": vf_explained_var,
            }
        )

    aggregation = torch.mean
    total_loss = aggregation(torch.stack([ld["total_loss"] for ld in loss_data]))
    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = aggregation(
        torch.stack([ld["mean_policy_loss"] for ld in loss_data])
    )
    model.tower_stats["mean_vf_loss"] = aggregation(
        torch.stack([ld["mean_vf_loss"] for ld in loss_data])
    )
    model.tower_stats["vf_explained_var"] = aggregation(
        torch.stack([ld["vf_explained_var"] for ld in loss_data])
    )
    model.tower_stats["mean_entropy"] = aggregation(
        torch.stack([ld["mean_entropy"] for ld in loss_data])
    )
    model.tower_stats["mean_kl_loss"] = aggregation(
        torch.stack([ld["mean_kl"] for ld in loss_data])
    )
    
    return total_loss



class MultiAgentValueNetworkMixin:
    """Assigns the `_value()` method to a TorchPolicy.

    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    """

    def __init__(self, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last on e in trajectory)
            # input_dict.
            def value(**input_dict):
                """This is exactly the as in PPOTorchPolicy,
                but that one calls .item() on self.model.value_function()[0],
                which will not work for us since our value function returns
                multiple values. Instead, we call .item() in
                compute_gae_for_sample_batch above.
                """
                print(f"MultiAgentValueNetworkMixin value()")
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0]
                # When not doing GAE, we do not require the value function's output.

        # When not doing GAE, we do not require the value function's output.
        else:
            print(f"MultiAgentValueNetworkMixin value()")
            def value(*args, **kwargs):
                return 0.0

        self._value = value


class MultiPPOTorchPolicy(PPOTorchPolicy, MultiAgentValueNetworkMixin):
    def __init__(self, observation_space, action_space, config):
        print(f"Initializing MultiPPOTorchPolicy")
        config = dict(ray.rllib.algorithms.ppo.ppo.PPOConfig().to_dict(), **config)
        # TODO: Move into Policy API, if needed at all here. Why not move this into
        #  `PPOConfig`?.
        self.alignment_type = config["model"]["custom_model_config"].get("alignment_type", None)
        self.use_gppo_dyn = config["model"]["custom_model_config"].get("use_gppo_dyn", False)
        if self.alignment_type is not None:
            obs_dim = observation_space.shape[0] // len(action_space) # obs space / no of agents
            act_dim = action_space[0].shape[0]  # action space shape[0] = no of agents
            self.dyn_models = [WorldModel(num_agent = len(action_space), 
                                            layer_num= config["model"]["custom_model_config"].get("dyn_model_layer_num", 2), 
                                            input_dim = (128 + act_dim) if self.use_gppo_dyn else (obs_dim + act_dim), # TODO @deeparghya check
                                            output_dim = 128 if self.use_gppo_dyn else obs_dim, # TODO @deeparghya what instead of 128?
                                            hidden_units = config["model"]["custom_model_config"].get("dyn_model_hidden_units", 128), 
                                            device=config["env_config"]["device"]).to(config["env_config"]["device"])
                            for _ in range(len(action_space))]
            print(f"dyn_model_hidden_layer: {config['model']['custom_model_config'].get('dyn_model_hidden_units', 128)}")
            # adam optimiser to dynamics model as well
            # use the actor learning rate
            self.dyn_model_optims = [Adam(dyn_model.parameters()) for dyn_model in self.dyn_models]

        validate_config(config)
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )
        # Only difference from ray code
        MultiAgentValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)
        self.grad_gnorm = 0

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return ppo_surrogate_loss(self, model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        from datetime import datetime
        print(f"postprocess_trajectory called at {datetime.now().strftime('%H:%M:%S')}")
        print(f"train_batch size: {sample_batch[SampleBatch.OBS].shape}")
        # print(f"Train batch example: {sample_batch[SampleBatch.OBS][:1]}")

        with torch.no_grad():
            sample_batch = compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)
        # WM stats here
        dyn_losses_t = self.model.tower_stats.get('dyn_models_loss', None)
        if torch.is_tensor(dyn_losses_t):
            dyn_losses = dyn_losses_t.tolist()
            for cur_agent_idx, dyn_loss in enumerate(dyn_losses):
                episode.custom_metrics[f'agent {cur_agent_idx}/dyn_model_loss'] = dyn_loss

        batch_size = len(sample_batch)
        n_agents = len(self.action_space)   # total no of agents
        intr_rew_t = torch.zeros((batch_size, n_agents)) # Initialize empty torch first, to be changed later
        cur_obs_batch = sample_batch[SampleBatch.OBS].reshape((batch_size, n_agents, -1))
        default_beta = 1 / 128 if self.use_gppo_dyn else 1 / cur_obs_batch.shape[2]
        intr_rew_beta =  self.config["model"]["custom_model_config"].get("intr_rew_beta", default_beta)
        intr_beta_type = self.config["model"]["custom_model_config"].get("intr_beta_type", "normal")
        
        next_obs_batch = sample_batch[SampleBatch.NEXT_OBS].reshape((batch_size, n_agents, -1))
        act_batch = sample_batch[SampleBatch.ACTIONS].reshape((batch_size, n_agents, -1))
        cur_obs_act_batch = to_torch(np.concatenate((cur_obs_batch, act_batch), axis=2))

        gppo_outputs_cur_obs = None
        gppo_outputs_next_obs = None
        cur_h_act_all_agents = None
        if self.alignment_type is not None and episode is not None and intr_rew_beta > 0 and self.use_gppo_dyn is True:
            gppo_outputs_cur_obs = self.model.get_gppo_embedding(to_torch(cur_obs_batch), self.config["env_config"]["device"])
            print(f"GPPO outputs in postprocess_trajectory(): {gppo_outputs_cur_obs[:2]}")
            gppo_outputs_next_obs = self.model.get_gppo_embedding(to_torch(next_obs_batch), self.config["env_config"]["device"])
            print(f"GPPO outputs in postprocess_trajectory(): {gppo_outputs_next_obs[:2]}")
            cur_h_act_all_agents = np.concatenate((gppo_outputs_cur_obs, act_batch), axis=2)

        train_start_index = 0
        if "goal_rel_start" in self.config["model"]["custom_model_config"]:
            assert "goal_rel_dim" in self.config["model"]["custom_model_config"]
        train_end_index = self.config["model"]["custom_model_config"].get("goal_rel_start", cur_obs_batch.shape[2]) + self.config["model"]["custom_model_config"].get("goal_rel_dim", 0)
        
        pos_start_index = self.config["model"]["custom_model_config"].get("pos_start", 0)
        pos_end_index = pos_start_index + self.config["model"]["custom_model_config"].get("pos_dim", 2)
        vel_start_index = self.config["model"]["custom_model_config"].get("vel_start", 2)
        vel_end_index = vel_start_index + self.config["model"]["custom_model_config"].get("vel_dim", 2)
        
        # Training the dynamics model with batch data
        if self.alignment_type is not None and episode is not None and intr_rew_beta > 0:
            # if not self.use_gppo_dyn:
            #     obs_act_all_agents = np.concatenate((
            #     sample_batch[SampleBatch.OBS].reshape((batch_size, n_agents, -1)),
            #     sample_batch[SampleBatch.ACTIONS].reshape((batch_size, n_agents, -1))), axis=2)
            
            inputs = to_torch(cur_h_act_all_agents) if self.use_gppo_dyn else to_torch(cur_obs_act_batch, dtype=torch.float) 
            # print(f"cur_obs0: {cur_obs_batch.shape},\n act_batch: {act_batch.shape},\n inputs: {inputs.shape}")
            # print(f"cur_obs0: {cur_obs_batch[0]}\nact_batch0: {act_batch[0]}\ninputs0: {inputs[0]}")
            pred_next_obs_or_h_all = []
            for i in range(n_agents):
                pred_next_obs_or_h_i = self.dyn_models[i](inputs[:, i, :])
                # print(f"Agent{i} Predicted next observation0: {pred_next_obs_or_h_i[0:10]}")
                # print(f"Agent{i} True next observation0: {to_torch(next_obs_batch[0:10, i, :])}")
                # print(f'pred_next_obs_or_h_i shape: {pred_next_obs_or_h_i.shape}')
                pred_next_obs_or_h_all.append(pred_next_obs_or_h_i)

            assert n_agents == len(pred_next_obs_or_h_all)
            # dyn_models_loss = []
            for i, pred_next_obs_or_h_i in enumerate(pred_next_obs_or_h_all):
                # print(f'pred_next_obs_or_h_i: {pred_next_obs_or_h_i[:, train_start_index:train_end_index].shape}')
                # print(f'next_obs_batch: {next_obs_batch[:, i, train_start_index:train_end_index].shape}')
                if self.use_gppo_dyn:
                    agent_i_dyn_loss = torch.nn.functional.mse_loss(
                        pred_next_obs_or_h_i,
                        gppo_outputs_next_obs[:, i, :])
                else:
                    agent_i_dyn_loss = torch.nn.functional.mse_loss(pred_next_obs_or_h_i[:, train_start_index:train_end_index], 
                                                                    to_torch(next_obs_batch[:, i, train_start_index:train_end_index]))
                self.dyn_models[i].zero_grad()
                agent_i_dyn_loss.backward()
                self.dyn_model_optims[i].step()
                print(f"Agent{i} dyn model loss: {agent_i_dyn_loss}")
                # dyn_models_loss.append(agent_i_dyn_loss)
                if episode is not None:
                    episode.custom_metrics[f"agent {i}/dyn_models_loss"] = agent_i_dyn_loss.item()
            
        comm_radius = self.config["model"]["custom_model_config"].get("comm_radius", float('inf'))

        if self.alignment_type == "team" and episode is not None and intr_rew_beta > 0:
            intr_rew_batch = []
            # calculate which neighbors are in the neighborhood at time t and t+1
            pos_batch = cur_obs_batch[:, :, pos_start_index:pos_end_index]
            next_pos_batch = next_obs_batch[:, :, pos_start_index:pos_end_index]
            dist_t = np.sqrt(((pos_batch[:, :, None] - pos_batch[:, None]) ** 2).sum(axis=3))
            dist_t1 = np.sqrt(((next_pos_batch[:, :, None] - next_pos_batch[:, None]) ** 2).sum(axis=3))
            dist_t_edges = [np.argwhere(inner_arr < comm_radius) for inner_arr in dist_t]
            dist_t1_edges = [np.argwhere(inner_arr < comm_radius) for inner_arr in dist_t1]
            neighbors_t_batch = []
            neighbors_t1_batch = []
            for sample_t, sample_t1 in zip(dist_t_edges, dist_t1_edges):
                _, counts_t = np.unique(sample_t[:, 0], return_counts=True)
                _, counts_t1 = np.unique(sample_t1[:, 0], return_counts=True)
                neighbors_t = np.split(sample_t[:, 1], np.cumsum(counts_t)[:-1])
                neighbors_t1 = np.split(sample_t1[:, 1], np.cumsum(counts_t1)[:-1])
                neighbors_t = [list(inner_arr) for inner_arr in neighbors_t]
                neighbors_t1 = [list(inner_arr) for inner_arr in neighbors_t1]
                neighbors_t_batch.append(neighbors_t)
                neighbors_t1_batch.append(neighbors_t1)
            # calculate common neighbors between time t and t+1
            common_neighbors_batch = []
            for neighbors_t, neighbors_t1 in zip(neighbors_t_batch, neighbors_t1_batch):
                common_neighbors = []
                for agent_t, agent_t1 in zip(neighbors_t, neighbors_t1):
                    common_neighbors.append(list(set(agent_t) & set(agent_t1)))
                common_neighbors_batch.append(common_neighbors)

            intr_rew_weighting = self.config["model"]["custom_model_config"].get('intr_rew_weighting', 'average')
            print(f'selected weighting: {intr_rew_weighting}')
            print(f"Common neighbors batch: {common_neighbors_batch[:10]}")
            for batch_idx, batch_data in enumerate(common_neighbors_batch):
                intr_rew_agent = []
                for cur_agent_idx, neighbors in enumerate(batch_data):
                    l2_loss = 0
                    total_dist_inv = 0
                    for neighbor in neighbors:
                        if cur_agent_idx != neighbor:
                            neighbor_pos = cur_obs_batch[batch_idx, neighbor, pos_start_index:pos_end_index]
                            agent_pos = cur_obs_batch[batch_idx, cur_agent_idx, pos_start_index:pos_end_index]
                            dist_inv = 1 / np.sqrt(((agent_pos - neighbor_pos) ** 2).sum())
                            total_dist_inv += dist_inv
                            # TODO: Find a way to send data in batches
                            if self.use_gppo_dyn:
                                neighbor_pred = self.dyn_models[neighbor](cur_h_act_all_agents[batch_idx, cur_agent_idx, :].view(1, -1))
                                true_next_obs_or_h = to_torch(gppo_outputs_next_obs.reshape(1, -1))
                            else:
                                neighbor_pred = self.dyn_models[neighbor](cur_obs_act_batch[batch_idx, cur_agent_idx, :].view(1, -1))
                                true_next_obs_or_h = to_torch(next_obs_batch[batch_idx, cur_agent_idx, :].reshape(1, -1))
                            
                            l2_loss += (torch.nn.functional.mse_loss(neighbor_pred, true_next_obs_or_h) * dist_inv if intr_rew_weighting == 'distance' else 1)
                    # log this l2 loss
                    if intr_rew_weighting == 'distance' and total_dist_inv > 0:
                        intr_rew_agent.append(-l2_loss / total_dist_inv)
                    else:
                        intr_rew_agent.append(-l2_loss / len(neighbors))
                intr_rew_batch.append(intr_rew_agent)
            intr_rew_t = torch.FloatTensor(intr_rew_batch)
        
        elif self.alignment_type == "self" and episode is not None and intr_rew_beta > 0:
            for cur_agent_idx in range(n_agents):
                if self.use_gppo_dyn:
                    self_pred = self.dyn_models[cur_agent_idx](cur_h_act_all_agents[:, cur_agent_idx, :])
                    true_next_obs_or_h = gppo_outputs_next_obs[:, cur_agent_idx, :]
                else:
                    self_pred = self.dyn_models[cur_agent_idx](cur_obs_act_batch[:, cur_agent_idx, :])
                    true_next_obs_or_h = to_torch(next_obs_batch[:, cur_agent_idx, :])
                l2_loss = torch.mean(torch.nn.functional.mse_loss(self_pred, true_next_obs_or_h, reduction='none'), dim=1)
                intr_rew_t[:, cur_agent_idx] = -l2_loss # not dividing by num of agents here
        


        if self.alignment_type is not None and episode is not None and intr_rew_beta > 0 and intr_beta_type == "percent":
            print(f'intrinsic reward shape: {intr_rew_t.shape}')
            print(f"intr_rew_beta: {intr_rew_beta}")
            max_extr = to_torch(np.max(np.absolute(sample_batch[SampleBatch.REWARDS]), 0))
            max_intr = torch.max(torch.abs(intr_rew_t), 0).values
            print(f"Episode Extrinsic reward shape: {sample_batch[SampleBatch.REWARDS].shape}")
            print(f"Episode Extrinsic rewards: {sample_batch[SampleBatch.REWARDS][0:20]}")
            print(f"Episode max extrinsic reward: {max_extr}")
            print(f"Episode max intrinsic reward: {max_intr}")
            print(f"Intrinsic reward before scaling: {intr_rew_t[0:20]}")
            frac = 1
            print(f"max_intr: {max_intr}, {type(max_intr)}, {max_intr.shape}")
            print(f"max_extr: {max_intr}, {type(max_extr)}, {max_extr.shape}")
            for i, val in enumerate(max_intr):
                print(f"max_intr val: {val} for agent{i}")
                if val > 0:
                    max_extr_beta_percent_val = (max_extr[i] / 100) * intr_rew_beta    
                    frac = max_extr_beta_percent_val / abs(val)
                    print(f"Intrinsic reward scaling frac: {frac}")
                    intr_rew_t[:, i] *= frac
            print(f"Intrinsic reward after scaling: {intr_rew_t[0:20]}")
        elif self.alignment_type is not None and episode is not None and intr_rew_beta > 0 and intr_beta_type == "normal":
            print(f'intrinsic reward shape: {intr_rew_t.shape}')
            print(f"intr_rew_beta: {intr_rew_beta}")
            print(f"Episode Extrinsic reward shape: {sample_batch[SampleBatch.REWARDS].shape}")
            print(f"Episode Extrinsic rewards: {sample_batch[SampleBatch.REWARDS][0:20]}")
            print(f"Intrinsic reward before scaling: {intr_rew_t[0:20]}")
            intr_rew_t *= intr_rew_beta
            print(f"Intrinsic reward after scaling: {intr_rew_t[0:20]}")
            
        # reward logging
        if episode is not None:
            for cur_agent_idx in range(n_agents):
                if self.alignment_type is not None and episode is not None and intr_rew_beta > 0:
                    agent_intr_mean = torch.mean(intr_rew_t[:, cur_agent_idx])
                    episode.custom_metrics[f'agent {cur_agent_idx}/intr_rew'] = agent_intr_mean.item()
                agent_extr_mean = torch.mean(to_torch(sample_batch[SampleBatch.REWARDS][:, cur_agent_idx]))
                episode.custom_metrics[f'agent {cur_agent_idx}/extr_rew'] = agent_extr_mean.item()

        # enter rewards for each agent
        print(f'int rew: \n{intr_rew_t[0]}')
        if self.alignment_type is not None and episode is not None and intr_rew_beta > 0:
            print(f'sample batch rew before adding int rew: \n{sample_batch[SampleBatch.REWARDS][0]}')
            sample_batch[SampleBatch.REWARDS] += intr_rew_t.detach().cpu().numpy()
            print(f'sample batch rew after adding int rew: \n{sample_batch[SampleBatch.REWARDS][0]}')

        return sample_batch


    @override(PPOTorchPolicy)
    def extra_grad_process(self, local_optimizer, loss):
        grad_gnorm = apply_grad_clipping(self, local_optimizer, loss)
        if "grad_gnorm" in grad_gnorm:
            self.grad_gnorm = grad_gnorm["grad_gnorm"]
        return grad_gnorm

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
                "grad_gnorm": self.grad_gnorm,
            }
        )   


class MultiPPOTrainer(PPOTrainer, ABC):
    @override(PPOTrainer)
    def get_default_policy_class(self, config):
        return MultiPPOTorchPolicy

    @override(PPOTrainer)
    def training_step(self) -> ResultDict:
        print("Inside MultiPPOTrainer training_step()")
        # Collect SampleBatches from sample workers until we have a full batch.
        if self._by_agent_steps:
            assert False
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config["train_batch_size"]
            )
        else:
            print(f"Calling synchronous_parallel_sample from MultiPPOTrainer")
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
            )
        train_batch = train_batch.as_multi_agent()
        print(f"Train Batch size: {len(train_batch)}")
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
        
        # Standardize advantage
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config["simple_optimizer"]:
            assert False
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)
                
        # For each policy: update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config["vf_loss_coeff"] * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if scaled_vf_loss > 100:
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
        
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if mean_reward > self.config["vf_clip_param"]:
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results
