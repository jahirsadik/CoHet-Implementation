import torch
import numpy as np
import torch.nn.functional as F

from typing import Dict
from torch import nn
from torch.optim import Adam
from tianshou.data import to_torch, Batch

class WorldModel(nn.Module):
    def __init__(self, num_agent, layer_num, input_dim, output_dim, hidden_units=128, device='cpu', wm_noise_level=0.0):
        super().__init__()
        self.device = device
        # plus one for the action
        self.model = [
            nn.Linear(input_dim, hidden_units),  # change here diggu
            nn.ReLU()]
        # this is code for the dynamics model MLP
        # default is input -> hidden layer -> hidden layer -> output layer
        for _ in range(layer_num - 1):
            self.model += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        self.model += [nn.Linear(hidden_units, np.prod(output_dim))]
        self.num_agent = num_agent
        self.model = nn.Sequential(*self.model)
        self.optim = Adam(self.model.parameters(), lr=1e-3)
        self.wm_noise_level = wm_noise_level

    # forward pass
    # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    def forward(self, s, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        # view is a guaranteed no-copy reshape
        # -1 means infer the other dimension
        # https://stackoverflow.com/questions/50792316/what-does-1-mean-in-pytorch-view
        s = s.view(batch, -1)
        logits = self.model(s)  # "In context of deep learning the logits layer means the layer that feeds in to softmax (or other such normalization)"
        # torch.normal samples from a normal dist using a mean and stddev
        # mean here is 0 and the wm_noise_level arg itself is the stddev
        if self.wm_noise_level != 0.0:
            logits += torch.normal(torch.zeros(logits.size()), self.wm_noise_level).to(logits.device)
        return logits

    # this function could not be more obvious
    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        total_loss = np.zeros(self.num_agent)
        for i in range(self.num_agent):
            # concatenating each agent's observation and action
            inputs = np.concatenate((batch.obs[:, i], np.expand_dims(batch.act[:, i], axis=-1)), axis=1)
            next_obs_pred = self.model(to_torch(inputs, device=self.device, dtype=torch.float))
            true_next_obs = to_torch(batch.obs_next[:, i], device=self.device, dtype=torch.float)
            loss = F.mse_loss(next_obs_pred, true_next_obs)
            # zero out the gradient before backprop
            # the batch will affect the policy and it will already affect subsequent batches
            # no need to accumulate the gradient across multiple batches if not RNN
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            self.optim.zero_grad()
            loss.backward()  # backprop
            self.optim.step()  # update the params https://pytorch.org/docs/stable/optim.html#taking-an-optimization-step

            # mse_loss returns a tensor with a single 1D value
            # this gets that value as a number
            total_loss[i] = loss.item()
        # pass in dummy state and action
        output = {}
        for i in range(self.num_agent):
            output[f'models/actor_{i}'] = total_loss[i]
        output[f'loss/world_model'] = total_loss.sum()
        return output



