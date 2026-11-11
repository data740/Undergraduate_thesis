import math
import numpy as np
import random
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import copy
from concurrent.futures import ThreadPoolExecutor


class ReplayBuffer:
    def __init__(
        self, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.float32, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        # mix uniform sample
        prob = 0.5 * prob + 0.5 / sample_limit
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination, episode):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    @torch.no_grad()
    def dry_sample(self, batch_size, batch_length):
        """
        For testing only
        """
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        return indices

    def dry_append(self, episode):
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length


class VisualReplayBuffer:
    def __init__(
        self,
        obs_shape,
        action_dim,
        num_envs,
        max_length=int(1e6),
        warmup_length=50000,
        store_on_gpu=False,
        wm_priority_fraction=0.0,
        wm_priority_init_loss=10.0,
        wm_priority_temperature=1.0,
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        # prioritized world-model replay (Simulus curious replay): wm_loss_buffer[i] is the latest
        # world-model loss of the window starting at i, new steps start at wm_priority_init_loss
        self.wm_priority_fraction = wm_priority_fraction
        self.wm_priority_init_loss = wm_priority_init_loss
        self.wm_priority_temperature = wm_priority_temperature
        self.wm_loss_buffer = torch.full((max_length,), wm_priority_init_loss, dtype=torch.float32, device="cuda")

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        # mix uniform sample
        prob = 0.5 * prob + 0.5 / sample_limit
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    def _gather(self, indices, batch_length):
        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        # convert uint8 obs to float32
        obs = obs.to(torch.float32) / 255

        return obs, action, reward, termination

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        return self._gather(indices, batch_length)

    @torch.no_grad()
    def sample_wm(self, batch_size, batch_length):
        """
        Sampling for world-model training: a wm_priority_fraction share of the batch is drawn without
        replacement from softmax(wm_loss), the rest comes from the original sampler. Also returns the
        window start indices so the caller can write the new losses back with update_wm_priorities.
        """
        assert batch_size > 0, "batch_size must be greater than 0"

        sample_limit = self.length + 1 - batch_length
        num_prioritized = math.ceil(self.wm_priority_fraction * batch_size)
        num_prioritized = min(num_prioritized, batch_size, sample_limit)

        parts = []
        if num_prioritized > 0:
            prob = torch.softmax(self.wm_loss_buffer[:sample_limit] / self.wm_priority_temperature, dim=0)
            parts.append(torch.multinomial(prob, num_prioritized, replacement=False).cpu().numpy())
        if batch_size - num_prioritized > 0:
            parts.append(self.sample_indices(batch_size - num_prioritized, sample_limit))
        indices = np.concatenate(parts)
        # shuffle so that every ensemble member (batch chunk) sees a random mix of prioritized / regular samples
        indices = indices[np.random.permutation(len(indices))]

        return (*self._gather(indices, batch_length), indices)

    @torch.no_grad()
    def wm_priority_stats(self):
        stored = self.wm_loss_buffer[: self.length]
        prob = torch.softmax(stored / self.wm_priority_temperature, dim=0)
        return {
            "wm_loss_mean": stored.mean().item(),
            "wm_loss_max": stored.max().item(),
            "untouched_ratio": (stored == self.wm_priority_init_loss).float().mean().item(),
            "eff_size": (1.0 / (prob**2).sum()).item(),  # 1/sum(p^2): how many windows the priority path really spreads over
        }

    @torch.no_grad()
    def update_wm_priorities(self, indices, losses):
        indices = torch.as_tensor(indices, dtype=torch.long, device="cuda")
        self.wm_loss_buffer[indices] = losses.detach().to(torch.float32)

    def append(self, obs, action, reward, termination, episode):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        # convert float32 obs to uint8
        obs = obs * 255
        obs = obs.to(torch.uint8)

        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode
        self.wm_loss_buffer[self.last_pointer] = self.wm_priority_init_loss

        if len(self) < self.max_length:
            self.length += 1

    @torch.no_grad()
    def dry_sample(self, batch_size, batch_length):
        """
        For testing only
        """
        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)
        return indices

    def dry_append(self, episode):
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length


class ReplayBufferVectorPlusVisual:
    """
    Cutie object vector + visual observation
    """

    def __init__(
        self, state_shape, obs_shape, action_dim, num_envs, max_length=int(1e6), warmup_length=50000, store_on_gpu=False
    ) -> None:
        self.store_on_gpu = store_on_gpu
        self.action_dim = action_dim

        assert num_envs == 1, "only support num_envs=1"
        assert self.store_on_gpu, "This version forces to store on GPU"

        # if an index is visited outside of current length, the empty may lead to unexpected behavior like idx_dim >= 0 && idx_dim < index_size
        self.state_buffer = torch.empty((max_length, *state_shape), dtype=torch.float32, device="cuda")
        self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device="cuda")
        self.action_buffer = torch.empty((max_length, action_dim), dtype=torch.int32, device="cuda")
        self.reward_buffer = torch.empty((max_length,), dtype=torch.float32, device="cuda")
        self.termination_buffer = torch.empty((max_length,), dtype=torch.int32, device="cuda")
        self.episode_buffer = torch.zeros((max_length,)).cuda()

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def ready(self):
        return self.length > self.warmup_length

    def sample_indices(self, batch_size, sample_limit):
        # power decay sample >>>
        logits = self.episode_buffer[:sample_limit] - torch.max(self.episode_buffer[:sample_limit])
        prob = torch.exp(logits * torch.log(torch.tensor(1.25)))
        prob = prob / torch.sum(prob)
        prob = 0.5 * prob + 0.5 / sample_limit  # mix uniform
        # <<< power decay sample

        indices = torch.multinomial(prob, batch_size, replacement=True)

        return indices.cpu().numpy()  # otherwise the "for idx in indices" later will be very slow

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        assert batch_size > 0, "batch_size must be greater than 0"

        indices = self.sample_indices(batch_size, self.length + 1 - batch_length)

        state = torch.stack([self.state_buffer[idx : idx + batch_length] for idx in indices])
        obs = torch.stack([self.obs_buffer[idx : idx + batch_length] for idx in indices])
        action = torch.stack([self.action_buffer[idx : idx + batch_length] for idx in indices])
        reward = torch.stack([self.reward_buffer[idx : idx + batch_length] for idx in indices])
        termination = torch.stack([self.termination_buffer[idx : idx + batch_length] for idx in indices])

        # convert uint8 obs to float32
        obs = obs.to(torch.float32) / 255

        return state, obs, action, reward, termination

    def append(self, state_obs, action, reward, termination, episode):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % self.max_length

        state, obs = state_obs  # tuple
        # convert float32 obs to uint8
        obs = obs * 255
        obs = obs.to(torch.uint8)

        self.state_buffer[self.last_pointer] = state
        self.obs_buffer[self.last_pointer] = obs
        self.action_buffer[self.last_pointer] = torch.from_numpy(action).cuda()
        self.reward_buffer[self.last_pointer] = reward
        self.termination_buffer[self.last_pointer] = termination
        self.episode_buffer[self.last_pointer] = episode

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length
