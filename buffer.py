"""
heavily borrowed from 
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
"""


import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, NamedTuple

import numpy as np 
import torch 
from gymnasium import spaces
from segment_tree import SumSegmentTree, MinSegmentTree

try:
    import psutil
except ImportError:
    psutil = None


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    weights: torch.Tensor
    batch_indices: np.ndarray


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return len(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
    

def get_obs_shape(
    observation_space: spaces.Space,
) -> Union[tuple[int, ...], dict[str, tuple[int, ...]]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (len(observation_space.nvec),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")
    

def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos
    
    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 
        """
        shape = arr.shape
        if len(shape) < 3:
            shape (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    @abstractmethod
    def sample(self, batch_size: int, env: Optional[Any] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
    :param use_uint8: Store observations using uint8 to save memory

    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        use_uint8: bool = False,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.buffer_size = max(buffer_size // n_envs, 1)
        self.optimize_memory_usage = optimize_memory_usage
        self.use_uint8 = use_uint8

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        obs_dtype = np.uint8 if self.use_uint8 else np.float32
        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=obs_dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape), dtype=obs_dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=np.int64 if isinstance(action_space, spaces.Discrete) else np.float32
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if handle_timeout_termination:
            self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        if self.use_uint8:
            self.observations[self.pos] = (np.array(obs) * 255).astype(np.uint8)
        else:
            self.observations[self.pos] = np.array(obs).copy()

        if self.use_uint8:
            next_obs_stored = (np.array(next_obs) * 255).astype(np.uint8)
        else:
            next_obs_stored = np.array(next_obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = next_obs_stored.copy()
        else:
            self.next_observations[self.pos] = next_obs_stored.copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if hasattr(self, 'timeouts') and infos is not None:
            timeouts = np.array([info.get("TimeLimit.truncated", False) for info in infos])
            self.timeouts[self.pos] = timeouts.copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[Any] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return self._sample_proportional(batch_size, env)

        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)
    
    def _sample_proportional(self, batch_size: int, env: Optional[Any] = None) -> ReplayBufferSamples:
        batch_inds = np.random.randint(0, self.size(), size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[Any] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._get_next_obs_optimized(batch_inds, env_indices)
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        obs = self.observations[batch_inds, env_indices, :]
        if self.use_uint8:
            obs = obs.astype(np.float32) / 255.0
            next_obs = next_obs.astype(np.float32) / 255.0

        if isinstance(self.action_space, spaces.Discrete):
            actions = self.actions[batch_inds, env_indices].reshape(-1, 1)
        else:
            actions = self.actions[batch_inds, env_indices, :]

        if hasattr(self, 'timeouts'):
            dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)
        else:
            dones = self.dones[batch_inds, env_indices].reshape(-1, 1)

        rewards = self.rewards[batch_inds, env_indices].reshape(-1, 1)

        data = (obs, actions, next_obs, dones, rewards)
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_next_obs_optimized(self, batch_inds: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """
        Get next Observation when using memory optimized variants
        """
        next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]

        for i, (batch_idx, env_idx) in enumerate(zip(batch_inds, env_indices)):
            if self.dones[batch_idx, env_idx]:
                next_obs[i] = self.observations[batch_idx, env_idx]
        return next_obs
    
    def __len__(self) -> int:
        return self.size()

    def update(self, s, a, r, s_prime, terminated):
        """
        Legacy Method for compatibility. Use add() instead
        Store a transition in the buffer
        """
        if isinstance(s, np.ndarray) and s.ndim == len(self.obs_shape):
            obs = s[np.newaxis]
            next_obs = s_prime[np.newaxis]
            actions = np.array([a])
            rewards = np.array([r])
            dones = np.array([terminated])
            infos = [{}]
        else:
            obs = np.array(s)
            next_obs = np.array(s_prime)
            actions = np.array(a)
            rewards = np.array(r)
            dones = np.array(terminated)
            infos = None

        self.add(obs, next_obs, actions, rewards, dones, infos)

    def get_memory_usage(self) -> dict:
        obs_memory = self.observations.nbytes
        next_obs_memory = self.next_observations.nbytes if hasattr(self, "next_observations") else 0
        action_memory = self.actions.nbytes
        rewards_memory = self.rewards.nbytes
        dones_memory = self.dones.nbytes
        timeouts_memory = self.timeouts.nbytes if hasattr(self, "timeouts") else 0

        total_memory = obs_memory + next_obs_memory + action_memory + \
            rewards_memory + dones_memory + timeouts_memory
        return {
            "observations_mb": obs_memory / 1e6,
            "next_obs_mb": next_obs_memory / 1e6,
            "action_mb": action_memory / 1e6,
            "rewards_mb": rewards_memory / 1e6,
            "dones_mb": dones_memory / 1e6,
            "timeouts_mb": timeouts_memory / 1e6,
            "total_mb": total_memory / 1e6,
            "optimization_enabled": self.optimize_memory_usage,
            "n_envs": self.n_envs,
            "buffer_size": self.buffer_size,
            "current_size": self.size(),
            "buffer_shape": {
                "observations": self.observations.shape,
                "actions": self.actions.shape,
                "rewards": self.rewards.shape,
                "dones": self.dones.shape
            }
        }
    
    def get_env_distribution(self) -> dict:
        """
        Get distribution of transitions per environment
        """
        current_size = self.size()
        if current_size == 0:
            return {}
        transitions_per_env = current_size
        return {f"env_{env_id}": transitions_per_env for env_id in range(self.n_envs)}


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized experience replay."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        it_capacity = 1
        while it_capacity < self.buffer_size:
            it_capacity *= 2

        self.sum_tree = SumSegmentTree(it_capacity)
        self.min_tree = MinSegmentTree(it_capacity)
        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Optional[list[dict[str, Any]]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        idx = (self.pos - 1) % self.buffer_size
        priority = self.max_priority ** self.alpha
        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

    def sample(
        self, batch_size: int, env: Optional[Any] = None
    ) -> PrioritizedReplayBufferSamples:
        batch_indices = []
        p_total = self.sum_tree.sum(0, self.size())
        segment = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.uniform(segment * i, segment * (i + 1))
            idx = self.sum_tree.find_prefixsum_idx(mass)
            batch_indices.append(idx)

        samples = self._get_samples(np.array(batch_indices), env)

        self.beta = min(1.0, self.beta + self.beta_increment)
        total = self.sum_tree.sum()
        min_prob = self.min_tree.min() / total
        max_weight = (min_prob * self.size()) ** (-self.beta)
        weights = []
        for idx in batch_indices:
            prob = self.sum_tree[idx] / total
            w = (prob * self.size()) ** (-self.beta)
            weights.append(w / max_weight)

        weight_t = self.to_torch(np.array(weights).reshape(-1, 1))

        return PrioritizedReplayBufferSamples(
            samples.observations,
            samples.actions,
            samples.next_observations,
            samples.dones,
            samples.rewards,
            weight_t,
            np.array(batch_indices),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.buffer_size
            self.sum_tree[idx] = (priority + self.epsilon) ** self.alpha
            self.min_tree[idx] = (priority + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
