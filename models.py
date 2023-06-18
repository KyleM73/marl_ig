import gymnasium as gym
import numpy as np
import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MlpExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, data_dim: int=256, context_dim: int=16, activation_fn: torch.nn.Module=torch.nn.Tanh):
        super().__init__(observation_space, features_dim=data_dim+context_dim)

        extractors = {}
        for key, subspace in observation_space.spaces.items():
            if key == "data":
                input_dim = np.product(subspace.shape)
                extractors[key] = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(input_dim, data_dim),
                    activation_fn(),
                    torch.nn.Linear(data_dim, data_dim),
                    activation_fn(),
                    )
            elif key == "context":
                extractors[key] = torch.nn.Sequential(
                    torch.nn.Linear(subspace.shape[0], context_dim),
                    activation_fn(),
                    )
        self.extractors = torch.nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        # observations: Dict("data", "context")
        # return (batch, self._features_dim)
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

class CnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, data_dim: int=256, context_dim: int=16, activation_fn: torch.nn.Module=torch.nn.Tanh):
        super().__init__(observation_space, features_dim=data_dim+context_dim)

        extractors = {}
        for key, subspace in observation_space.spaces.items():
            if key == "data":
                extractors[key] = torch.nn.Sequential(
                    torch.nn.Conv2d(subspace.shape[0], 16, kernel_size=25, stride=5, padding=0),
                    activation_fn(),
                    torch.nn.Conv2d(16, 32, kernel_size=6, stride=3, padding=0),
                    activation_fn(),
                    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                    torch.nn.Flatten(),
                    torch.nn.Linear(5184, data_dim),
                    activation_fn(),
                )
            elif key == "context":
                extractors[key] = torch.nn.Sequential(
                    torch.nn.Linear(subspace.shape[0], context_dim),
                    activation_fn(),
                    )
        self.extractors = torch.nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        # observations: Dict("data", "context")
        # return (batch, self._features_dim)
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)