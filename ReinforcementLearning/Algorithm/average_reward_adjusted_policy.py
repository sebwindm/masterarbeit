from typing import Optional, List, Callable, Union, Type, Any, Dict
import gym, csv
import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class QNetworkAverageRewardAdjusted(QNetwork):
    """
    QNetworkAverageRewardAdjusted is based on stable_baselines3.dqn.policies.QNetwork.
    This class overwrites some methods from QNetwork.
    Documentation at https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 features_extractor: nn.Module,
                 features_dim: int,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True):
        super(QNetworkAverageRewardAdjusted, self).__init__(observation_space,
                                                            action_space,
                                                            features_extractor,
                                                            features_dim,
                                                            net_arch,
                                                            device,
                                                            activation_fn,
                                                            normalize_images)

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.forward(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action, q_values


class DQNPolicyAverageRewardAdjusted(DQNPolicy):
    """
    DQNPolicyAverageRewardAdjusted is based on stable_baselines3.dqn.policies.DQNPolicy
    This class overwrites some methods from DQNPolicy
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super(DQNPolicyAverageRewardAdjusted, self).__init__(observation_space,
                                                             action_space,
                                                             lr_schedule,
                                                             net_arch,
                                                             device,
                                                             activation_fn,
                                                             features_extractor_class,
                                                             features_extractor_kwargs,
                                                             normalize_images,
                                                             optimizer_class,
                                                             optimizer_kwargs)


    def make_q_net(self) -> QNetworkAverageRewardAdjusted:
        # Make sure we always have separate networks for feature extractors etc
        features_extractor = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        features_dim = features_extractor.features_dim
        return QNetworkAverageRewardAdjusted(features_extractor=features_extractor, features_dim=features_dim, **self.net_args).to(self.device)


    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        action, q_values = self.q_net._predict(obs, deterministic=deterministic)
        return action


register_policy("MlpAverageRewardAdjustedPolicy", DQNPolicyAverageRewardAdjusted)
