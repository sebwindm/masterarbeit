from typing import Optional, List, Callable, Union, Type, Any, Dict

import gym, csv
import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor



with open('../' + 'q_values_learned_results.csv', mode='w') as results_CSV:
    results_writer = csv.writer(results_CSV, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['Action_0','Action_1', 'Action_2'])

prediction_counter = 0

class QNetwork_average_reward_adjusted(QNetwork):
    def __init__(self, observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 features_extractor: nn.Module,
                 features_dim: int,
                 net_arch: Optional[List[int]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True):
        super(QNetwork_average_reward_adjusted, self).__init__(observation_space,
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



class DQN_policy_average_reward_adjusted(DQNPolicy):
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
        super(DQN_policy_average_reward_adjusted, self).__init__(observation_space,
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

    def make_q_net(self) -> QNetwork_average_reward_adjusted:
        return QNetwork_average_reward_adjusted(**self.net_args).to(self.device)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:

        action, q_values = self.q_net._predict(obs, deterministic=deterministic)
        global prediction_counter
        if prediction_counter % 100 == 0:
            #print("Action: ",action," | Q-values: ",q_values)
            #print("Observation: ", obs)
            # Fixed observation
            obs2 = th.tensor([[10.,  0.,  0.,  0.,  5.,  0.,  7.,  1.,  0.,  0.,  5.,  1.,  6.,  1.,
              0.,  0.,  3.,  0.,  8.,  0.,  0.,  0.,  3.,  2., 14.,  2.,  0.,  0.,
              4.,  2., 17.,  1.,  0.,  0.,  3.,  1.]])
            fix_observation = self.q_net._predict(obs2)[1][0]
            #print("fix observation:", fix_observation)
            with open('../' + 'q_values_learned_results.csv', mode='a') as results_CSV:
                results_writer = csv.writer(results_CSV, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                results_writer.writerow([float(fix_observation[0]),float(fix_observation[1]),float(fix_observation[2])])

        prediction_counter += 1
        return action



AraPolicy = DQN_policy_average_reward_adjusted
register_policy("MlpAverageRewardAdjustedPolicy", AraPolicy)
