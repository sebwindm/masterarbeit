from typing import Union, Type, Optional, Dict, Any, Callable, Tuple

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common import logger
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import is_vectorized_observation
import csv

from .average_reward_adjusted_policy import DQNPolicyAverageRewardAdjusted
from .util import *


class DQNAverageRewardAdjusted(DQN):
    """
    DQNAverageRewardAdjusted is based on stable_baselines3.dqn
    This class overwrites some methods from DQN
    """

    def __init__(self, policy: Union[str, Type[DQNPolicyAverageRewardAdjusted]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 1e-4,
                 buffer_size: int = 1000000,
                 learning_starts: int = 50000,
                 batch_size: Optional[int] = 32,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 train_freq: int = 4,
                 gradient_steps: int = 1,
                 n_episodes_rollout: int = -1,
                 optimize_memory_usage: bool = False,
                 target_update_interval: int = 10000,
                 exploration_fraction: float = 0.15,  # Default for 3 machines: 0.40 edited todo
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.01,  # edited todo
                 max_grad_norm: float = 10,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = 'auto',
                 _init_setup_model: bool = True,

                 # Parameters for util.py: todo
                 alpha: float = 0.01,
                 alpha_min: float = 1e-5,
                 alpha_decay_rate: float = 0.55,
                 alpha_decay_steps: int = 15000  # default for 3 machines: 50000
                 ):

        super(DQNAverageRewardAdjusted, self).__init__(policy,
                                                       env,
                                                       learning_rate,
                                                       buffer_size,
                                                       learning_starts,
                                                       batch_size,
                                                       tau,
                                                       gamma,
                                                       train_freq,
                                                       gradient_steps,
                                                       n_episodes_rollout,
                                                       optimize_memory_usage,
                                                       target_update_interval,
                                                       exploration_fraction,
                                                       exploration_initial_eps,
                                                       exploration_final_eps,
                                                       max_grad_norm,
                                                       tensorboard_log,
                                                       create_eval_env,
                                                       policy_kwargs,
                                                       verbose,
                                                       seed,
                                                       device,
                                                       _init_setup_model)
        self.rho = 0
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay_rate = alpha_decay_rate
        self.alpha_decay_steps = alpha_decay_steps
        self.period_counter = 0
        self.current_observation = None
        self.current_unnormalized_reward = None

        # Create CSV file to store Q-Values for a fixed observation (for debugging purposes):
        with open('../' + 'q_values_learned_results.csv', mode='w') as results_CSV:
            results_writer = csv.writer(results_CSV, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(['Period', 'Action_0', 'Action_1', 'Action_2'])

        # Create CSV file to store rewards after each period
        with open('../' + 'rewards_per_period.csv', mode='w') as rewards_per_period_CSV:
            results_writer = csv.writer(rewards_per_period_CSV, delimiter='\t', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(['Period', 'Reward', 'Rho'])

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the target Q values
                target_q = self.q_net_target(replay_data.next_observations)
                # print("replay_data.next_observations",replay_data.next_observations)

                # Follow greedy policy: use the one with the highest value
                target_q, _ = target_q.max(dim=1)
                # Avoid potential broadcast issue
                target_q = target_q.reshape(-1, 1)
                # 1-step TD target
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q - self.rho

            # TODO: aus dem paper seite 12
            # X = (1 − γ)X γ π 1 (s t , a t ) + γ[r t + γ 1 max X γ π 1 (s t+1 , a) − ρ π ]
            # wir ignorieren komplett (1 − γ)X γ π 1 (s t , a t ) und das γ vor der eckigen Klammer
            # --> wir beachten nur was in der eckigen Klammer steht; der Rest ist für exponential smoothing
            # wir benötigen aber kein exp. smoothing, da wir ein neuronales netz nutzen

            # Get current Q estimates
            current_q = self.q_net(replay_data.observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())
            # print("current q values from replay buffer : \n",current_q)
            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q, target_q)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude='tensorboard')

    def _sample_action(self, learning_starts: int,
                       action_noise: Optional[ActionNoise] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: (Optional[ActionNoise]) Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: (int) Number of steps before learning for the warm-up phase.
        :return: (Tuple[np.ndarray, np.ndarray]) action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        is_random_action = 0
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _, is_random_action = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action, is_random_action

    def collect_rollouts(self,
                         env: VecEnv,
                         callback: BaseCallback,
                         n_episodes: int = 1,
                         n_steps: int = -1,
                         action_noise: Optional[ActionNoise] = None,
                         learning_starts: int = 0,
                         replay_buffer: Optional[ReplayBuffer] = None,
                         log_interval: Optional[int] = None) -> RolloutReturn:
        """
        Collect experiences and store them into a ReplayBuffer.

        :param env: (VecEnv) The training environment
        :param callback: (BaseCallback) Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: (int) Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: (int) Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: (Optional[ActionNoise]) Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: (int) Number of steps before learning for the warm-up phase.
        :param replay_buffer: (ReplayBuffer)
        :param log_interval: (int) Log data every ``log_interval`` episodes
        :return: (RolloutReturn)
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and total_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action, is_random_action = self._sample_action(learning_starts, action_noise)
                # print("action/buffer action/israndomaction ", action, buffer_action,is_random_action)
                # action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                # Afterwards, normalize new observation and reward to make it easier for the
                # neural network to handle the values. Single values of Reward and observation
                # should be (roughly) between -1 and +1
                new_obs, reward, done, infos = env.step(action)
                self.current_unnormalized_reward = reward
                new_obs = normalize_observation(new_obs)  # custom normalization of observation
                reward = normalize_reward(reward)  # custom normalization of reward
                self.period_counter += 1
                if self.period_counter % 5000 == 0:
                    decayed_alpha = self.exp_decay_alpha()
                    print("decayed_alpha: ", decayed_alpha, " | Exploration rate: ", self.exploration_rate)
                self.current_observation = new_obs

                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, total_steps, total_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    # Store only the unnormalized version
                    if self._vec_normalize_env is not None:
                        new_obs_ = self._vec_normalize_env.get_original_obs()
                        reward_ = self._vec_normalize_env.get_original_reward()
                    else:
                        # Avoid changing the original ones
                        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

                    replay_buffer.add(self._last_original_obs, new_obs_, buffer_action, reward_,
                                      done)
                old_observation = self._last_original_obs
                self._last_obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    self._last_original_obs = new_obs_

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1
                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if 0 < n_steps <= total_steps:
                    break

                # Compute the target Q values
                target_st1 = self.q_net_target(th.tensor(new_obs))
                # Follow greedy policy: use the one with the highest value
                target_st1, _ = target_st1.max(dim=1)
                # Avoid potential broadcast issue
                target_st1 = float(target_st1.reshape(-1, 1))

                # Compute the target Q values
                target_st = self.q_net_target(th.tensor(old_observation))
                # Follow greedy policy: use the one with the highest value
                target_st, _ = target_st.max(dim=1)

                # Avoid potential broadcast issue -> IS THIS NECESSARY?
                target_st = float(target_st.reshape(-1, 1))
                # TODO: Welchen der drei indices von target_st nehmen wir? ->maximalen
                # TODO 2: laut paper wird zuerst rho berechnet, aber hier machen wir es einen step verzögert
                # buffer_action.astype(int)[0]
                if is_random_action == 0 or self.period_counter < 1000:
                    decayed_alpha = self.exp_decay_alpha()
                    self.rho = (1 - decayed_alpha) * self.rho + decayed_alpha * (reward_ + target_st1 - target_st)

                # Fixed observation for debugging purposes
                # TODO: use real state and normalize
                obs2 = th.tensor([[0, 2, 2, 1, 0, 2, 1, 1, 1, 0,
                                   0, 0, 0,
                                   6, 0, 0, 0,
                                   0, 1, 0, 0, 0,

                                   0, 0, 2, 1, 1, 4, 1, 0, 1, 0,
                                   0, 0, 0,
                                   3, 0, 0, 0,
                                   0, 0, 0, 0, 0,

                                   0, 1, 0, 1, 6, 1, 1, 2, 2, 0,
                                   0, 0, 0,
                                   2, 0, 0, 0,
                                   0, 2, 1, 0, 0,

                                   0, 1, 0, 0, 3, 1, 1, 3, 2, 0,
                                   0, 0, 0,
                                   2, 0, 0, 0,
                                   0, 1, 0, 0, 0,

                                   0, 0, 0, 1, 2, 0, 0, 2, 2, 0,
                                   0, 0, 0,
                                   6, 0, 0, 0,
                                   0, 0, 1, 0, 0,

                                   0, 0, 3, 2, 0, 1, 4, 0, 0, 0,
                                   0, 0, 0,
                                   7, 0, 0, 0,
                                   0, 1, 1, 1, 1]])
                obs2 = normalize_observation(obs2)
                fix_observation = self.q_net._predict(obs2)[1][0]
                with open('../' + 'q_values_learned_results.csv', mode='a') as results_CSV:
                    results_writer = csv.writer(results_CSV, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow(
                        [self.period_counter, float(fix_observation[0]), float(fix_observation[1]),
                         float(fix_observation[2])])

                # Write reward to CSV file after each period
                with open('../' + 'rewards_per_period.csv', mode='a') as rewards_per_period_CSV:
                    results_writer = csv.writer(rewards_per_period_CSV, delimiter='\t', quotechar='"',
                                                quoting=csv.QUOTE_MINIMAL)
                    results_writer.writerow([self.period_counter, float(reward_), float(self.rho)])

            if done:
                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, total_steps, total_episodes, continue_training)

    def exp_decay_alpha(self) -> float:
        """
        Get the decayed alpha value.
        """
        return exponential_decay(self.alpha_min, self.alpha_decay_rate, self.alpha_decay_steps,
                                      self.period_counter,
                                      self.alpha)

    def predict(self, observation: np.ndarray,
                state: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: (np.ndarray) the input observation
        :param state: (Optional[np.ndarray]) The last states (can be None, used in recurrent policies)
        :param mask: (Optional[np.ndarray]) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (Tuple[np.ndarray, Optional[np.ndarray]]) the model's action and the next state
            (used in recurrent policies), 'is_random_action'(0 or 1) indicates if the action taken was random or not
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            # choose random action
            n_batch = observation.shape[0]
            # action = np.array([self.action_space.sample() for _ in range(n_batch)])
            action = np.array([self.action_space.sample()])
            is_random_action = 1
            vectorized_env = is_vectorized_observation(observation, self.policy.observation_space)
            if not vectorized_env:
                action = action[0]
            # print("is random action",action)
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
            is_random_action = 0
            # print("is nonrandom action", action, observation.shape[0])
        # if type(action) is not np.array:
        #     action = np.array([action])
        return action, state, is_random_action

    def get_q_values_for_current_observation(self):
        """
        Custom method for debugging
        :return: the Q-values for each of the three possible actions and which action was chosen
        """
        action, q_values = self.q_net._predict(th.tensor(normalize_observation(self.current_observation)))
        q1 = float(q_values[0][0])
        q2 = float(q_values[0][1])
        q3 = float(q_values[0][2])
        return q1, q2, q3, int(action[0])
