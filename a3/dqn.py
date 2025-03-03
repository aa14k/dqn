import numpy as np
import torch
import copy
import collections


def target_network_refresh(q_network):
    target_network = copy.deepcopy(q_network)
    return target_network


class DQN:
    """Class that implements Deep Q-networks."""

    def __init__(self,
                 q_network,
                 optimizer,
                 replay_buffer,
                 explorer,
                 discount,
                 gradient_updates_per_target_refresh,
                 gradient_update_frequency=1,
                 input_preprocessor= lambda x: x,
                 minibatch_size=32,
                 min_replay_size_before_updates=32,
                 track_statistics=False,
                 reward_phi=lambda reward: reward):
        self.q_network = q_network
        self.optimizer = optimizer
        self.target_network = target_network_refresh(self.q_network)
        self.replay_buffer = replay_buffer
        self.explorer = explorer
        self.discount = discount
        self.gradient_updates_per_target_refresh = gradient_updates_per_target_refresh
        self.gradient_update_frequency = gradient_update_frequency
        self.input_preprocessor = input_preprocessor
        self.minibatch_size = minibatch_size
        self.min_replay_size_before_updates = min_replay_size_before_updates
        self.track_statistics = track_statistics
        self.reward_phi = reward_phi
        # your code here
        # end your code

    def act(self, obs) -> int:
        """Returns an integer 
        """
        # Your code here
        action = 0 # replace this line
        # End your code here
        return action

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        # your code here
        pass
        # End your code here

    def gradient_update(self):
        minibatch = self.replay_buffer.sample(self.minibatch_size)
        # your code here
        # End your code here


    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Observe consequences of the last action and update estimates accordingly.

        Returns:
            None
        """
        reward = self.reward_phi(reward)
        # append transition to buffer
        # do gradient updates if necessary
        # refresh target networks if needed, etc.
        
        # Your code here
        # End your code here
