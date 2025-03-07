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
                 input_preprocessor=lambda x: x,
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

        # Additional attributes for tracking gradient updates and transitions.
        self.num_updates = 0
        self.step_counter = 0
        self.last_state = None
        self.last_action = None

    def act(self, obs) -> int:
        """Selects an action using the Q-network and exploration strategy."""
        # Convert observation to tensor and preprocess.
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        processed_obs = self.input_preprocessor(obs_tensor)
        with torch.no_grad():
            q_vals = self.q_network(processed_obs).squeeze(0).cpu().numpy()
        action = self.explorer.select_action(q_vals)
        # Save the current state and chosen action for later transition creation.
        self.last_state = obs
        self.last_action = action
        return action

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        """Computes target Q-values for a batch of transitions."""
        with torch.no_grad():
            # Compute Q-values for next states using the target network.
            q_next = self.target_network(batched_next_states)
            max_q_next, _ = torch.max(q_next, dim=1)
            # If the episode terminated, we do not bootstrap.
            not_terminated = 1.0 - batch_terminated.float()
            targets = batched_rewards + batched_discounts * max_q_next * not_terminated
        return targets

    def gradient_update(self):
        # For prioritized replay, supply a beta value (could be annealed over training)
        beta = 0.4  # or use self.beta if you maintain one
        sample_result = self.replay_buffer.sample(self.minibatch_size, beta)
        # Unpack the returned tuple
        indices, minibatch, sampling_weights = sample_result
        # Convert sampling weights to a tensor (for later use in loss weighting)
        sampling_weights = torch.tensor(sampling_weights, dtype=torch.float32)

        # Build tensors from the minibatch transitions.
        states = torch.tensor(np.array([transition['state'] for transition in minibatch]), dtype=torch.float32)
        actions = torch.tensor(np.array([transition['action'] for transition in minibatch]), dtype=torch.int64)
        rewards = torch.tensor(np.array([transition['reward'] for transition in minibatch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([transition['next_state'] for transition in minibatch]), dtype=torch.float32)
        discounts = torch.tensor(np.array([transition['discount'] for transition in minibatch]), dtype=torch.float32)
        terminated = torch.tensor(np.array([transition['terminated'] for transition in minibatch]), dtype=torch.bool)

        # Preprocess states if necessary.
        states = self.input_preprocessor(states)
        next_states = self.input_preprocessor(next_states)

        # Compute current Q-values for the taken actions.
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values (this method should handle multi-step targets).
        targets = self.compute_targets(rewards, next_states, discounts, terminated)

        # Compute loss with no reduction initially.
        loss = torch.nn.functional.mse_loss(q_values, targets, reduction='none')
        # Weight the loss by the importance sampling weights and average.
        loss = (loss * sampling_weights.to(loss.device)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.num_updates += 1
        if self.num_updates % self.gradient_updates_per_target_refresh == 0:
            self.target_network = target_network_refresh(self.q_network)

    def process_transition(self, obs: int, reward: float, terminated: bool, truncated: bool) -> None:
        """Processes a transition and performs learning updates.

        This function should be called after taking an action.
        It appends the transition to the replay buffer and, if conditions are met,
        performs a gradient update.
        """
        # Optionally transform the reward.
        reward = self.reward_phi(reward)
        # Append the transition using the stored last_state and last_action.
        self.replay_buffer.append(self.last_state, self.last_action, reward, obs, terminated, truncated)
        self.step_counter += 1
        # Perform a gradient update if the replay buffer is sufficiently full
        # and if the update frequency condition is satisfied.
        if len(self.replay_buffer) >= self.min_replay_size_before_updates and (self.step_counter % self.gradient_update_frequency == 0):
            self.gradient_update()
