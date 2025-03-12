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
                 reward_phi=lambda reward: reward,
                 replay_type = 'uniform'):
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
        self.replay_type = replay_type

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
        beta = 0.4
        indices, minibatch, sampling_weights = self.replay_buffer.sample(self.minibatch_size, beta)

        # Convert sampling weights (already a NumPy array) to a single Tensor
        sampling_weights = torch.from_numpy(sampling_weights.astype(np.float32))

        # Gather each field into ONE NumPy array, then convert to Tensor
        states_np = np.array([t['state'] for t in minibatch], dtype=np.float32)
        actions_np = np.array([t['action'] for t in minibatch], dtype=np.int64)
        rewards_np = np.array([t['reward'] for t in minibatch], dtype=np.float32)
        next_states_np = np.array([t['next_state'] for t in minibatch], dtype=np.float32)
        discounts_np = np.array([t['discount'] for t in minibatch], dtype=np.float32)
        terminated_np = np.array([t['terminated'] for t in minibatch], dtype=bool)

        # Convert all those NumPy arrays to Torch tensors
        states = torch.from_numpy(states_np)
        actions = torch.from_numpy(actions_np)
        rewards = torch.from_numpy(rewards_np)
        next_states = torch.from_numpy(next_states_np)
        discounts = torch.from_numpy(discounts_np)
        terminated = torch.from_numpy(terminated_np)

        # Preprocess if needed
        states = self.input_preprocessor(states)
        next_states = self.input_preprocessor(next_states)

        q_values_all = self.q_network(states)
        # gather Q for the chosen actions
        q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        targets = self.compute_targets(rewards, next_states, discounts, terminated)

        # MSE loss, no reduction
        elementwise_loss = torch.nn.functional.mse_loss(q_values, targets, reduction='none')

        # Multiply by sampling weights
        loss = (elementwise_loss * sampling_weights.to(elementwise_loss.device)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ─────────────────────────────────────────────────────────────
        # 1) Compute absolute TD errors for each transition
        if self.replay_type == 'tderror':
            error = q_values-targets
            errors = (error).detach().abs().cpu().numpy()

            # 2) Update replay buffer priorities with these TD errors
            self.replay_buffer.update_priorities(indices, errors)
        elif self.replay_type == 'targets':
            error = targets
            errors = (error).detach().abs().cpu().numpy()

            # 2) Update replay buffer priorities with these TD errors
            self.replay_buffer.update_priorities(indices, errors)
        elif self.replay_type == 'uniform':
            pass

        # ─────────────────────────────────────────────────────────────

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