import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, discount=0.99, n_step=1):
        self.buffer = collections.deque([], maxlen=buffer_size)
        self.discount = discount
        self.n_step = n_step

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, terminated, truncated):
        transition = {'state': state,
                      'action': action,
                      'reward': reward,
                      'next_state': next_state,
                      'discount': self.discount,
                      'terminated': terminated,
                      'truncated': truncated}
        self.buffer.append(transition)

    def create_multistep_transition(self, index):
        # Start with the transition at the given index.
        transition = self.buffer[index]
        state = transition['state']
        action = transition['action']
        n_step_reward = 0.0
        discount_factor = 1.0
        next_state = transition['next_state']
        terminated = transition['terminated']
        truncated = transition['truncated']

        # Accumulate n-step rewards (or until termination/truncation)
        for i in range(self.n_step):
            if index + i >= len(self.buffer):
                break
            t = self.buffer[index + i]
            n_step_reward += discount_factor * t['reward']
            discount_factor *= self.discount
            next_state = t['next_state']
            terminated = t['terminated']
            truncated = t['truncated']
            if terminated:
                # If the episode ended, set the effective discount to 0 and break.
                discount_factor = 0.0
                break
            if truncated:
                # If the episode was truncated break.
                break

        return {'state': state,
                'action': action,
                'reward': n_step_reward,
                'next_state': next_state,
                'discount': discount_factor,
                'terminated': terminated,
                'truncated': truncated}

    def sample(self, n_transitions):
        assert len(self.buffer) >= n_transitions
        batch_indices = np.random.choice(len(self.buffer), size=n_transitions, replace=False)
        batch = [self.create_multistep_transition(index) for index in batch_indices]
        return batch
