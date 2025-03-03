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
        # your code here
        pass
        # end your code

    def sample(self, n_transitions):
        assert len(self.buffer) >= n_transitions
        batch_indices = np.random.choice(len(self.buffer), size=n_transitions, replace=False)
        batch = [self.create_multistep_transition(index) for index in batch_indices]
        return batch
