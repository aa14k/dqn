import dqn
import torch

class DoubleDQN(dqn.DQN):

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        # begin your code
        pass
        # end your code