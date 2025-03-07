import dqn
import torch

class DoubleDQN(dqn.DQN):

    def compute_targets(self, batched_rewards, batched_next_states, batched_discounts, batch_terminated):
        # Begin your code
        with torch.no_grad():
            # Use the online network to choose the greedy actions for the next states.
            q_online_next = self.q_network(batched_next_states)
            greedy_actions = torch.argmax(q_online_next, dim=1, keepdim=True)
            
            # Use the target network to evaluate the selected actions.
            q_target_next = self.target_network(batched_next_states)
            q_selected = q_target_next.gather(1, greedy_actions).squeeze(1)
            
            # For terminal transitions, zero out the bootstrapped value.
            not_terminated = 1.0 - batch_terminated.float()
            
            # Compute the target: immediate reward plus discounted Q-value of the next state.
            targets = batched_rewards + batched_discounts * q_selected * not_terminated
        return targets
        # End your code