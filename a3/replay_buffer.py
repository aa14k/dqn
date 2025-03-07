import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, discount=0.99, n_step=1, alpha=0.6):
        """
        Parameters:
            buffer_size (int): Maximum number of transitions to store.
            discount (float): Discount factor used for n‑step returns.
            n_step (int): Number of steps to accumulate for multi-step returns.
            alpha (float): Exponent controlling the degree of prioritization (α=0 is uniform sampling).
        """
        self.buffer = []         # list to store transitions
        self.priorities = []     # parallel list to store priority for each transition
        self.max_size = buffer_size
        self.discount = discount
        self.n_step = n_step
        self.alpha = alpha
        self.pos = 0             # pointer for circular replacement

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, terminated, truncated):
        """
        Add a new transition with an initial priority equal to the current max priority (or 1.0 if empty).
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'discount': self.discount,  # used later in n-step return calculation
            'terminated': terminated,
            'truncated': truncated
        }
        # New transitions get maximum priority (or 1.0 if buffer is empty)
        priority = 1.0 if len(self.priorities) == 0 else max(self.priorities)
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.max_size

    def create_multistep_transition(self, index):
        """
        Given a starting index, accumulate rewards over n steps (or until termination/truncation)
        and return a new transition dictionary with the n‑step reward and effective discount.
        """
        transition = self.buffer[index]
        state = transition['state']
        action = transition['action']
        n_step_reward = 0.0
        discount_factor = 1.0
        next_state = transition['next_state']
        terminated = transition['terminated']
        truncated = transition['truncated']

        # Accumulate rewards over up to n_step transitions
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
                discount_factor = 0.0
                break
            if truncated:
                break

        return {
            'state': state,
            'action': action,
            'reward': n_step_reward,
            'next_state': next_state,
            'discount': discount_factor,
            'terminated': terminated,
            'truncated': truncated
        }

    def sample(self, n_transitions, beta=0.4):
        """
        Samples a batch of transitions according to their priorities.
        
        Parameters:
            n_transitions (int): Number of transitions to sample.
            beta (float): Importance-sampling exponent (β ≥ 0). Typically, β is annealed from a small value to 1.
                          
        Returns:
            indices (np.array): Indices of the sampled transitions.
            batch (list): List of n-step transitions.
            weights (np.array): Importance-sampling weights for each sampled transition.
        """
        assert len(self.buffer) >= n_transitions, "Not enough transitions to sample from."
        ps = np.array(self.priorities)
        # Compute sampling probabilities: P(i) ∝ p_i^α
        probs = ps ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), size=n_transitions, replace=False, p=probs)
        
        # Importance sampling weights: w_i = (N * P(i))^(-β), normalized by max weight.
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.create_multistep_transition(idx) for idx in indices]
        return indices, batch, weights

    def update_priorities(self, indices, priorities):
        """
        Update the priorities for the transitions at the given indices.
        """
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = p