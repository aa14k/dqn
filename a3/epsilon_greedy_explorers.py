import numpy as np

def compute_epsilon_greedy_action_probs(q_vals, epsilon):
    """Takes in Q-values and produces epsilon-greedy action probabilities
    where ties are broken evenly.

    Args:
        q_vals: a numpy array of action values
        epsilon: epsilon-greedy epsilon in ([0,1])
         
    Returns:
        numpy array of action probabilities
    """
    # Ensure q_vals is one-dimensional.
    assert len(q_vals.shape) == 1
    
    num_actions = q_vals.shape[0]
    
    # Start with each action receiving epsilon/num_actions probability.
    action_probabilities = np.full(num_actions, epsilon / num_actions, dtype=float)
    
    # Identify the greedy actions (could be more than one if there are ties).
    best_actions = np.flatnonzero(q_vals == np.max(q_vals))
    
    # Distribute the remaining probability mass evenly among the greedy actions.
    action_probabilities[best_actions] += (1 - epsilon) / len(best_actions)
    
    # Final check (optional) to ensure probabilities sum to 1.
    assert np.allclose(action_probabilities.sum(), 1.0)
    
    return action_probabilities	

class ConstantEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
    """

    def __init__(self, epsilon, num_actions):
        self.epsilon = epsilon
        self.num_actions = num_actions

    def select_action(self, action_values) -> int:
        action_probs = compute_epsilon_greedy_action_probs(action_values, self.epsilon)
        return np.random.choice(len(action_probs), p=action_probs)