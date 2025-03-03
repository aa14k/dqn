import torch
import numpy as np

def agent_environment_episode_loop(agent, env, num_episodes, debug=False, track_q=False):
    episode_returns = []
    mean_q_predictions = [] # the average Q-value for all state-action pairs visited in the episode
    for episode in range(num_episodes):
        if track_q:
            episode_q_values = []
        observation, info = env.reset()
        # start your code
        pass
        # end your code
    if track_q:
        return episode_returns, mean_q_predictions
    else:
        return episode_returns, None
