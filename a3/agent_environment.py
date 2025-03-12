import torch
import numpy as np
from tqdm import tqdm

def agent_environment_episode_loop(agent, env, num_episodes, debug=False, track_q=False,seed=None):
    episode_returns = []
    mean_q_predictions = []  # the average Q-value for all state-action pairs visited in the episode
    for episode in tqdm(range(num_episodes)):
        if track_q:
            episode_q_values = []
        observation, info = env.reset()
        total_return = 0.0
        
        while True:
            # If tracking Q-values, compute the mean Q-value prediction for the current observation.
            if track_q:
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                processed_obs = agent.input_preprocessor(obs_tensor)
                with torch.no_grad():
                    q_vals = agent.q_network(processed_obs).squeeze(0).cpu().numpy()
                episode_q_values.append(np.mean(q_vals))
            
            # Agent selects an action based on the current observation.
            action = agent.act(observation)
            # Environment returns the next observation, reward, termination and truncation signals.
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            
            # Agent processes the transition (e.g., storing in the replay buffer and learning)
            agent.process_transition(next_observation, reward, terminated, truncated)
            
            observation = next_observation
            # End the episode if termination or truncation is signaled.
            if terminated or truncated:
                break
        episode_returns.append(total_return)
        if track_q:
            mean_q_predictions.append(np.mean(episode_q_values))
        if debug:
            print(f"Episode {episode}: Return = {total_return}")
    
    if track_q:
        return episode_returns, mean_q_predictions
    else:
        return episode_returns, None
