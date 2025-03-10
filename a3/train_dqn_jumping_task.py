import argparse
import gymnasium as gym
import agent_environment
import numpy as np
import matplotlib.pyplot as plt
import jumping_task
from jumping_task.envs import JumpTaskEnv
import epsilon_greedy_explorers as epsilon_greedy_explorers
import torch
import torch.nn as nn
import replay_buffer as replay_buffer  # assuming replay_buffer_old is now renamed/re-exported
import dqn
import double_dqn
import pandas as pd
import os
from joblib import Parallel, delayed

CCID = "aayoub"

class LinearDecayEpsilonGreedyExploration:
    """Epsilon-greedy with linearly decaying epsilon.

    Args:
      start_epsilon: initial epsilon value
      end_epsilon: final epsilon value
      decay_steps: number of steps over which epsilon decays
      num_actions: number of possible actions
    """
    def __init__(self, start_epsilon, end_epsilon, decay_steps, num_actions):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        assert start_epsilon >= end_epsilon >= 0
        self.epsilon = start_epsilon
        self.decay_steps = decay_steps
        self.num_actions = num_actions
        self.steps = 0

    def select_action(self, action_values):
        epsilon_decay_step_size = (self.start_epsilon - self.end_epsilon) / self.decay_steps
        epsilon = max(self.start_epsilon - self.steps * epsilon_decay_step_size, self.end_epsilon)
        action_probs = epsilon_greedy_explorers.compute_epsilon_greedy_action_probs(action_values, epsilon)
        self.steps += 1
        return np.random.choice(len(action_probs), p=action_probs)

class CountBasedExploration:
    """
    Count-based exploration that uses state-action counts.
    
    For a given state and each action, maintain a count of how many times that 
    action has been taken in that state. The exploration bonus is computed as:
    
        bonus = bonus_coef / sqrt(N(s,a) + 1)
    
    and the selected action maximizes Q(s,a) + bonus, with ties broken randomly.
    """
    def __init__(self, num_actions, bonus_coef=0.1):
        self.num_actions = num_actions
        self.bonus_coef = bonus_coef
        # Store counts in a dictionary keyed by (state, action) pair.
        self.counts = {}
        self.is_count_based = True
        self.steps = 0

    def _get_state_action_key(self, state, action):
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if state.ndim == 3:
            state = state[0]
        state_key = tuple(state.flatten().tolist())
        return (state_key, action)

    def randomized_argmax(self, q_values):
        max_val = np.max(q_values)
        max_indices = np.flatnonzero(q_values == max_val)
        return np.random.choice(max_indices)

    def select_action(self, q_values, state):
        q_bonus = np.copy(q_values)
        for a in range(self.num_actions):
            key = self._get_state_action_key(state, a)
            count = self.counts.get(key, 0)
            bonus = self.bonus_coef / np.sqrt(count + 1)
            q_bonus[a] += bonus
        action = self.randomized_argmax(q_bonus)
        key = self._get_state_action_key(state, action)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.steps += 1
        return action

# Convolutional Q-network for the Jumping Task with grayscale input.
# Modified to expect input of shape (N, 1, 60, 60)
class JumpingQNetwork(nn.Module):
    def __init__(self, num_actions):
        super(JumpingQNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # with 60x60 input: outputs (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # outputs (64, 6, 6)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # outputs (64, 4, 4)
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),  # 64 * 4 * 4 = 1024
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return"):
    running_avg = np.mean(np.array(episode_returns_list), axis=0)
    new_running_avg = running_avg.copy()
    for i in range(len(running_avg)):
        new_running_avg[i] = np.mean(running_avg[max(0, i-100):min(len(running_avg), i+100)])
    running_avg = new_running_avg

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(running_avg)), running_avg, color='r', label=label)
    plt.title(f"({CCID}) Episodic Returns")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(file)
    plt.close()

def extract_config(filename_without_ext):
    configs = ['config_1', 'config_2', 'config_3']
    for configuration in configs:
        if configuration in filename_without_ext:
            return configuration
    return None

def produce_plots_for_all_configs():
    configs = ['config_1', 'config_2', 'config_3']
    data_dict = {config: [] for config in configs}
    files = os.listdir("data")
    for file in files:
        full_path = os.path.join("data", file)
        if os.path.isfile(full_path) and os.path.splitext(file)[-1] == '.csv':
            config = extract_config(os.path.splitext(file)[0])
            assert config is not None, f"{file} is not in the required format."
            df = pd.read_csv(full_path)
            data_dict[config].append(np.squeeze(df.values))
    for configuration in configs:
        if data_dict[configuration]:
            plot_alg_results(data_dict[configuration], f"dqn_jumping_{configuration}.png")

def get_env(config_num, render=False):
    if config_num == 1:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 25, 30],
                           agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                           obstacle_position=0, obstacle_size=(9,10),
                           rendering=render, zoom=8, slow_motion=False, with_left_action=False,
                           max_number_of_steps=300, two_obstacles=False, finish_jump=False)
    elif config_num == 2:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[30, 40],
                           agent_w=7, agent_h=7, agent_init_pos=0, agent_speed=1,
                           obstacle_position=0, obstacle_size=(11,17),
                           rendering=render, zoom=8, slow_motion=False, with_left_action=False,
                           max_number_of_steps=300, two_obstacles=False, finish_jump=False,
                           jump_height=24)
    else:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 30, 40],
                           agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                           obstacle_position=0, obstacle_size=(9,10),
                           rendering=render, zoom=8, slow_motion=False, with_left_action=False,
                           max_number_of_steps=300, two_obstacles=True, finish_jump=False)
    return env

# Input preprocessor now keeps the image at its original 60x60 resolution.
def input_preprocessor(x):
    if len(x.shape) == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif len(x.shape) == 3:
        x = x.unsqueeze(1)
    x = x.float()
    # No need to interpolate since the environment already outputs 60x60 images.
    return x

def reward_phi(r):
    return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", help="Which environment configuration (1, 2, or 3)", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--num-training-episodes", help="Number of training episodes", default=10_000, type=int)
    parser.add_argument("--run-label", help="Run label (akin to a random seed)", default=1, type=int)
    parser.add_argument("--min-replay-size-before-updates", help="Minimum replay buffer size before updates", default=32, type=int)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--track-q", action='store_true')
    parser.add_argument("--exploration", choices=["epsilon", "count"], default="epsilon")
    args = parser.parse_args()

    if not os.path.exists("data"):
        os.makedirs("data")
    config = 1
    env = get_env(config, args.render)
    num_actions = env.action_space.n
    buffer_size = 25_000 
    discount = 0.99
    n_step = 20

    # Choose exploration strategy.
    if args.exploration == "epsilon":
        explorer = LinearDecayEpsilonGreedyExploration(1.0, 0.001, 100_000, num_actions)
    else:
        explorer = CountBasedExploration(num_actions, bonus_coef=1000)

    # Define a function to run the experiment for one seed.
    def run_experiment_for_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Use our convolutional Q-network for grayscale images.
        q_network = JumpingQNetwork(num_actions)
        optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001, eps=1e-4)
        buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)
        agent = dqn.DQN(q_network, optimizer, buffer, explorer, discount, 100,
                        min_replay_size_before_updates=500,
                        input_preprocessor=input_preprocessor,
                        minibatch_size=128,
                        reward_phi=reward_phi)
        # If count-based exploration is used, override act() to pass the raw observation.
        if args.exploration == "count":
            def act_override(obs):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                processed_obs = agent.input_preprocessor(obs_tensor)
                with torch.no_grad():
                    q_vals = agent.q_network(processed_obs).squeeze(0).cpu().numpy()
                action = agent.explorer.select_action(q_vals, obs)
                agent.last_state = obs
                agent.last_action = action
                return action
            agent.act = act_override

        print(f"Running training for seed {seed} with {args.exploration} exploration...")
        episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, args.num_training_episodes, args.debug, args.track_q, seed)
        # Save CSV for this seed.
        df = pd.DataFrame(episode_returns)
        df.to_csv(f'data/config_{config}_run_{seed}.csv', index=False)
        return episode_returns

    seeds = [38,39,40]
    # Run experiments in parallel.
    for config in [1]:
        env = get_env(config, args.render)
        all_episode_returns = Parallel(n_jobs=4)(delayed(run_experiment_for_seed)(seed) for seed in seeds)
        produce_plots_for_all_configs()