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

    def select_action(self, action_values) -> int:
        epsilon_decay_step_size = (self.start_epsilon - self.end_epsilon) / self.decay_steps
        epsilon = max(self.start_epsilon - self.steps * epsilon_decay_step_size, self.end_epsilon)
        action_probs = epsilon_greedy_explorers.compute_epsilon_greedy_action_probs(action_values, epsilon)
        self.steps += 1
        return np.random.choice(len(action_probs), p=action_probs)

# Convolutional Q-network for the Jumping Task with grayscale input.
class JumpingQNetwork(nn.Module):
    def __init__(self, num_actions):
        super(JumpingQNetwork, self).__init__()
        # The network expects input of shape (N, 1, 84, 84).
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # roughly outputs (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # roughly outputs (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # roughly outputs (64, 7, 7)
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return"):
    # Compute running average over a 200-episode window (centered)
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
                           rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                           max_number_of_steps=300, two_obstacles=False, finish_jump=False)
    elif config_num == 2:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[30, 40],
                           agent_w=7, agent_h=7, agent_init_pos=0, agent_speed=1,
                           obstacle_position=0, obstacle_size=(11,17),
                           rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                           max_number_of_steps=300, two_obstacles=False, finish_jump=False,
                           jump_height=24)
    else:
        env = JumpTaskEnv(scr_w=60, scr_h=60, floor_height_options=[10, 20], obstacle_locations=[20, 30, 40],
                           agent_w=5, agent_h=10, agent_init_pos=0, agent_speed=1,
                           obstacle_position=0, obstacle_size=(9,10),
                           rendering=render, zoom=8, slow_motion=True, with_left_action=False,
                           max_number_of_steps=300, two_obstacles=True, finish_jump=False)
    return env

def input_preprocessor(x):
    """
    Preprocess input images for the Jumping Task.
    Expected input:
      - A 2D tensor (H, W) for a single grayscale image, or
      - A 3D tensor (N, H, W) for a batch of grayscale images,
      - Or other formats.
    Output:
      A 4D tensor of shape (N, 1, 84, 84) with values in [0, 1].
    """
    # Case 1: Single image (H, W)
    if len(x.shape) == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
    # Case 2: Batched images (N, H, W)
    elif len(x.shape) == 3:
        # Assume shape is (N, H, W)
        x = x.unsqueeze(1)  # -> (N, 1, H, W)
    # If already 4D, assume NCHW
    x = x.float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(84, 84), mode='bilinear', align_corners=False)
    return x

def reward_phi(r):
    return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Which environment configuration (1, 2, or 3)", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--num-training-episodes", help="Number of training episodes", default=20000, type=int)
    parser.add_argument("--run-label", help="Run label (akin to a random seed)", default=1, type=int)
    parser.add_argument("--min-replay-size-before-updates", help="Minimum replay buffer size before updates", default=32, type=int)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--track-q", action='store_true')
    args = parser.parse_args()

    if not os.path.exists("data"):
        os.makedirs("data")

    env = get_env(args.config, args.render)
    num_actions = env.action_space.n  # For the Jumping Task, actions are typically 2 (e.g., right and up)

    # Run experiment over three seeds.
    seeds = [42, 43, 44]
    all_episode_returns = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        explorer = LinearDecayEpsilonGreedyExploration(1.0, 0.01, 10, num_actions)
        # Use our convolutional Q-network for grayscale images.
        q_network = JumpingQNetwork(num_actions)
        # Update optimizer hyperparameters to baseline settings:
        optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001, eps=0.0001)
        # Use replay buffer of size 100 (can be tuned as needed).
        buffer = replay_buffer.ReplayBuffer(100, discount=0.99, n_step=1)

        agent = dqn.DQN(q_network, optimizer, buffer, explorer, 0.99, 10, gradient_update_frequency=1,
                        input_preprocessor=input_preprocessor,
                        min_replay_size_before_updates=args.min_replay_size_before_updates,
                        reward_phi=reward_phi)
        print(f"Running training for seed {seed}...")
        episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, args.num_training_episodes, args.debug, args.track_q)
        all_episode_returns.append(episode_returns)
        df = pd.DataFrame(episode_returns)
        df.to_csv(f'data/config_{args.config}_run_{seed}.csv', index=False)

    produce_plots_for_all_configs()