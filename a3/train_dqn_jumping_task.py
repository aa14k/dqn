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
import itertools
from joblib import Parallel, delayed

CCID = "aayoub"

class LinearDecayEpsilonGreedyExploration:
    """Epsilon-greedy with linearly decaying epsilon."""
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
    """
    def __init__(self, num_actions, bonus_coef=0.1):
        self.num_actions = num_actions
        self.bonus_coef = bonus_coef
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
            nn.Linear(64 * 4 * 4, 512),
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

def plot_hyperparam_results(hyperparams):
    lr_str = str(hyperparams['lr']).replace(".", "p")
    disc_str = str(hyperparams['discount']).replace(".", "p")
    n_step = hyperparams['n_step']
    minibatch_size = hyperparams['minibatch_size']
    num_updates = hyperparams['num_updates_per_gradient']
    pattern = f"config_1_lr_{lr_str}_nstep_{n_step}_mb_{minibatch_size}_disc_{disc_str}_nupdate_{num_updates}_"
    data = []
    files = os.listdir("data")
    for file in files:
        if file.endswith(".csv") and pattern in file:
            full_path = os.path.join("data", file)
            df = pd.read_csv(full_path)
            data.append(np.squeeze(df.values))
    if data:
        plot_file = f"dqn_jumping_config1_lr_{lr_str}_nstep_{n_step}_mb_{minibatch_size}_disc_{disc_str}_nupdate_{num_updates}.png"
        plot_alg_results(data, plot_file)

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

def input_preprocessor(x):
    if len(x.shape) == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif len(x.shape) == 3:
        x = x.unsqueeze(1)
    x = x.float()
    return x

def reward_phi(r):
    return r

def run_experiment_for_seed(seed, hyperparams, env_config, args):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = get_env(env_config, args.render)
    num_actions = env.action_space.n
    buffer_size = 25_000 
    lr = hyperparams['lr']
    n_step = hyperparams['n_step']
    minibatch_size = hyperparams['minibatch_size']
    discount = hyperparams['discount']
    num_updates_per_gradient = hyperparams['num_updates_per_gradient']

    if args.exploration == "epsilon":
        explorer = LinearDecayEpsilonGreedyExploration(1.0, 0.001, 10_000, num_actions)
    else:
        explorer = CountBasedExploration(num_actions, bonus_coef=1000)

    q_network = JumpingQNetwork(num_actions)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=1e-4)
    buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)
    agent = dqn.DQN(q_network, optimizer, buffer, explorer, discount, num_updates_per_gradient,
                    min_replay_size_before_updates=500,
                    input_preprocessor=input_preprocessor,
                    minibatch_size=minibatch_size,
                    reward_phi=reward_phi)
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

    print(f"Running training for seed {seed} with exploration '{args.exploration}' and hyperparams: {hyperparams}")
    episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, args.num_training_episodes, args.debug, args.track_q, seed)
    
    lr_str = str(lr).replace(".", "p")
    disc_str = str(discount).replace(".", "p")
    filename = f"data/config_{env_config}_lr_{lr_str}_nstep_{n_step}_mb_{minibatch_size}_disc_{disc_str}_nupdate_{num_updates_per_gradient}_run_{seed}.csv"
    df = pd.DataFrame(episode_returns)
    df.to_csv(filename, index=False)
    return episode_returns

def run_experiment_and_plot(hyperparams, env_config, args, seed):
    # Run the experiment for a given hyperparameter configuration and then produce its plot.
    run_experiment_for_seed(seed, hyperparams, env_config, args)
    plot_hyperparam_results(hyperparams)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-training-episodes", help="Number of training episodes", default=1_000, type=int)
    parser.add_argument("--run-label", help="Run label (used as seed)", default=42, type=int)
    parser.add_argument("--min-replay-size-before-updates", help="Minimum replay buffer size before updates", default=32, type=int)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--track-q", action='store_true')
    parser.add_argument("--exploration", choices=["epsilon", "count"], default="epsilon")
    parser.add_argument("--sweep", action='store_true', help="Perform hyperparameter sweep over several values")
    args = parser.parse_args()

    if not os.path.exists("data"):
        os.makedirs("data")
    
    env_config = 1
    default_hyperparams = {
        "lr": 0.001,
        "n_step": 20,
        "minibatch_size": 128,
        "discount": 0.99,
        "num_updates_per_gradient": 100
    }
    
    # Use one seed per hyperparameter configuration
    seed = args.run_label
    
    if args.sweep:
        hyperparams_grid = {
            "lr": [1e-1,1e-2,1e-3,1e-4,1e-5],
            "n_step": [5,10,20],
            "minibatch_size": [64, 128,256,512],
            "discount": [0.9,0.95, 0.99],
            "num_updates_per_gradient": [10, 100, 1000]
        }
        # Create a list of hyperparameter configurations from the grid.
        hyperparams_list = []
        for hp_values in itertools.product(
                hyperparams_grid["num_updates_per_gradient"],
                hyperparams_grid["minibatch_size"],
                hyperparams_grid["n_step"],
                hyperparams_grid["discount"],
                hyperparams_grid["lr"]):
            hyperparams_list.append({
                "lr": hp_values[0],
                "n_step": hp_values[1],
                "minibatch_size": hp_values[2],
                "discount": hp_values[3],
                "num_updates_per_gradient": hp_values[4]
            })
        # Parallelize the hyperparameter sweep with Joblib.
        Parallel(n_jobs=5)(
            delayed(run_experiment_and_plot)(hyperparams, env_config, args, seed)
            for hyperparams in hyperparams_list
        )
    else:
        print(f"Running a single experiment with default hyperparameters: {default_hyperparams}")
        run_experiment_for_seed(seed, default_hyperparams, env_config, args)
        produce_plots_for_all_configs()