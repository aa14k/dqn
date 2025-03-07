import gymnasium as gym
import agent_environment
import numpy as np
import epsilon_greedy_explorers as epsilon_greedy_explorers
import dqn
import double_dqn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import replay_buffer
import argparse

CCID="aayoub"

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


class CartpoleQNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.network = torch.nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, input):
        return self.network(input)


def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return", title="Episodic Returns"):
    # Compute running average
    running_avg = np.mean(np.array(episode_returns_list), axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(running_avg)), running_avg, color='r', label=label)
    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(file)
    plt.close()


def plot_many_algs(lists, labels, colors, file, ylabel="Return", title="Episodic Returns"):
    running_avgs = []
    for lst in lists:
        running_avgs.append(np.mean(np.array(lst), axis=0))
    plt.figure(figsize=(10, 6))
    # If no colors provided, use default color cycle.
    for i in range(len(running_avgs)):
        if colors is not None:
            plt.plot(range(len(running_avgs[i])), running_avgs[i], color=colors[i], label=labels[i])
        else:
            plt.plot(range(len(running_avgs[i])), running_avgs[i], label=labels[i])
    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(file)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=5)
    # New option "both" will run target and buffer ablations sequentially.
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "target", "buffer", "both"])
    args = parser.parse_args()

    num_seeds = args.num_runs
    lr = 0.0001
    optimizer_eps = 1e-2
    initial_epsilon = 1.0
    final_epsilon = 0.001
    epsilon_decay_steps = 12500
    discount = 0.99
    min_replay_size_before_updates = 500
    minibatch_size = 128
    num_training_episodes = 500

    agent_class_to_text = {dqn.DQN: 'DQN', double_dqn.DoubleDQN: 'DoubleDQN'}

    if args.ablation == "both":
        print('running experiments for both target and buffer ablations')
        # ----------------------------
        # Target Network Ablation (n_step fixed to 1)
        # ----------------------------
        target_update_intervals = [1, 10, 100]
        perf_dict_target = {}
        for t_update in target_update_intervals:
            perf_dict_target[t_update] = {}
            for agent_class in [dqn.DQN, double_dqn.DoubleDQN]:
                agent_text = agent_class_to_text[agent_class]
                returns_list = []
                for seed in range(42, 42 + num_seeds):
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    env = gym.make("CartPole-v1")
                    num_actions = env.action_space.n
                    q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
                    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
                    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
                    buffer = replay_buffer.ReplayBuffer(25000, discount=discount, n_step=1)
                    agent = agent_class(q_network, optimizer, buffer, explorer, discount, t_update,
                                          min_replay_size_before_updates=min_replay_size_before_updates, minibatch_size=minibatch_size)
                    episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
                    returns_list.append(episode_returns)
                perf_dict_target[t_update][agent_text] = returns_list
                print(f"Finished target update {t_update} for {agent_text}")

        # ----------------------------
        # Replay Buffer Ablation (n_step fixed to 1, target update fixed)
        # ----------------------------
        buffer_sizes = [100, 500, 5000, 25000]
        target_update_interval = 100
        perf_dict_buffer = {}
        for buf_size in buffer_sizes:
            perf_dict_buffer[buf_size] = {}
            for agent_class in [dqn.DQN, double_dqn.DoubleDQN]:
                agent_text = agent_class_to_text[agent_class]
                returns_list = []
                for seed in range(42, 42 + num_seeds):
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    env = gym.make("CartPole-v1")
                    num_actions = env.action_space.n
                    q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
                    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
                    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
                    buffer = replay_buffer.ReplayBuffer(buf_size, discount=discount, n_step=1)
                    agent = agent_class(q_network, optimizer, buffer, explorer, discount, target_update_interval,
                                          min_replay_size_before_updates=min_replay_size_before_updates, minibatch_size=minibatch_size)
                    episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
                    returns_list.append(episode_returns)
                print(f"Finished buffer size {buf_size} for {agent_text}")
                perf_dict_buffer[buf_size][agent_text] = returns_list

        # ----------------------------
        # Plot Combined Target Ablation (all results in one plot)
        # ----------------------------
        combined_target_returns = []
        combined_target_labels = []
        for t_update in target_update_intervals:
            for agent_text in ['DQN', 'DoubleDQN']:
                combined_target_returns.append(perf_dict_target[t_update][agent_text])
                combined_target_labels.append(f"{agent_text} (update {t_update})")
        plot_many_algs(combined_target_returns, combined_target_labels, None, "combined_target_ablation.png", title="Combined Target Network Ablation")

        # Plot per algorithm for target ablation
        for agent_text in ['DQN', 'DoubleDQN']:
            agent_returns = []
            labels = []
            for t_update in target_update_intervals:
                agent_returns.append(perf_dict_target[t_update][agent_text])
                labels.append(f"update {t_update}")
            plot_many_algs(agent_returns, labels, None, f"{agent_text}_target_ablation.png", title=f"{agent_text} Target Network Ablation")

        # ----------------------------
        # Plot Combined Buffer Ablation (all results in one plot)
        # ----------------------------
        combined_buffer_returns = []
        combined_buffer_labels = []
        for buf_size in buffer_sizes:
            for agent_text in ['DQN', 'DoubleDQN']:
                combined_buffer_returns.append(perf_dict_buffer[buf_size][agent_text])
                combined_buffer_labels.append(f"{agent_text} (buffer {buf_size})")
        plot_many_algs(combined_buffer_returns, combined_buffer_labels, None, "combined_buffer_ablation.png", title="Combined Replay Buffer Ablation")

        # Plot per algorithm for buffer ablation
        for agent_text in ['DQN', 'DoubleDQN']:
            agent_returns = []
            labels = []
            for buf_size in buffer_sizes:
                agent_returns.append(perf_dict_buffer[buf_size][agent_text])
                labels.append(f"buffer {buf_size}")
            plot_many_algs(agent_returns, labels, None, f"{agent_text}_buffer_ablation.png", title=f"{agent_text} Replay Buffer Ablation")

        # ----------------------------
        # Plot Everything in One Plot (target and buffer results combined)
        # ----------------------------
        combined_all_returns = combined_target_returns + combined_buffer_returns
        combined_all_labels = combined_target_labels + combined_buffer_labels
        plot_many_algs(combined_all_returns, combined_all_labels, None, "combined_all_ablation.png", title="All Ablation Results Combined")

        # Plot per algorithm for all experiments
        for agent_text in ['DQN', 'DoubleDQN']:
            agent_all_returns = []
            agent_all_labels = []
            # Add target ablation results
            for t_update in target_update_intervals:
                agent_all_returns.append(perf_dict_target[t_update][agent_text])
                agent_all_labels.append(f"target {t_update}")
            # Add buffer ablation results
            for buf_size in buffer_sizes:
                agent_all_returns.append(perf_dict_buffer[buf_size][agent_text])
                agent_all_labels.append(f"buffer {buf_size}")
            plot_many_algs(agent_all_returns, agent_all_labels, None, f"{agent_text}_all_ablation.png", title=f"{agent_text} All Ablation Results")

    elif args.ablation == "target":
        # Target ablation only (n_step=1)
        n_step = 1
        buffer_size = 25000
        target_update_intervals = [1, 10, 100]
        perf_dict = {}
        for t_update in target_update_intervals:
            perf_dict[t_update] = {}
            for agent_class in [dqn.DQN, double_dqn.DoubleDQN]:
                agent_text = agent_class_to_text[agent_class]
                returns_list = []
                for seed in range(42, 42 + num_seeds):
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    env = gym.make("CartPole-v1")
                    num_actions = env.action_space.n
                    q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
                    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
                    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
                    buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)
                    agent = agent_class(q_network, optimizer, buffer, explorer, discount, t_update,
                                          min_replay_size_before_updates=min_replay_size_before_updates, minibatch_size=minibatch_size)
                    episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
                    returns_list.append(episode_returns)
                perf_dict[t_update][agent_text] = returns_list
        # Plot individual and combined target ablation plots.
        for t_update in target_update_intervals:
            for agent_text in ['DQN', 'DoubleDQN']:
                file_name = f"{agent_text}_target_update_{t_update}_cartpole.png"
                plot_alg_results(perf_dict[t_update][agent_text], file_name, label=f"{agent_text} (update {t_update})")
        combined_target_returns = []
        combined_target_labels = []
        for t_update in target_update_intervals:
            for agent_text in ['DQN', 'DoubleDQN']:
                combined_target_returns.append(perf_dict[t_update][agent_text])
                combined_target_labels.append(f"{agent_text} (update {t_update})")
        plot_many_algs(combined_target_returns, combined_target_labels, None, "combined_target_ablation.png", title="Combined Target Network Ablation")
        for agent_text in ['DQN', 'DoubleDQN']:
            agent_returns = []
            labels = []
            for t_update in target_update_intervals:
                agent_returns.append(perf_dict[t_update][agent_text])
                labels.append(f"update {t_update}")
            plot_many_algs(agent_returns, labels, None, f"{agent_text}_target_ablation.png", title=f"{agent_text} Target Network Ablation")
    elif args.ablation == "buffer":
        # Buffer ablation only (n_step=1, fixed target update interval)
        n_step = 1
        target_update_interval = 100
        buffer_sizes = [100, 500, 5000, 25000]
        perf_dict = {}
        for buf_size in buffer_sizes:
            perf_dict[buf_size] = {}
            for agent_class in [dqn.DQN, double_dqn.DoubleDQN]:
                agent_text = agent_class_to_text[agent_class]
                returns_list = []
                for seed in range(42, 42 + num_seeds):
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    env = gym.make("CartPole-v1")
                    num_actions = env.action_space.n
                    q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
                    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
                    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
                    buffer = replay_buffer.ReplayBuffer(buf_size, discount=discount, n_step=n_step)
                    agent = agent_class(q_network, optimizer, buffer, explorer, discount, target_update_interval,
                                          min_replay_size_before_updates=min_replay_size_before_updates, minibatch_size=minibatch_size)
                    episode_returns, _ = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
                    returns_list.append(episode_returns)
                perf_dict[buf_size][agent_text] = returns_list
        for buf_size in buffer_sizes:
            for agent_text in ['DQN', 'DoubleDQN']:
                file_name = f"{agent_text}_buffer_size_{buf_size}_cartpole.png"
                plot_alg_results(perf_dict[buf_size][agent_text], file_name, label=f"{agent_text} (buffer {buf_size})")
        combined_buffer_returns = []
        combined_buffer_labels = []
        for buf_size in buffer_sizes:
            for agent_text in ['DQN', 'DoubleDQN']:
                combined_buffer_returns.append(perf_dict[buf_size][agent_text])
                combined_buffer_labels.append(f"{agent_text} (buffer {buf_size})")
        plot_many_algs(combined_buffer_returns, combined_buffer_labels, None, "combined_buffer_ablation.png", title="Combined Replay Buffer Ablation")
        for agent_text in ['DQN', 'DoubleDQN']:
            agent_returns = []
            labels = []
            for buf_size in buffer_sizes:
                agent_returns.append(perf_dict[buf_size][agent_text])
                labels.append(f"buffer {buf_size}")
            plot_many_algs(agent_returns, labels, None, f"{agent_text}_buffer_ablation.png", title=f"{agent_text} Replay Buffer Ablation")
    else:
        pass