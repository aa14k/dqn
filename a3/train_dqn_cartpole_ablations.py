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


# Adapted from ChatGPT
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


def plot_many_algs(lists, labels, colors, file, ylabel="Return", title="Episodic Returns"):
    running_avgs = []
    for lst in lists:
        running_avgs.append(np.mean(np.array(lst), axis=0))
    plt.figure(figsize=(10, 6))
    for i in range(len(lists)):
        plt.plot(range(len(running_avgs[i])), running_avgs[i], color=colors[i], label=labels[i])
    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "target", "buffer"])
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

    # Ablation study: target network update intervals.
    if args.ablation == "target":
        # For target network ablation, use one-step returns and default buffer size.
        n_step = 1
        buffer_size = 25000
        target_update_intervals = [1, 10, 100]
        perf_dict = {}  # Keys: target update interval, then agent type.
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
        # Plot each algorithm's performance for each target update interval.
        for t_update in target_update_intervals:
            for agent_text in ['DQN', 'DoubleDQN']:
                file_name = f"{agent_text}_target_update_{t_update}_cartpole.png"
                plot_alg_results(perf_dict[t_update][agent_text], file_name, label=f"{agent_text} (update {t_update})")
        for t_update in target_update_intervals:
            file_name = f"dqns_target_update_{t_update}_cartpole.png"
            plot_many_algs([perf_dict[t_update][agent_text] for agent_text in ['DQN', 'DoubleDQN']],
                           ['DQN', 'DoubleDQN'], ['r', 'b'], file_name)

    # Ablation study: replay buffer size.
    elif args.ablation == "buffer":
        # For buffer ablation, use one-step returns and fixed target update interval.
        n_step = 1
        target_update_interval = 100
        buffer_sizes = [100, 500, 5000, 25000]
        perf_dict = {}  # Keys: buffer size, then agent type.
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
        # Plot each algorithm's performance for each buffer size.
        for buf_size in buffer_sizes:
            for agent_text in ['DQN', 'DoubleDQN']:
                file_name = f"{agent_text}_buffer_size_{buf_size}_cartpole.png"
                plot_alg_results(perf_dict[buf_size][agent_text], file_name, label=f"{agent_text} (buffer {buf_size})")
        for buf_size in buffer_sizes:
            file_name = f"dqns_buffer_size_{buf_size}_cartpole.png"
            plot_many_algs([perf_dict[buf_size][agent_text] for agent_text in ['DQN', 'DoubleDQN']],
                           ['DQN', 'DoubleDQN'], ['r', 'b'], file_name)
    

#python train_dqn_cartpole_ablations.py --ablation target
#python train_dqn_cartpole_ablations.py --ablation buffer