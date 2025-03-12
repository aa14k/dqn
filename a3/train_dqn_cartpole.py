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
#from joblib import Parallel, delayed  # <-- For parallelization

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


class CartpoleQNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.network = nn.Sequential(
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
    for i, avg in enumerate(running_avgs):
        if colors is not None:
            plt.plot(range(len(avg)), avg, color=colors[i], label=labels[i])
        else:
            plt.plot(range(len(avg)), avg, label=labels[i])
    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(file)
    plt.close()


def run_single_experiment(
    seed,
    agent_class,
    n_step,
    lr,
    optimizer_eps,
    initial_epsilon,
    final_epsilon,
    epsilon_decay_steps,
    buffer_size,
    discount,
    target_update_interval,
    min_replay_size_before_updates,
    minibatch_size,
    num_training_episodes,
    debug,
    track_q
):
    """
    Helper function to run a single experiment (single seed).
    Returns a tuple of (episode_returns, q_values).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    num_actions = env.action_space.n

    # Create network, optimizer, explorer
    q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
    buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)

    agent = agent_class(
        q_network,
        optimizer,
        buffer,
        explorer,
        discount,
        target_update_interval,
        min_replay_size_before_updates=min_replay_size_before_updates,
        minibatch_size=minibatch_size
    )

    # Run the training loop
    episode_returns, q_values = agent_environment.agent_environment_episode_loop(
        agent, env, num_training_episodes, debug, track_q
    )

    return episode_returns, q_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=10)
    args = parser.parse_args()

    num_seeds = args.num_runs
    lr = 0.0001
    optimizer_eps = 1e-2
    initial_epsilon = 1.0
    final_epsilon = 0.001
    epsilon_decay_steps = 12500
    buffer_size = 25_000
    discount = 0.99
    target_update_interval = 100
    min_replay_size_before_updates = 500
    minibatch_size = 128
    num_training_episodes = 500

    agent_class_to_text = {dqn.DQN: 'DQN', double_dqn.DoubleDQN: 'DoubleDQN'}

    n_steps = [1, 5, 10]
    seeds = np.arange(num_seeds) + 42
    n_step_colors = ['r', 'b', 'green']
    agent_classes = [dqn.DQN, double_dqn.DoubleDQN]

    # Dictionaries to store performance and Q-values
    perf_dict = {}
    q_val_dict = {}

    for n_step in n_steps:
        perf_dict[n_step] = {}
        q_val_dict[n_step] = {}
        for agent_class in agent_classes:
            agent_text = agent_class_to_text[agent_class]

            # Parallelize across seeds
            results = [run_single_experiment(
                    seed,
                    agent_class,
                    n_step,
                    lr,
                    optimizer_eps,
                    initial_epsilon,
                    final_epsilon,
                    epsilon_decay_steps,
                    buffer_size,
                    discount,
                    target_update_interval,
                    min_replay_size_before_updates,
                    minibatch_size,
                    num_training_episodes,
                    args.debug,
                    args.track_q
                )
                for seed in seeds
            ]

            # Separate the episode returns and q_values
            alg_returns, alg_q_values = zip(*results)
            alg_returns = list(alg_returns)
            alg_q_values = list(alg_q_values)

            print('==================================')
            print(f"Completed {agent_text} with {n_step}-step returns")
            print('==================================')

            perf_dict[n_step][agent_text] = alg_returns
            q_val_dict[n_step][agent_text] = alg_q_values

            # Plot per-alg performance across seeds
            plot_alg_results(
                perf_dict[n_step][agent_text],
                f"{agent_text}_{n_step}_step_cartpole.png",
                label=agent_text
            )
            if args.track_q:
                plot_alg_results(
                    q_val_dict[n_step][agent_text],
                    f"{agent_text}_{n_step}_step_cartpole_q_vals.png",
                    label=agent_text,
                    ylabel="Q-values",
                    title="Q-values"
                )

    # Plot combined results (DQN vs DoubleDQN) for each n_step
    for n_step in n_steps:
        plot_many_algs(
            [perf_dict[n_step][agent_text] for agent_text in ['DQN', 'DoubleDQN']],
            ['DQN', 'Double DQN'],
            ['r', 'b'],
            f"dqns_{n_step}_step_cartpole.png"
        )
        if args.track_q:
            plot_many_algs(
                [q_val_dict[n_step][agent_text] for agent_text in ['DQN', 'DoubleDQN']],
                ['DQN', 'Double DQN'],
                ['r', 'b'],
                f"cartpole_{n_step}_q_vals.png",
                ylabel="Q-values",
                title="Q-values"
            )

    # Plot per-agent results over the different n_steps
    for agent_class in agent_classes:
        agent_text = agent_class_to_text[agent_class]
        plot_many_algs(
            [perf_dict[n_step][agent_text] for n_step in n_steps],
            [f"{n_step}-step {agent_text}" for n_step in n_steps],
            n_step_colors,
            f"{agent_text}_cartpole.png"
        )