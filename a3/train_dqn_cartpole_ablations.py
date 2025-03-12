import gymnasium as gym
import agent_environment
import numpy as np
import epsilon_greedy_explorers
import dqn
import double_dqn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import replay_buffer
import argparse
#from joblib import Parallel, delayed

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
    # Compute running average (here simply the mean across runs)
    running_avg = np.mean(np.array(episode_returns_list), axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(running_avg)), running_avg, color='r', label=label)
    plt.title(f"({CCID}) {title}")
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
    for i in range(len(running_avgs)):
        if colors is not None:
            plt.plot(range(len(running_avgs[i])), running_avgs[i], color=colors[i], label=labels[i])
        else:
            plt.plot(range(len(running_avgs[i])), running_avgs[i], label=labels[i])
    plt.title(f"({CCID}) {title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(file)
    plt.close()


def run_single_cartpole_experiment(
    seed,
    agent_class,
    buffer_size,
    target_update_interval,
    n_step,
    lr,
    optimizer_eps,
    initial_epsilon,
    final_epsilon,
    epsilon_decay_steps,
    discount,
    min_replay_size_before_updates,
    minibatch_size,
    num_training_episodes,
    debug=False,
    track_q=False
):
    """
    Helper function to run a single Cartpole experiment with the given hyperparameters.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    num_actions = env.action_space.n

    q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
    explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
    replay_buf = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)

    agent = agent_class(
        q_network,
        optimizer,
        replay_buf,
        explorer,
        discount,
        target_update_interval,
        min_replay_size_before_updates=min_replay_size_before_updates,
        minibatch_size=minibatch_size
    )

    episode_returns, _ = agent_environment.agent_environment_episode_loop(
        agent, env, num_training_episodes, debug, track_q
    )
    return episode_returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--ablation", type=str, default="target", choices=["target", "buffer"])
    args = parser.parse_args()

    # Common hyperparameters.
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

    agent_class_to_text = {dqn.DQN: 'DQN'}

    if args.ablation == "target":
        # --------------------------------------------------------------------
        # Experiment 1: Target Network Ablation Study
        #
        # For n=1, we test three target network update frequencies:
        #   - update every 1 step
        #   - update every 10 steps
        #   - update every 100 steps
        # --------------------------------------------------------------------
        n_step = 1
        buffer_size = 25000
        target_update_intervals = [1, 10, 100]
        perf_dict = {}

        for t_update in target_update_intervals:
            perf_dict[t_update] = {}
            for agent_class in [dqn.DQN]:
                agent_text = agent_class_to_text[agent_class]

                # Run experiments in parallel across seeds
                results = [run_single_cartpole_experiment(
                        seed=seed,
                        agent_class=agent_class,
                        buffer_size=buffer_size,
                        target_update_interval=t_update,
                        n_step=n_step,
                        lr=lr,
                        optimizer_eps=optimizer_eps,
                        initial_epsilon=initial_epsilon,
                        final_epsilon=final_epsilon,
                        epsilon_decay_steps=epsilon_decay_steps,
                        discount=discount,
                        min_replay_size_before_updates=min_replay_size_before_updates,
                        minibatch_size=minibatch_size,
                        num_training_episodes=num_training_episodes,
                        debug=args.debug,
                        track_q=args.track_q
                    )
                    for seed in range(42, 42 + num_seeds)
                ]

                # Store results
                perf_dict[t_update][agent_text] = results

        # Combine results from both DQN and DoubleDQN for the three update intervals into one plot.
        combined_returns = []
        combined_labels = []
        for t_update in target_update_intervals:
            for agent_text in ['DQN']:
                combined_returns.append(perf_dict[t_update][agent_text])
                combined_labels.append(f"{agent_text} (target update {t_update})")

        plot_many_algs(
            combined_returns,
            combined_labels,
            None,
            "target_network_ablation.png",
            ylabel="Return",
            title="Target Network Ablation Study on CartPole"
        )
        print("Target network ablation plot saved to target_network_ablation.png")

    elif args.ablation == "buffer":
        # --------------------------------------------------------------------
        # Experiment 2: Replay Buffer Ablation Study
        #
        # For n=1 and a fixed target update interval (100), we test different
        # replay buffer sizes: 100, 500, 5000, and 25000.
        # --------------------------------------------------------------------
        n_step = 1
        target_update_interval = 100
        buffer_sizes = [100, 500, 5000, 25000]
        perf_dict = {}

        for buf_size in buffer_sizes:
            perf_dict[buf_size] = {}
            for agent_class in [dqn.DQN]:
                agent_text = agent_class_to_text[agent_class]

                # Parallelize across seeds
                results = [run_single_cartpole_experiment(
                        seed=seed,
                        agent_class=agent_class,
                        buffer_size=buf_size,
                        target_update_interval=target_update_interval,
                        n_step=n_step,
                        lr=lr,
                        optimizer_eps=optimizer_eps,
                        initial_epsilon=initial_epsilon,
                        final_epsilon=final_epsilon,
                        epsilon_decay_steps=epsilon_decay_steps,
                        discount=discount,
                        min_replay_size_before_updates=min_replay_size_before_updates,
                        minibatch_size=minibatch_size,
                        num_training_episodes=num_training_episodes,
                        debug=args.debug,
                        track_q=args.track_q
                    )
                    for seed in range(42, 42 + num_seeds)
                ]

                # Store results
                perf_dict[buf_size][agent_text] = results

        # Combine results from both DQN and DoubleDQN for all tested buffer sizes into one plot.
        combined_returns = []
        combined_labels = []
        for buf_size in buffer_sizes:
            for agent_text in ['DQN']:
                combined_returns.append(perf_dict[buf_size][agent_text])
                combined_labels.append(f"{agent_text} (buffer size {buf_size})")

        plot_many_algs(
            combined_returns,
            combined_labels,
            None,
            "replay_buffer_ablation.png",
            ylabel="Return",
            title="Replay Buffer Ablation Study on CartPole"
        )
        print("Replay buffer ablation plot saved to replay_buffer_ablation.png")