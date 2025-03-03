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

CCID="machado"

class LinearDecayEpsilonGreedyExploration:
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: float indicating the value of epsilon
      num_actions: integer indicating the number of actions
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
        self.network = torch.nn.Sequential(nn.Linear(input_size, 64), 
                                           nn.ReLU(),
                                           nn.Linear(64, 64),
                                           nn.ReLU(),
                                           nn.Linear(64, num_actions))

    def forward(self, input):
        return self.network(input)


# Adapted from ChatGPT
def plot_alg_results(episode_returns_list, file, label="Algorithm", ylabel="Return", title="Episodic Returns"):

    # Compute running average
    running_avg = np.mean(np.array(episode_returns_list), axis=0)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original data
    # plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Original Data')

    # Plot the running average
    plt.plot(
        range(0, len(running_avg)),
        running_avg,
        color='r',
        label=label
    )

    # Adding labels and title
    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


def plot_many_algs(lists, labels, colors, file, ylabel="Return", title="Episodic Returns"):
    # Define a function to calculate the running average

    running_avgs = []
    for i in range(len(lists)):
        running_avgs.append(np.mean(np.array(lists[i]), axis=0))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the original data
    # plt.plot(episode_rewards, marker='o', linestyle='-', color='b', label='Original Data')

    for i in range(len(lists)):
        # Plot the running average
        plt.plot(
            range(0, len(running_avgs[i])),
            running_avgs[i],
            color=colors[i],
            label=labels[i],
        )

    # Adding labels and title
    plt.title(f"({CCID}){title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    # Add legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Display the plot
    plt.savefig(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--track-q", action="store_true", default=False)
    parser.add_argument("--num-runs", type=int, default=5)
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
    n_step_colors = ['r', 'b', 'green']
    agent_classes = [dqn.DQN, double_dqn.DoubleDQN]

    perf_dict = {}
    q_val_dict = {}
    for n_step in n_steps:
        perf_dict[n_step] = {}
        q_val_dict[n_step] = {}
        for agent_class in agent_classes:
            agent_text = agent_class_to_text[agent_class]
            alg_returns = []
            alg_q_values = []
            for seed in range(num_seeds):
                env = gym.make("CartPole-v1")
                num_actions = env.action_space.n
                q_network = CartpoleQNetwork(env.observation_space.low.size, num_actions)
                optimizer = torch.optim.Adam(q_network.parameters(), lr=lr, eps=optimizer_eps)
                explorer = LinearDecayEpsilonGreedyExploration(initial_epsilon, final_epsilon, epsilon_decay_steps, num_actions)
                buffer = replay_buffer.ReplayBuffer(buffer_size, discount=discount, n_step=n_step)
                agent = dqn.DQN(q_network, optimizer, buffer, explorer, discount, target_update_interval,
                                min_replay_size_before_updates=min_replay_size_before_updates, minibatch_size=minibatch_size)
                episode_returns, q_values = agent_environment.agent_environment_episode_loop(agent, env, num_training_episodes, args.debug, args.track_q)
                alg_returns.append(episode_returns)
                alg_q_values.append(q_values)

            perf_dict[n_step][agent_text] = alg_returns
            q_val_dict[n_step][agent_text] = alg_q_values

            plot_alg_results(perf_dict[n_step][agent_text], f"{agent_text}_{n_step}_step_cartpole.png", label=agent_text)
            if args.track_q:
                plot_alg_results(q_val_dict[n_step][agent_text], f"{agent_text}_{n_step}_step_cartpole_q_vals.png",
                                 label=agent_text, ylabel="Q-values", title="Q-values")


    for n_step in n_steps:
        plot_many_algs([perf_dict[n_step][agent_text] for agent_text in ['DQN', 'DoubleDQN']],
                       ['DQN', 'Double DQN'], ['r', 'b'], f"dqns_{n_step}_step_cartpole.png",)
        if args.track_q:
            plot_many_algs([q_val_dict[n_step][agent_text] for agent_text in ['DQN', 'DoubleDQN']],
                       ['DQN', 'Double DQN'], ['r', 'b'], f"cartpole_{n_step}_q_vals.png", ylabel="Q-values", title="Q-values")
    
    for agent_class in agent_classes:
        agent_text = agent_class_to_text[agent_class]
        plot_many_algs([perf_dict[n_step][agent_text] for n_step in n_steps],
                       [f"{n_step}-step {agent_text}"for n_step in n_steps], n_step_colors, f"{agent_text}_cartpole.png")
