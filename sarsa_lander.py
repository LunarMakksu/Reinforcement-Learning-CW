import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# https://crist2201.github.io/project_lunar_lander.html
class LunarLanderSarsa(object):
    def __init__(self,
                 env,
                 n_bins=10,
                 state_low=None,
                 state_high=None,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay=0.9995,
                 seed=None):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((self.env.action_space.n, env.observation_space.shape[0]))

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            try:
                env.reset(seed=seed)
                env.action_space.seed(seed)
            except Exception:
                pass

    def simple_epsilon_greedy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = np.dot(self.q_table, state)
            return np.argmax(q_values)


if __name__ == "__main__":
    num_agents = 50
    num_episodes = 500
    sarsa_rewards = []
    render = True
    epsilon_end = 0.001

    env = gym.make("LunarLander-v3",
                   continuous=False,
                   gravity=-10.0,
                   enable_wind=False,
                   wind_power=15.0,
                   turbulence_power=1.5,
                   render_mode="human" if render else None)

    sarsa_rewards = []
    for i in range(num_agents):
        # print("Agent {}".format(agent + 1))

        agent = LunarLanderSarsa(env=env,
                                 gamma=0.99,
                                 alpha=0.2,
                                 epsilon=0.15,
                                 epsilon_decay=0.9995,
                                 epsilon_min=0.05,
                                 )
        state, _ = env.reset()
        action = agent.simple_epsilon_greedy(state)
        episode_rewards = []
        for episode in range(num_episodes):
            sum_rewards = 0
            next_state, reward, terminal, truncated, _ = env.step(action)

            if terminal or truncated:
                break

            next_action = agent.simple_epsilon_greedy(next_state)
            q_sa = np.dot(agent.q_table[action], state)
            q_s_next_a = np.dot(agent.q_table[next_action], next_state)
            td_error = reward + agent.gamma * q_s_next_a - q_sa

            agent.q_table[action] += agent.alpha * td_error * state

            state = next_state
            action = next_action
            sum_rewards += reward
            agent.epsilon = max(epsilon_end, agent.epsilon * agent.epsilon_decay)
            episode_rewards.append(episode_rewards)
            if episode % 10 == 0:
                print(f"Episode {episode} — Total Reward: {sum_rewards:.2f} — Epsilon: {agent.epsilon:.3f}")
        sarsa_rewards.append(episode_rewards)
