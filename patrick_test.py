import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define deep Q-learning model
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Memory for experience replay
# Stores experiences which we randomly sample from to break correlation between consecutive states
class ReplayMemory:
    def __init__(self, maxlen):
        # Deque allows us to automatically remove the oldest entries
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        #  Here we store the tuple: (State, Action, Next_State, Reward, Done)
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# Deep Q-learning
class LunarLanderDQL:
    # hyperparameters
    learning_rate_alpha = 0.0001       # Learning rate alpha
    discount_factor_gamma = 0.99       # Discount rate gamma (0.99 implies we care a lot about the long term goal (landing))
    network_sync_rate = 500            # Number of steps before syncing policy and target
    replay_memory_size = 50000         # Size of replay memory
    mini_batch_size = 128              # How many memories we learn from in one training step

    def train(self, episodes, render=False):
        env = gym.make("LunarLander-v3", render_mode="human" if render else None)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        epsilon = 1.0  # Exploration rate (Starting at 100% random actions)
        memory = ReplayMemory(maxlen=self.replay_memory_size)

        # policy_dqn  : the network we actually train (online network)
        # target_dqn  : frozen copy used to compute stable target values
        # If we only had one network, the network would be updating itself based on its own guesses which causes instability
        policy_dqn = QNetwork(num_states, num_actions)
        target_dqn = QNetwork(num_states, num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict()) # Initially making the two networks identical
        target_dqn.eval()   # Setting to evaluation mode because this network does not train

        optimizer = optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_alpha)
        criterion = nn.SmoothL1Loss() # Huber loss (smoothL1loss) instead of MSE (Huber loss is prefered for RL because it is less sensitive to very large errors which is the case in the beginning.)

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        step_count = 0  # Counting environment steps.
        train_freq = 4  # We don't need to train on every single frame. Training every 4 saves compute.

        for i in range(episodes):
            # Reset environment at start of each episode
            state, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

            while not (terminated or truncated):
                # epsilon-greedy
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    # Convert numpy state to PyTorch tensor
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        q_values = policy_dqn(state_tensor)
                    action = q_values.argmax(1).item() # Pick the action with highest Q-value

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                memory.append((state, action, next_state, reward, done))

                state = next_state
                total_reward += reward

                step_count += 1

                # Training
                if len(memory) >= self.mini_batch_size and (step_count % train_freq == 0):
                    mini_batch = memory.sample(self.mini_batch_size)

                    states = np.array([t[0] for t in mini_batch])
                    actions = np.array([t[1] for t in mini_batch])
                    rewards = np.array([t[3] for t in mini_batch])
                    next_states = np.array([t[2] for t in mini_batch])
                    dones = np.array([t[4] for t in mini_batch], dtype=np.float32)

                    states_tensor = torch.tensor(states, dtype=torch.float32)
                    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
                    actions_tensor = torch.tensor(actions, dtype=torch.long)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32)


                    with torch.no_grad():
                        # Policy net selects the best action (Double DQN)
                        next_q_policy = policy_dqn(next_states_tensor)
                        best_next_actions = next_q_policy.argmax(1)

                        # Target net evaluates that action (unbiased)
                        q_next_target = target_dqn(next_states_tensor)
                        max_q_next = q_next_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

                    target_q = rewards_tensor + (1.0 - dones_tensor) * self.discount_factor_gamma * max_q_next

                    # Current Q values (policy network)
                    q_values = policy_dqn(states_tensor)  # (batch, n_actions)
                    q_action = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # (batch,)

                    # loss and update
                    loss = criterion(q_action, target_q)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10.0)  # Prevents explosions
                    optimizer.step()

                # Sync target network
                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode[i] = total_reward

            # Decay epsilon
            epsilon = max(epsilon * 0.995, 0.01)
            epsilon_history.append(epsilon)

            if (i + 1) % 10 == 0:
                avg_reward = np.mean(rewards_per_episode[max(0, i - 9):i + 1])
                print(f"Episode {i + 1}/{episodes} | Total Reward: {total_reward:.2f} | "
                      f"Average Reward (last 10): {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

        env.close()
        torch.save(policy_dqn.state_dict(), "lunar_lander_dqn_weights.pt")

        # Plotting
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode, label='Total Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epsilon_history, label='Epsilon', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("lunar_lander_rewards_and_epsilon.png")
        plt.show()

    def test(self, episodes=10):
        env = gym.make("LunarLander-v3", render_mode="human")
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # no target network during test. Using saved weights from tarining
        policy_dqn = QNetwork(num_states, num_actions)
        policy_dqn.load_state_dict(torch.load("lunar_lander_dqn_weights.pt"))
        policy_dqn.eval()

        rewards = []

        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = policy_dqn(state_tensor)
                action = q_values.argmax(1).item()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state

            rewards.append(total_reward)
            print(f"Episode {episode+1}: Total Reward = {total_reward}")

        env.close()

        print(f"Average reward over {episodes} episodes: {np.mean(rewards)}")
        print(f"Max reward: {np.max(rewards)}, Min reward: {np.min(rewards)}")


def random_agent(episodes=20):
    env = gym.make("LunarLander-v3", render_mode="human")
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    env.close()

    print("\n--- Summary Random agent ---")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")



if __name__ == "__main__":
    agent = LunarLanderDQL()
    agent.train(episodes=150)
    agent.test(episodes=10)
    random_agent(episodes=20)