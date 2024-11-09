import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque

import socket

def run_training_loop(dqn_model, num_episodes, max_steps, batch_size, target_update_freq):
    total_rewards = []

    # Connect to Unity socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("127.0.0.1", 4657))
        
        for episode in range(num_episodes):
            state = dqn_model.reset_environment(s)  # Reset environment within the model
            total_reward = 0
            print("Chal raha hai")

            for step in range(max_steps):
                # Select action based on the current state
                action = dqn_model.act(state)
                
                # Send action to Unity and get new state and reward
                s.sendall(f"{action}\n".encode())
                next_state, done = dqn_model.receive_unity_data(s)
                
                # Calculate reward internally
                reward = dqn_model.calculate_reward(state, next_state, done)
                
                # Store the experience in replay buffer
                dqn_model.remember(state, action, reward, next_state, done)
                
                # Train the model on a sampled batch
                if len(dqn_model.replay_buffer.buffer) >= batch_size:
                    loss = dqn_model.train(batch_size)

                # Update state and accumulated reward
                state = next_state
                total_reward += reward

                # Check if the episode is done
                if done:
                    break

            # Update target network at fixed intervals
            if episode % target_update_freq == 0:
                dqn_model.update_target_network()

            # Decay epsilon after each episode
            dqn_model.epsilon = max(dqn_model.epsilon * dqn_model.epsilon_decay, dqn_model.epsilon_min)
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            s.sendall("\n".encode())

        return total_rewards


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNModel:
    def __init__(self, state_size, action_size, maze, device):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.maze = maze
        self.buffer_size = self.params['BUFFER_SIZE']
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.gamma = self.params['GAMMA']
        self.alpha = self.params['ALPHA']
        self.epsilon = self.params['EPSILON']
        self.epsilon_min = self.params['EPSILON_MIN']
        self.epsilon_decay = self.params['EPSILON_DECAY']

        self.main_network = DQN(self.state_size, self.action_size).to(device=device)
        self.target_network = DQN(self.state_size, self.action_size).to(device=device)
        self.update_target_network()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.alpha)

    def reset_environment(self, socket_conn):
        """Reset Unity environment and get the initial state."""
        socket_conn.sendall("RESET\n".encode())
        initial_state, _ = self.receive_unity_data(socket_conn)
        return initial_state
   

    def calculate_reward(self, state, next_state, done):
        """
        Calculate reward based on state transitions.
        Penalizes collisions with walls, rewards reaching the goal, and adds a small step penalty.
        """
        if done:
            # If the agent reached the goal, reward it; if it hit a wall, penalize it.
            return 10 if next_state == self.maze.destination else -10  # Penalize wall collision
        return -0.1  # Small step penalty to encourage efficient pathfinding

    def receive_unity_data(self, socket_conn):
        """
        Receive data from Unity and interpret it.
        """
        data = socket_conn.recv(2048).decode().strip().split(',')
        agent_x = int(float(data[0]))
        agent_y = int(float(data[1]))
        target_x = int(float(data[2]))
        target_y = int(float(data[3]))
        
        state = (agent_x, agent_y)
        done = bool(int(float(data[4])))  # The done flag from Unity now reflects either goal or wall collision
        return state, done
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = self.encode_state(state).to(self.device)
        with torch.no_grad():
            return self.main_network(state_tensor).argmax().item()

    def train(self, batch_size):
        minibatch = self.replay_buffer.sample(batch_size)
        predicted_Q_values, target_Q_values = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = self.encode_state(state).to(self.device)
            next_state_tensor = self.encode_state(next_state).to(self.device)
            reward_tensor = torch.tensor([reward], device=self.device)

            with torch.no_grad():
                target = reward_tensor + (0 if done else self.gamma * self.target_network(next_state_tensor).max())
            target_Q = self.main_network(state_tensor).clone()
            target_Q[action] = target
            predicted_Q = self.main_network(state_tensor)
            predicted_Q_values.append(predicted_Q)
            target_Q_values.append(target_Q)

        loss = self.loss_fn(torch.stack(predicted_Q_values), torch.stack(target_Q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # def encode_state(self, state):
    #     """Encodes the state to a one-hot tensor representation."""
    #     one_hot_tensor = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)
    #     one_hot_tensor[state] = 1
    #     target_index = self.maze.destination[0] * self.maze.maze_size + self.maze.destination[1]
    #     one_hot_tensor[target_index] = 2
    #     return one_hot_tensor
    def encode_state(self, state):
        """Encodes the state to a one-hot tensor representation."""
        one_hot_tensor = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)
        
        # Convert 2D state coordinates to a 1D index for the one-hot encoding
        row, col = state
        state_index = row * self.maze.maze_size + col
        
        one_hot_tensor[state_index] = 1
        
        # Encode the target (goal) position
        target_index = self.maze.destination[0] * self.maze.maze_size + self.maze.destination[1]
        one_hot_tensor[target_index] = 2
        
        return one_hot_tensor


    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())


class Maze:
    def __init__(self):
        self.maze = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        ]
        self.maze_size = 10
        self.destination = (8, 8)  # Adjust based on your Unity setup


if __name__ == "__main__":
    state_size = 100
    action_size = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maze = Maze()
    dqn_model = DQNModel(state_size, action_size, maze, device)
    num_episodes = 1000
    max_steps = 200
    batch_size = 32
    target_update_freq = 10
    run_training_loop(dqn_model, num_episodes, max_steps, batch_size, target_update_freq)
