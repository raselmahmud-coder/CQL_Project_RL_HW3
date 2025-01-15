import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# Hyperparameters
# =========================
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
TAU = 0.01
LR_Q = 3e-4
BATCH_SIZE = 64
MEMORY_SIZE = 100000
MAX_EPISODES = 400
MAX_STEPS = 500
SEED = 42

# CQL specific parameters
CQL_ALPHA = 1.0  # Coefficient for the conservative Q-learning term
MIN_Q_WEIGHT = 10.0  # Weight for the minimum Q value term

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create environment
env = gym.make(ENV_NAME)
env.reset(seed=SEED)
env.action_space.seed(SEED)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def load(self, filename):
        """Load dataset from file"""
        data = np.load(filename)
        for s, a, r, ns, d in zip(data['states'], data['actions'], 
                                 data['rewards'], data['next_states'], 
                                 data['dones']):
            self.push(s, a, r, ns, d)

    def __len__(self):
        return len(self.buffer)

class CQLAgent:
    def __init__(self, obs_dim, act_dim):
        self.q1 = QNetwork(obs_dim, act_dim)
        self.q2 = QNetwork(obs_dim, act_dim)
        self.q1_target = QNetwork(obs_dim, act_dim)
        self.q2_target = QNetwork(obs_dim, act_dim)

        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LR_Q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LR_Q)

    def to(self, device):
        self.q1.to(device)
        self.q2.to(device)
        self.q1_target.to(device)
        self.q2_target.to(device)

    def train_step(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute CQL loss
        with torch.no_grad():
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = torch.max(next_q, dim=1, keepdim=True)[0]
            target_q = rewards + GAMMA * (1 - dones) * next_v

        # Current Q values
        current_q1 = self.q1(states)
        current_q2 = self.q2(states)
        
        # Standard TD error
        q1_loss = nn.MSELoss()(current_q1.gather(1, actions.unsqueeze(1)), target_q)
        q2_loss = nn.MSELoss()(current_q2.gather(1, actions.unsqueeze(1)), target_q)

        # CQL loss
        q1_cql = (torch.logsumexp(current_q1, dim=1, keepdim=True) - 
                  current_q1.gather(1, actions.unsqueeze(1)))
        q2_cql = (torch.logsumexp(current_q2, dim=1, keepdim=True) - 
                  current_q2.gather(1, actions.unsqueeze(1)))

        # Total loss
        total_q1_loss = q1_loss + CQL_ALPHA * q1_cql.mean()
        total_q2_loss = q2_loss + CQL_ALPHA * q2_cql.mean()

        # Update Q networks
        self.q1_optimizer.zero_grad()
        total_q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        total_q2_loss.backward()
        self.q2_optimizer.step()

        # Soft update target networks
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

    def _soft_update(self, net, target_net, tau=TAU):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def get_action(self, state, deterministic=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q1_values = self.q1(state)
            if deterministic:
                action = q1_values.argmax(dim=1).item()
            else:
                probs = torch.softmax(q1_values, dim=1)
                action = torch.multinomial(probs, 1).item()
            return action

def evaluate_policy(agent, n_episodes=5):
    eval_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < MAX_STEPS:
            action = agent.get_action(state, deterministic=True)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
            
        eval_rewards.append(episode_reward)
    return np.mean(eval_rewards)

def main():
    # Initialize agent
    agent = CQLAgent(obs_dim, act_dim)
    
    # Initialize replay buffer and load offline dataset
    replay_buffer = ReplayBuffer()
    
    # Load the specified dataset
    dataset_path = "dataset_episode_350.npz"
    try:
        replay_buffer.load(dataset_path)
        print(f"Loaded dataset from {dataset_path}")
        print(f"Dataset size: {len(replay_buffer)}")
    except FileNotFoundError:
        print(f"Dataset file {dataset_path} not found!")
        return

    # Training loop
    print("Starting offline training with CQL...")
    total_steps = 0
    eval_interval = 1000  # Evaluate every 1000 steps
    
    for episode in range(MAX_EPISODES):
        # Training steps
        for _ in range(MAX_STEPS):
            total_steps += 1
            agent.train_step(replay_buffer)
            
            # Periodic evaluation
            if total_steps % eval_interval == 0:
                eval_reward = evaluate_policy(agent, n_episodes=5)
                print(f"Step: {total_steps}, Evaluation reward: {eval_reward:.2f}")
        
        # Episode end evaluation
        if (episode + 1) % 10 == 0:
            eval_reward = evaluate_policy(agent, n_episodes=5)
            print(f"Episode: {episode+1}, Average Reward: {eval_reward:.2f}")
    
    # Final evaluation
    final_reward = evaluate_policy(agent, n_episodes=10)
    print(f"\nTraining completed!")
    print(f"Final evaluation reward (10 episodes): {final_reward:.2f}")

if __name__ == "__main__":
    main()