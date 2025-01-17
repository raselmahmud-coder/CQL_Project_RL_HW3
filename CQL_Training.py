# CQL_Training.py

import os
from pathlib import Path
import random
import argparse
import gymnasium as gym
from matplotlib.pylab import f
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

# =========================
# Hyperparameters
# =========================
ENV_NAME = "CartPole-v1"
GAMMA = 0.99            # Discount factor
TAU = 0.005             # Soft update coefficient
LR_Q = 3e-4             # Q-network learning rate
LR_POLICY = 3e-4        # Policy network learning rate
ALPHA = 0.2             # Temperature parameter for policy entropy
BATCH_SIZE = 64         # Batch size
MAX_EPISODES = 400      # Number of training episodes
MAX_STEPS = 500         # Max steps per episode
SEED = 42               # Random seed
# CQL specific parameters
CQL_ALPHA = 0.5         # Coefficient for the conservative Q-learning term
MIN_Q_WEIGHT = 5.0     # Weight for the minimum Q value term

# Dataset paths
DATASET_PATHS = {
    '50': 'datasets/dataset_episode_50.npz',
    '150': 'datasets/dataset_episode_150.npz',
    '250': 'datasets/dataset_episode_250.npz',
    '350': 'datasets/dataset_episode_350.npz'
}

# =========================
# Set Random Seeds for Reproducibility
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Define Networks
# =========================

class PolicyNetwork(nn.Module):
    """
    Policy Network: Outputs log probabilities for each discrete action.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs):
        """
        Returns logits (unnormalized log probabilities).
        """
        return self.net(obs)

    def get_action(self, obs, deterministic=False):
        """
        Given a single state, return a discrete action.
        If deterministic=True, choose the action with the highest probability.
        Otherwise, sample an action based on the probability distribution.
        """
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, 1)
        return action.item()

    def get_log_probs(self, obs):
        """
        Given a batch of states, return log_probs and probs for each action.
        log_probs: [batch_size, act_dim]
        probs:     [batch_size, act_dim]
        """
        logits = self.forward(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return log_probs, probs


class QNetwork(nn.Module):
    """
    Q Network: Predicts Q(s, a) for each action.
    """
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
        """
        Returns Q-values for each action [batch_size, act_dim]
        """
        return self.net(obs)

# =========================
# Define Offline Replay Buffer
# =========================

class OfflineReplayBuffer:
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
        data = np.load(dataset_path)
        self.states = data['states']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.next_states = data['next_states']
        self.dones = data['dones']
        self.size = len(self.states)
        print(f"Loaded dataset from {dataset_path} with {self.size} samples.")

    def sample(self, batch_size=BATCH_SIZE):
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices]).unsqueeze(-1)
        rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(-1)
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices]).unsqueeze(-1)
        return states, actions, rewards, next_states, dones

# =========================
# Utility Functions
# =========================

def soft_update(net, target_net, tau=TAU):
    """
    Soft update: target_net = tau * net + (1 - tau) * target_net
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def evaluate_policy(env, policy, device, n_episodes=5, record=False, video_folder="videos", episode_prefix="eval"):
    """
    Evaluate the current policy: use deterministic actions during evaluation.
    Returns the average total reward and collects episode lengths and rewards.
    """
    eval_rewards = []
    eval_lengths = []
    for i in range(n_episodes):
        if record:
            env = RecordVideo(
                env, 
                video_folder=video_folder, 
                name_prefix=f"{episode_prefix}_{i+1}",
                episode_trigger=lambda episode: True  # Record every episode
            )
        state, _ = env.reset(seed=SEED)
        done = False
        episode_reward = 0
        episode_length = 0

        while not done and episode_reward < MAX_STEPS:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = policy.get_action(state_tensor, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            state = next_state

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)

        if record:
            env.close()

    avg_reward = np.mean(eval_rewards)
    avg_length = np.mean(eval_lengths)
    print(f"Evaluation over {n_episodes} episodes: Average Reward = {avg_reward:.2f}, Average Length = {avg_length:.2f}")
    return avg_reward, avg_length

# =========================
# CQL Training Step
# =========================

def train_step(q1, q2, q1_target, q2_target, policy, replay_buffer, 
              q1_optimizer, q2_optimizer, policy_optimizer, device, act_dim):
    """
    Sample a batch from the replay buffer and perform a training step with CQL.
    Returns the losses for logging.
    """
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # --------------------------
    # 1) Compute Target Q-values
    # --------------------------
    with torch.no_grad():
        next_log_probs, next_probs = policy.get_log_probs(next_states)
        q1_next = q1_target(next_states)
        q2_next = q2_target(next_states)
        min_q_next = torch.min(q1_next, q2_next)
        V_next = (next_probs * (min_q_next - ALPHA * next_log_probs)).sum(dim=-1, keepdim=True)
        target_q = rewards + GAMMA * (1 - dones) * V_next

    # --------------------------
    # 2) Compute CQL Loss for Q1
    # --------------------------
    # For discrete actions, compute Q-values for all possible actions
    batch_size = states.size(0)
    expanded_states = states.unsqueeze(1).repeat(1, act_dim, 1).view(-1, states.size(1))  # [batch_size * act_dim, obs_dim]
    all_actions = torch.arange(act_dim).unsqueeze(0).repeat(batch_size, 1).view(-1).to(device)  # [batch_size * act_dim]

    # Compute Q-values for all actions using Q1 network
    q1_all = q1(expanded_states).gather(1, all_actions.unsqueeze(1)).view(batch_size, act_dim)  # [batch_size, act_dim]
    min_q1_all = torch.min(q1_all, dim=1, keepdim=True).values  # [batch_size, 1]

    # Q-data for Q1
    q1_data = q1(states).gather(1, actions)  # [batch_size, 1]

    # CQL loss for Q1
    cql_loss_q1 = CQL_ALPHA * (q1_all.mean(dim=1, keepdim=True) - q1_data).mean()

    # --------------------------
    # 3) Compute CQL Loss for Q2
    # --------------------------
    # Compute Q-values for all actions using Q2 network
    q2_all = q2(expanded_states).gather(1, all_actions.unsqueeze(1)).view(batch_size, act_dim)  # [batch_size, act_dim]
    min_q2_all = torch.min(q2_all, dim=1, keepdim=True).values  # [batch_size, 1]

    # Q-data for Q2
    q2_data = q2(states).gather(1, actions)  # [batch_size, 1]

    # CQL loss for Q2
    cql_loss_q2 = CQL_ALPHA * (q2_all.mean(dim=1, keepdim=True) - q2_data).mean()

    # --------------------------
    # 4) Compute Q1 and Q2 Losses
    # --------------------------
    q1_pred = q1(states).gather(1, actions)  # [batch_size, 1]
    q2_pred = q2(states).gather(1, actions)  # [batch_size, 1]

    q1_loss = nn.MSELoss()(q1_pred, target_q)
    q2_loss = nn.MSELoss()(q2_pred, target_q)

    # Total Q Loss with CQL penalty
    total_q1_loss = q1_loss + MIN_Q_WEIGHT * cql_loss_q1
    total_q2_loss = q2_loss + MIN_Q_WEIGHT * cql_loss_q2

    # --------------------------
    # 5) Update Q1 and Q2 Networks
    # --------------------------
    q1_optimizer.zero_grad()
    total_q1_loss.backward()
    q1_optimizer.step()

    q2_optimizer.zero_grad()
    total_q2_loss.backward()
    q2_optimizer.step()

    # --------------------------
    # 6) Update Policy Network
    # --------------------------
    log_probs, probs = policy.get_log_probs(states)
    q1_vals = q1(states)
    q2_vals = q2(states)
    min_q = torch.min(q1_vals, q2_vals)  # [batch_size, act_dim]

    # Policy loss: E_{s ~ D}[ E_{a ~ pi}[ alpha * log pi(a|s) - Q(s,a) ] ]
    policy_loss = (probs * (ALPHA * log_probs - min_q)).sum(dim=1).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # --------------------------
    # 7) Soft Update Target Networks
    # --------------------------
    soft_update(q1, q1_target, TAU)
    soft_update(q2, q2_target, TAU)

    return q1_loss.item(), q2_loss.item(), cql_loss_q1.item(), cql_loss_q2.item(), policy_loss.item()

# =========================
# Main Training Loop
# =========================

def main(dataset_choice):
    """
    Main function to train the CQL agent on a chosen dataset.
    """
    # Verify dataset choice
    if dataset_choice not in DATASET_PATHS:
        raise ValueError(f"Dataset choice '{dataset_choice}' is invalid. Choose from {list(DATASET_PATHS.keys())}.")

    dataset_path = DATASET_PATHS[dataset_choice]
    print(f"Training CQL on dataset: {dataset_path}")

    # Initialize Replay Buffer
    replay_buffer = OfflineReplayBuffer(dataset_path)

    # Initialize environment for evaluation with RecordEpisodeStatistics
    eval_env = gym.make(ENV_NAME, render_mode='rgb_array')
    eval_env = RecordEpisodeStatistics(eval_env)
    eval_env.reset(seed=SEED)
    eval_env.action_space.seed(SEED)

    # Determine action dimension from the environment
    act_dim = eval_env.action_space.n

    # Initialize networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = QNetwork(obs_dim=4, act_dim=act_dim).to(device)
    q2 = QNetwork(obs_dim=4, act_dim=act_dim).to(device)
    q1_target = QNetwork(obs_dim=4, act_dim=act_dim).to(device)
    q2_target = QNetwork(obs_dim=4, act_dim=act_dim).to(device)
    policy = PolicyNetwork(obs_dim=4, act_dim=act_dim).to(device)

    # Copy parameters to target networks
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # Optimizers
    q1_optimizer = optim.Adam(q1.parameters(), lr=LR_Q)
    q2_optimizer = optim.Adam(q2.parameters(), lr=LR_Q)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LR_POLICY)

    # Lists to store training metrics
    q1_losses = []
    q2_losses = []
    cql_losses_q1 = []
    cql_losses_q2 = []
    policy_losses = []
    eval_avg_rewards = []
    eval_avg_lengths = []

    # Directory for saving videos
    video_dir = Path("CQL_ALPHA=0.5_videos")
    dataset_choice_str = str(dataset_choice)
    video_save_path = video_dir/f"{dataset_choice_str}_dataset_episode_record"
    os.makedirs(video_save_path, exist_ok=True)

    for episode in range(1, MAX_EPISODES + 1):
        # Perform a training step
        q1_loss, q2_loss, cql_loss_q1, cql_loss_q2, policy_loss = train_step(
            q1, q2, q1_target, q2_target, policy, replay_buffer, 
            q1_optimizer, q2_optimizer, policy_optimizer, device, act_dim
        )

        # Log losses
        q1_losses.append(q1_loss)
        q2_losses.append(q2_loss)
        cql_losses_q1.append(cql_loss_q1)
        cql_losses_q2.append(cql_loss_q2)
        policy_losses.append(policy_loss)

        # Evaluation every 10 episodes
        if episode % 10 == 0:
            avg_reward, avg_length = evaluate_policy(
                eval_env, policy, device, n_episodes=5, 
                record=False
            )
            eval_avg_rewards.append(avg_reward)
            eval_avg_lengths.append(avg_length)
            print(f"Episode {episode}/{MAX_EPISODES} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Q1 Loss: {q1_loss:.4f} | Q2 Loss: {q2_loss:.4f} | CQL Loss Q1: {cql_loss_q1:.4f} | CQL Loss Q2: {cql_loss_q2:.4f} | Policy Loss: {policy_loss:.4f}")

        # Record video every 100 episodes
        if episode % 100 == 0:
            print(f"Recording Episode {episode}...")
            avg_reward, avg_length = evaluate_policy(
                eval_env, policy, device, n_episodes=1, 
                record=True, video_folder=video_save_path, 
                episode_prefix=f"episode_{episode}"
            )

    # =========================
    # Plotting Training Metrics
    # =========================
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    plt.plot(q1_losses, label='Q1 Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Q1 Network Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(q2_losses, label='Q2 Loss', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Q2 Network Loss')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(cql_losses_q1, label='CQL Loss Q1', color='green')
    plt.plot(cql_losses_q2, label='CQL Loss Q2', color='red')
    plt.xlabel('Training Steps')
    plt.ylabel('CQL Loss')
    plt.title('CQL Losses for Q1 and Q2')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(policy_losses, label='Policy Loss', color='purple')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Policy Network Loss')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(eval_avg_rewards, label='Average Reward', color='brown')
    plt.xlabel('Evaluation Episodes (x10)')
    plt.ylabel('Average Reward')
    plt.title('Policy Performance')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(eval_avg_lengths, label='Average Episode Length', color='blue')
    plt.xlabel('Evaluation Episodes (x10)')
    plt.ylabel('Average Length')
    plt.title('Episode Lengths')
    plt.legend()

    plt.tight_layout()
    dataset_choice_str = str(dataset_choice)
    save_dir = Path("CQL_ALPHA=0.5_visual_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create the file path dynamically
    fig_save_path = save_dir / f"{dataset_choice_str}_training_metrics.png"
    plt.savefig(fig_save_path)


    # Final Evaluation
    print("Final Evaluation:")
    final_reward, final_length = evaluate_policy(
        eval_env, policy, device, n_episodes=10, 
        record=False
    )
    print(f"Final Evaluation over 10 episodes: Average Reward = {final_reward:.2f}, Average Length = {final_length:.2f}")

    # Close environments
    eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CQL on CartPole-v1 using Offline RL.")
    parser.add_argument('--dataset', type=str, default='350', choices=['50', '150', '250', '350'],
                        help='Choose which dataset to train on: 50, 150, 250, 350')
    args = parser.parse_args()
    main(dataset_choice=args.dataset)
