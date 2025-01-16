import os
from pathlib import Path
import random
import argparse
import gymnasium as gym
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
BATCH_SIZE = 64         # Batch size
MAX_EPISODES = 400      # Number of training episodes
MAX_STEPS = 500         # Max steps per episode
SEED = 42               # Random seed

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
# Policy Network
# =========================


class PolicyNetwork(nn.Module):
    """
    Policy Network: Predicts actions based on states.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        """
        Returns action probabilities for each action [batch_size, act_dim]
        """
        return self.net(obs)

    def get_action(self, obs, deterministic=False):
        """
        Returns an action based on the state observation.
        If deterministic, returns the action with the highest probability.
        """
        action_probs = self.forward(obs)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = torch.multinomial(action_probs, num_samples=1)
        return action.cpu().numpy().flatten()

    def get_log_probs(self, obs):
        """
        Returns log probabilities and probabilities of actions.
        """
        action_probs = self.forward(obs)
        log_probs = torch.log(action_probs + 1e-8)
        return log_probs, action_probs

# =========================
# Offline Replay Buffer
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
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data)


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
            action = action.item()
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
    print(
        f"Evaluation over {n_episodes} episodes: Average Reward = {avg_reward:.2f}, Average Length = {avg_length:.2f}")
    return avg_reward, avg_length

# =========================
# Standard Q-learning Training Step
# =========================


def train_step_standard(q1, q1_target, policy, replay_buffer,
                        q1_optimizer, policy_optimizer, device, act_dim):
    """
    Sample a batch from the replay buffer and perform a training step for Standard Q-learning.
    """
    states, actions, rewards, next_states, dones = replay_buffer.sample(
        BATCH_SIZE)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # --------------------------
    # 1) Compute Target Q-values
    # --------------------------
    with torch.no_grad():
        q1_next = q1_target(next_states)
        target_q = rewards + GAMMA * \
            (1 - dones) * torch.max(q1_next, dim=-1, keepdim=True)[0]

    # --------------------------
    # 2) Compute Q Loss
    # --------------------------
    q1_pred = q1(states).gather(1, actions)  # [batch_size, 1]
    q1_loss = nn.MSELoss()(q1_pred, target_q)

    # --------------------------
    # 3) Update Q1 Network
    # --------------------------
    q1_optimizer.zero_grad()
    q1_loss.backward()
    q1_optimizer.step()

    # --------------------------
    # 4) Update Policy Network
    # --------------------------
    log_probs, probs = policy.get_log_probs(states)
    q1_vals = q1(states)
    policy_loss = (probs * (log_probs - torch.max(q1_vals,
                   dim=-1, keepdim=True)[0])).sum(dim=1).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # --------------------------
    # 5) Soft Update Target Networks
    # --------------------------
    soft_update(q1, q1_target, TAU)

    return q1_loss.item(), policy_loss.item()

# =========================
# Main Training Loop for Standard Q-learning
# =========================


def main_standard(dataset_choice):
    """
    Main function to train the Standard Q-learning agent on a chosen dataset.
    """
    # Verify dataset choice
    if dataset_choice not in DATASET_PATHS:
        raise ValueError(
            f"Dataset choice '{dataset_choice}' is invalid. Choose from {list(DATASET_PATHS.keys())}.")

    dataset_path = DATASET_PATHS[dataset_choice]
    print(f"Training Standard Q-learning on dataset: {dataset_path}")

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
    q1_target = QNetwork(obs_dim=4, act_dim=act_dim).to(device)
    policy = PolicyNetwork(obs_dim=4, act_dim=act_dim).to(device)

    # Copy parameters to target networks
    q1_target.load_state_dict(q1.state_dict())

    # Optimizers
    q1_optimizer = optim.Adam(q1.parameters(), lr=LR_Q)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LR_POLICY)

    # Lists to store training metrics
    q1_losses = []
    policy_losses = []
    eval_avg_rewards = []
    eval_avg_lengths = []

    # Directory for saving videos
    video_dir = Path("Standard_Q_Learning_videos")
    video_save_path = video_dir/f"{dataset_choice}_dataset_episode_record"
    os.makedirs(video_save_path, exist_ok=True)

    for episode in range(1, MAX_EPISODES + 1):
        # Perform a training step
        q1_loss, policy_loss = train_step_standard(
            q1, q1_target, policy, replay_buffer,
            q1_optimizer, policy_optimizer, device, act_dim
        )
# =========================
# Standard Q-Learning
# =========================


class StandardQNetwork(nn.Module):
    """
    A standard Q-Network without the conservative Q-learning (CQL) loss.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(StandardQNetwork, self).__init__()
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
# Standard Q-Learning Training Step
# =========================


def standard_q_train_step(q, q_target, policy, replay_buffer,
                          q_optimizer, device, act_dim):
    """
    Sample a batch from the replay buffer and perform a training step using standard Q-learning.
    Returns the Q-loss for logging.
    """
    states, actions, rewards, next_states, dones = replay_buffer.sample(
        BATCH_SIZE)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # --------------------------
    # 1) Compute Target Q-values (Standard Q-Learning)
    # --------------------------
    with torch.no_grad():
        target_q = rewards + GAMMA * \
            (1 - dones) * torch.max(q_target(next_states), dim=-1, keepdim=True).values

    # --------------------------
    # 2) Compute Q Loss (Standard Q-Learning)
    # --------------------------
    q_pred = q(states).gather(1, actions)  # [batch_size, 1]
    q_loss = nn.MSELoss()(q_pred, target_q)

    # --------------------------
    # 3) Update Q Network
    # --------------------------
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    return q_loss.item()

# =========================
# Main Standard Q-Learning Loop
# =========================


def standard_q_main(dataset_choice):
    """
    Main function to train the standard Q-learning agent on a chosen dataset.
    """
    # Verify dataset choice
    if dataset_choice not in DATASET_PATHS:
        raise ValueError(
            f"Dataset choice '{dataset_choice}' is invalid. Choose from {list(DATASET_PATHS.keys())}.")

    dataset_path = DATASET_PATHS[dataset_choice]
    print(f"Training Standard Q-Learning on dataset: {dataset_path}")

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
    q = StandardQNetwork(obs_dim=4, act_dim=act_dim).to(device)
    q_target = StandardQNetwork(obs_dim=4, act_dim=act_dim).to(device)
    policy = PolicyNetwork(obs_dim=4, act_dim=act_dim).to(device)

    # Copy parameters to target network
    q_target.load_state_dict(q.state_dict())

    # Optimizer for Q network
    q_optimizer = optim.Adam(q.parameters(), lr=LR_Q)

    # Lists to store training metrics
    q_losses = []
    eval_avg_rewards = []
    eval_avg_lengths = []

    # Directory for saving videos
    video_dir = Path("Standard_Q_Learning_videos")
    dataset_choice_str = str(dataset_choice)
    video_save_path = video_dir / \
        f"{dataset_choice_str}_dataset_episode_record"
    os.makedirs(video_save_path, exist_ok=True)

    for episode in range(1, MAX_EPISODES + 1):
        # Perform a training step
        q_loss = standard_q_train_step(
            q, q_target, policy, replay_buffer,
            q_optimizer, device, act_dim
        )

        # Log losses
        q_losses.append(q_loss)

        # Evaluation every 10 episodes
        if episode % 10 == 0:
            avg_reward, avg_length = evaluate_policy(
                eval_env, policy, device, n_episodes=5,
                record=False
            )
            eval_avg_rewards.append(avg_reward)
            eval_avg_lengths.append(avg_length)
            print(
                f"Episode {episode}/{MAX_EPISODES} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Q Loss: {q_loss:.4f}")

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
    plt.plot(q_losses, label='Q Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Q Network Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(eval_avg_rewards, label='Average Reward', color='brown')
    plt.xlabel('Evaluation Episodes (x10)')
    plt.ylabel('Average Reward')
    plt.title('Policy Performance')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(eval_avg_lengths, label='Average Episode Length', color='blue')
    plt.xlabel('Evaluation Episodes (x10)')
    plt.ylabel('Average Length')
    plt.title('Episode Lengths')
    plt.legend()

    plt.tight_layout()
    dataset_choice_str = str(dataset_choice)
    save_dir = Path("Standard_Q_Learning_visual_results")
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
    print(
        f"Final Evaluation over 10 episodes: Average Reward = {final_reward:.2f}, Average Length = {final_length:.2f}")

    # Close environments
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Standard Q-Learning on CartPole-v1 using Offline RL.")
    parser.add_argument('--dataset', type=str, default='350', choices=['50', '150', '250', '350'],
                        help='Choose which dataset to train on: 50, 150, 250, 350')
    args = parser.parse_args()
    standard_q_main(dataset_choice=args.dataset)
