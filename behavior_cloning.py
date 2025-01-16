import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Define the BC Model
class BehaviorCloningModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BehaviorCloningModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.fc3(x)
        return action


# Step 2: Define the Behavior Cloning (BC) Training Class
class BehaviorCloning:
    def __init__(self, state_dim, action_dim, dataset, lr=1e-3, batch_size=64):
        self.model = BehaviorCloningModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()  # For discrete actions (categorical)
        self.dataset = dataset  # Dictionary with 'states' and 'actions'
        self.batch_size = batch_size
        self.losses = []  # List to store the loss values for visualization
        self.predictions = []  # List to store predictions for analysis
        self.true_labels = []  # List to store true actions for analysis

    def train(self, epochs=10):
        # Convert dataset to tensors (if not already)
        states = torch.tensor(self.dataset['states'], dtype=torch.float32)
        actions = torch.tensor(self.dataset['actions'], dtype=torch.long)
        
        for epoch in range(epochs):
            # Shuffle the dataset for each epoch
            perm = torch.randperm(states.size(0))
            states = states[perm]
            actions = actions[perm]

            # Train in batches
            epoch_loss = 0.0
            for i in range(0, len(states), self.batch_size):
                state_batch = states[i:i + self.batch_size]
                action_batch = actions[i:i + self.batch_size]

                # Forward pass
                predicted_actions = self.model(state_batch)
                
                # Calculate the loss
                loss = self.loss_fn(predicted_actions, action_batch)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                # Store predictions and true labels for evaluation
                if i == 0:  # Store only for the first batch (you can adjust this as needed)
                    predicted = torch.argmax(predicted_actions, dim=1).cpu().numpy()
                    true = action_batch.cpu().numpy()
                    self.predictions.append(predicted)
                    self.true_labels.append(true)

            avg_loss = epoch_loss / len(states)
            self.losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, state):
        # Predict the action for a given state
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            action = self.model(state)
            return torch.argmax(action).item()


# Step 3: Function to Load Datasets
def load_dataset(file_path):
    # Assuming dataset is saved as npz format with states and actions
    data = np.load(file_path)
    return {
        'states': data['states'],  # State data (features)
        'actions': data['actions']  # Action data (labels)
    }

# Step 4: Main function to Execute the BC Algorithm
if __name__ == "__main__":
    # Load your dataset (choose one: 50, 150, 250, 350 episodes)
    dataset = load_dataset('datasets/dataset_episode_250.npz')  # Update the path as per your directory structure

    # Initialize the Behavior Cloning model
    bc_model = BehaviorCloning(state_dim=dataset['states'].shape[1],  # Assuming states are 2D (batch_size, state_dim)
                               action_dim=len(np.unique(dataset['actions'])),  # Number of unique actions
                               dataset=dataset,
                               lr=1e-3)

    # Train the model
    bc_model.train(epochs=10)

    # Test with a sample state (e.g., first state)
    sample_state = dataset['states'][0]  # First state from dataset
    predicted_action = bc_model.predict(sample_state)
    print(f"Predicted Action for the first state: {predicted_action}")

    # Visualization - Plotting Loss vs Epoch
    plt.plot(bc_model.losses)
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Visualization - Compare predictions to actual actions for the first batch
    predictions_flat = np.concatenate(bc_model.predictions, axis=0)
    true_labels_flat = np.concatenate(bc_model.true_labels, axis=0)

    # Plot Predicted vs Actual Actions
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(predictions_flat)), predictions_flat, color='r', label='Predicted Actions')
    plt.scatter(np.arange(len(true_labels_flat)), true_labels_flat, color='b', label='True Actions')
    plt.title('Predicted vs Actual Actions')
    plt.xlabel('Index')
    plt.ylabel('Action')
    plt.legend()
    plt.show()

    # Recording system - Save predictions and true labels for future analysis
    np.save('predictions.npy', predictions_flat)
    np.save('true_labels.npy', true_labels_flat)
