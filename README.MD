<!-- README.MD File -->

## File Description

- **`rl_hw3_demo.py`**  
  This file contains the script for the expert algorithm used to collect training data, generating datasets based on expert strategies.

- **`dataset_episode_*.npz`**  
  These files are the collected training datasets, with each file corresponding to a dataset of different size or quality. The specific reward results are shown in the figure below:  
  ![](reward.png)

- **`environment.yml`**  
  Please use the following recommended environment setup.


  # Behavior Cloning and CQL Algorithm Implementation

This repository contains the implementation of **Behavior Cloning (BC)** and **Conservative Q-Learning (CQL)** algorithms for training reinforcement learning models using a dataset of state-action pairs. It provides an environment for experimenting with these algorithms, comparing their performance, and analyzing the effects of different parameters and dataset variations on training results.


## Installation

Clone the repository and install the required packages using the provided `environment.yml` file:

```bash
git clone
cd rl_hw3
conda env create -f environment.yml
conda activate rl_hw3
```

## Running the Code
python StandardQL_Training.py --dataset 50      # Train a standard Q-Learning model using a dataset of 50 episodes
python BC_Training.py --dataset 50              # Train a Behavior Cloning model using a dataset of 50 episodes
python CQL_Training.py --dataset 50             # Train a Conservative Q-Learning model using a dataset of 50 episodes

You may also specify the number of epochs, batch size, and other hyperparameters using command-line arguments. For more information, run the scripts with the `--help` flag such as 50, 150, 250, and 350 argument parameters supported.
