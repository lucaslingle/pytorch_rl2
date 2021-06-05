# pytorch_rl2_mdp_lstm

Implementation of Duan et al., 2016 - RL^2: Fast Reinforcement via Slow Reinforcement Learning [[1]](https://arxiv.org/abs/1611.02779).

Previously, we implemented RL^2 for multi-armed bandit problems. This implementation focuses on tabular MDPs. 
A more general implementation may follow. 

## Background

The main idea of RL^2 and the related approach by Wang et al., 2016 [[2]](https://arxiv.org/abs/1611.05763),
is as follows: 

A stateful reinforcement learning agent can be trained on a distribution of environments, using standard reinforcement learning algorithms, 
so that at every state they are encouraged to maximize the expected cumulative discounted reward [[3]](https://www.cis.upenn.edu/~mkearns/finread/BaxterBartlett.pdf#page=6).

This stateful agent can be provided its previous action and the resulting immediate reward from that action.  
The agent's hidden state is believed to approximate sufficient statistics for the belief of which MDP the agent is in [[4]](https://arxiv.org/abs/1905.03030).
This hidden state guides the agent's learning and behavior in the new environment. 

In practice, maximizing this objective leads to policies which can explore a new environment to learn relevant information to obtain more reward.
This occurs for exploratory actions that lead the agent to an information state in which it can better exploit the environment. 
A simplified explanation is that these information states have higher value, and this value is backed up to the exploratory action. 

Through such exploration, the agent can acquire the relevant information and transition from a highly exploratory policy to a highly exploitative policy purely through changes in its hidden state.

After training, this means the agent can actively attempt to learn how to solve a given task even when the optimizer is turned off. 

## Getting Started

Install the following system dependencies:
#### Ubuntu     
```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of the system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Everyone
Once the system dependencies have been installed, it's time to install the python dependencies. 
Install the conda package manager from https://docs.conda.io/en/latest/miniconda.html

Then run
```bash
conda create --name rl2_mdp python=3.8.1
conda activate rl2_mdp
git clone https://github.com/lucaslingle/pytorch_rl2_mdp_lstm
cd pytorch_rl2_mdp_lstm
pip install -e .
```

## Usage

### Training
To train the default settings, you can simply type:
```bash
mpirun -np 8 python -m train
```

This will launch 8 parallel processes, each running the ```train.py``` script. These processes will progress through several meta-episodes of distinct MDPs in parallel, and communicate gradient information and synchronize parameters using [OpenMPI](https://www.open-mpi.org/).

To see additional options, you can simply type ```python train.py --help```. 

### Checkpoints
By default, checkpoints are saved to ```./checkpoints/defaults```. To pick a different checkpoint directory, 
you can set the ```--checkpoint_dir``` flag, and to pick a different checkpoint name, you can set the 
```--model_name``` flag.

