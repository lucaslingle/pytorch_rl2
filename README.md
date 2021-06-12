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

## Reproducing the Paper

### Bandit case:

| Setup      | Random | Gittins |    TS |   OTS |  UCB1 | eps-Greedy | Greedy | RL^2 (paper) | RL^2 (ours) |
| ---------- | ------ | ------- | ----- | ----- | ----- | ---------- | ------ | ------------ | ----------- | 
|  n=10,k=5  |    5.0 |     6.6 |   5.7 |   6.5 |   6.7 |        6.6 |    6.6 |          6.7 |         6.7 |
|  n=10,k=10 |    5.0 |     6.6 |   5.5 |   6.2 |   6.7 |        6.6 |    6.6 |          6.7 |             |
|  n=10,k=50 |    5.1 |     6.5 |   5.2 |   5.5 |   6.6 |        6.5 |    6.5 |          6.8 |             |
| n=100,k=5  |   49.9 |    78.3 |  74.7 |  77.9 |  78.0 |       75.4 |   74.8 |         78.7 |        78.7 |
| n=100,k=10 |   49.9 |    82.8 |  76.7 |  81.4 |  82.4 |       77.4 |   77.1 |         83.5 |             |
| n=100,k=50 |   49.8 |    85.2 |  64.5 |  67.7 |  84.3 |       78.3 |   78.0 |         84.9 |             |
| n=500,k=5  |  249.8 |   405.8 | 402.0 | 406.7 | 405.8 |      388.2 |  380.6 |        401.6 |             |
| n=500,k=10 |  249.0 |   437.8 | 429.5 | 438.9 | 437.1 |      408.0 |  395.0 |        432.5 |             |
| n=500,k=50 |  249.6 |   463.7 | 427.2 | 437.6 | 457.6 |      413.6 |  402.8 |        438.9 |             |

### MDP case:

| Setup      | Random |   PSRL |  OPSRL |  UCRL2 |    BEB | eps-Greedy | Greedy | RL^2 (paper) | RL^2 (ours) |
| ---------- | ------ | ------ | ------ | ------ | ------ | ---------- | ------ | ------------ | ----------- |
| n=10       |  100.1 |  138.1 |  144.1 |  146.6 |  150.2 |      132.8 |  134.8 |        156.2 |       174.2 |
| n=25       |  250.2 |  408.8 |  425.2 |  424.1 |  427.8 |      377.3 |  368.8 |        445.7 |             |
| n=50       |  499.7 |  904.4 |  930.7 |  918.9 |  917.8 |      823.3 |  769.3 |        936.1 |             |
| n=75       |  749.9 | 1417.1 | 1449.2 | 1427.6 | 1422.6 |     1293.9 | 1172.9 |       1428.8 |             |
| n=100      |  999.4 | 1939.5 | 1973.9 | 1942.1 | 1935.1 |     1778.2 | 1578.5 |       1913.7 |             |

Note that in our case, we use PPO instead of TRPO, and we report peak performance over training. This was always similar to, 
but not always identical to, final performance. 

In all cases, we used a configuration where the total number of observations per policy improvement phase was equal to 240,000. 
The per-process batch size was 60 trajectories. There were 8 processes. There were 200 gradient steps per policy improvement phase. 
To stabilize training, we used the Adam hyperparameters from [Kapturowski et al., 2019](https://openreview.net/pdf?id=r1lyTjAqYX). 

Finally, note that the numbers for the MDP case are provisional, as we sample a stochastic reward at each timestep from a normal distribution 
with unit variance, and resample the means when instantiating a new MDP. It is possible, and perhaps likely, that Duan et al., 2016 
sampled the means once when creating the benchmark, and then sampled new deterministic reward for each MDP from a normal distribution 
with these means and with unit variances. (If true, this could make their results a bit difficult to reproduce, since the distribution over MDPs 
would depend heavily on the random means sampled from the hyperprior.)
