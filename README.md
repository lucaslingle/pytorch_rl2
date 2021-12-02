# RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning

This repo contains implementations of the algorithms, architectures, and environments from Duan et al., 2016 - ['RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning'](https://arxiv.org/pdf/1611.02779.pdf), and Mishra et al., 2017 - ['A Simple Neural Attentive Meta-Learner'](https://arxiv.org/pdf/1707.03141.pdf).

It has also recently been redesigned to facilitate rapid prototyping of new stateful architectures for memory-based meta-reinforcement learning agents.   

## Background

The main idea of RL^2 is that a reinforcement learning agent with memory can be trained on a distribution of environments, 
and can thereby learn an algorithm to effectively transition from exploring these environments to exploiting them.

In fact, the RL^2 training curriculum effectively trains an agent to behave as if it possesses a probabilistic model of the possible environments it is acting in. 

This theoretical background of RL^2 is discussed by Ortega et al., 2019 and a concise treatment can be found in [my blog post](https://lucaslingle.wordpress.com/2021/10/07/on-memory-based-meta-reinforcement-learning/).   

## Getting Started

Install the following system dependencies:
#### Ubuntu     
```bash
sudo apt-get update
sudo apt-get install -y cmake openmpi-bin openmpi-doc libopenmpi-dev
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
conda create --name pytorch_rl2 python=3.8.1
conda activate pytorch_rl2
git clone https://github.com/lucaslingle/pytorch_rl2
cd pytorch_rl2
pip install -e .
```

## Usage

### Training
To train the default settings, you can simply type:
```bash
mpirun -np 8 python -m train
```

This will launch 8 parallel processes, each running the ```train.py``` script. These processes each generate meta-episodes separately and then synchronously train on the collected experience in a data-parallel manner, with gradient information and model parameters synchronized across processes using mpi4py.

To see additional configuration options, you can simply type ```python train.py --help```. Among other options, we support various architectures including GRU, LSTM, SNAIL, and Transformer models.

### Checkpoints
By default, checkpoints are saved to ```./checkpoints/defaults```. To pick a different checkpoint directory during training, 
you can set the ```--checkpoint_dir``` flag, and to pick a different checkpoint name, you can set the 
```--model_name``` flag.

## Reproducing the Papers

Our implementations closely matched or slightly exceeded the published performance of RL^2 GRU (Duan et al., 2016) and RL^2 SNAIL (Mishra et al., 2017) in every setting we tested.

In the tables below, ```n``` is the number of episodes per meta-episode, and ```k``` is the number of actions. 
Following Duan et al., 2016 and Mishra et al., 2017, in our tabular MDP experiments, all MDPs have 10 states and 5 actions, and the episode length is 10.  

### Bandit case:

| Setup      | Random | Gittins |    TS |   OTS |  UCB1 | eps-Greedy | Greedy | RL^2 GRU (paper) | RL^2 GRU (ours) | RL^2 SNAIL (paper) | RL^2 SNAIL (ours)  |
| ---------- | ------ | ------- | ----- | ----- | ----- | ---------- | ------ | ---------------- | --------------- | ------------------ | ------------------ |
|  n=10,k=5  |    5.0 |     6.6 |   5.7 |   6.5 |   6.7 |        6.6 |    6.6 |              6.7 |            6.7  |                6.6 |                6.8 |
|  n=10,k=10 |    5.0 |     6.6 |   5.5 |   6.2 |   6.7 |        6.6 |    6.6 |              6.7 |                 |                6.7 |                    |
|  n=10,k=50 |    5.1 |     6.5 |   5.2 |   5.5 |   6.6 |        6.5 |    6.5 |              6.8 |                 |                6.7 |                    |
| n=100,k=5  |   49.9 |    78.3 |  74.7 |  77.9 |  78.0 |       75.4 |   74.8 |             78.7 |            78.7 |               79.1 |               78.5 |
| n=100,k=10 |   49.9 |    82.8 |  76.7 |  81.4 |  82.4 |       77.4 |   77.1 |             83.5 |                 |               83.5 |                    |
| n=100,k=50 |   49.8 |    85.2 |  64.5 |  67.7 |  84.3 |       78.3 |   78.0 |             84.9 |                 |               85.1 |                    |
| n=500,k=5  |  249.8 |   405.8 | 402.0 | 406.7 | 405.8 |      388.2 |  380.6 |            401.6 |                 |              408.1 |                    |
| n=500,k=10 |  249.0 |   437.8 | 429.5 | 438.9 | 437.1 |      408.0 |  395.0 |            432.5 |                 |              432.4 |                    |
| n=500,k=50 |  249.6 |   463.7 | 427.2 | 437.6 | 457.6 |      413.6 |  402.8 |            438.9 |                 |              442.6 |                    |

### MDP case:

| Setup      | Random |   PSRL |  OPSRL |  UCRL2 |    BEB | eps-Greedy | Greedy | RL^2 GRU (paper) | RL^2 GRU (ours) | RL^2 SNAIL (paper) | RL^2 SNAIL (ours)  |
| ---------- | ------ | ------ | ------ | ------ | ------ | ---------- | ------ | ---------------- | --------------- | ------------------ | ------------------ |
| n=10       |  100.1 |  138.1 |  144.1 |  146.6 |  150.2 |      132.8 |  134.8 |            156.2 |           157.3 |              159.1 |              160.1 |
| n=25       |  250.2 |  408.8 |  425.2 |  424.1 |  427.8 |      377.3 |  368.8 |            445.7 |                 |              447.2 |                    |
| n=50       |  499.7 |  904.4 |  930.7 |  918.9 |  917.8 |      823.3 |  769.3 |            936.1 |                 |              942.3 |                    |
| n=75       |  749.9 | 1417.1 | 1449.2 | 1427.6 | 1422.6 |     1293.9 | 1172.9 |           1428.8 |                 |             1447.5 |                    |
| n=100      |  999.4 | 1939.5 | 1973.9 | 1942.1 | 1935.1 |     1778.2 | 1578.5 |           1913.7 |                 |             1953.1 |                    |

To perform policy optimization, we used PPO. We used layer norm instead of weight norm, and we report peak performance over training. Our performance statistics are averaged over 1000 meta-episodes.

In all cases, for training we used a configuration where the total number of observations per policy improvement phase was equal to 240,000. This is comparable to the 250,000 used in prior works.
The per-process batch size was 60 trajectories. There were 8 processes. There were 8 PPO optimization epochs per policy improvement phase. 

All other hyperparameters were set to their default values in the ```train.py``` script, except for the SNAIL experiments, where we used ```--num_features=32``` due to the skip-connections. 
