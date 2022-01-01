from typing import Union

import torch as tc

from rl2.agents.integration.policy_net import StatefulPolicyNet
from rl2.agents.integration.value_net import StatefulValueNet
from rl2.agents.integration.ddp import StatefulDDP


Module = tc.nn.Module
Optimizer = tc.optim.Optimizer
Scheduler = tc.optim.lr_scheduler._LRScheduler
Checkpointable = Union[Module, Optimizer, Scheduler]

DDP = tc.nn.parallel.DistributedDataParallel
StatefulPolicyNet = Union[StatefulDDP, StatefulPolicyNet]
StatefulValueNet = Union[StatefulDDP, StatefulValueNet]
