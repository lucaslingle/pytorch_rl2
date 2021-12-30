import functools

from torch.nn.parallel import DistributedDataParallel as DDP


class StatefulDDP(DDP):
    @functools.wraps(DDP.__init__)
    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)
        self._module = module

    def initial_state(self, batch_size):
        return self._module.initial_state(batch_size)
