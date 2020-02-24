from torch.optim.optimizer import Optimizer


class _LRScheduler(object):
    def __init__(self, optimizer, last_global_step=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_global_step == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_global_step + 1)
        self.last_global_step = last_global_step

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, global_step=None):
        if global_step is None:
            global_step = self.last_global_step + 1
        self.last_global_step = global_step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class GlobalStepLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size global_steps. When last_global_step=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_global_step (int): The index of last global_step. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if global_step < 30
        >>> # lr = 0.005    if 30 <= global_step < 60
        >>> # lr = 0.0005   if 60 <= global_step < 90
        >>> # ...
        >>> scheduler = GlobalStepLR(optimizer, step_size=30, gamma=0.1)
        >>> for global_step in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_global_step=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(GlobalStepLR, self).__init__(optimizer, last_global_step)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_global_step // self.step_size)
                for base_lr in self.base_lrs]