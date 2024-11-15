import math
from contextlib import contextmanager
from torch.optim import Optimizer


def _check_optimizer(optimizer):
    if not isinstance(optimizer, Optimizer):
        raise TypeError('{} ({}) is not an Optimizer.'.format(
            optimizer, type(optimizer).__name__))


class BaseWarmup(object):
    """Base class for all warmup schedules.

    The learning rate :math:`\\alpha_{t}` is dampened by multiplying it by
    the warmup factor :math:`\\omega_{t} \\in [0, 1]` at each iteration :math:`t`.
    Thus, the modified learning rate

        .. math::
            \\hat \\alpha_{t} = \\alpha_{t} \\cdot \\omega_{t}

    is used by the optimizer.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_params (list): Warmup parameters.
        last_step (int): The index of last step. Default: -1.
    """

    def __init__(self, optimizer, warmup_params, last_step=-1):
        self.optimizer = optimizer
        self.warmup_params = warmup_params
        self.last_step = last_step
        self.lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.dampen()

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.

        It contains an entry for every variable in :attr:`self.__dict__` which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.

        Args:
            state_dict (dict): Warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def dampen(self, step=None):
        """Dampens the learning rate.

        It is not recommended to explicitly call this method for PyTorch 1.4.0 or later.
        Please use the :meth:`dampening` context manager that calls this method correctly.

        Args:
            step (int): The index of current step. Default: ``None``.
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step

        for group, params in zip(self.optimizer.param_groups, self.warmup_params):
            omega = self.warmup_factor(step, **params)
            group['lr'] *= omega

    @contextmanager
    def dampening(self):
        """Dampens the learning rate after calling the :meth:`step` method of the learning
        rate scheduler.

        The :meth:`step` method calls must be placed in a suite of the ``with`` statement having
        the :meth:`dampening` context manager.

        Examples:
            >>> # For no LR scheduler
            >>> with warmup_scheduler.dampening():
            >>>     pass

            >>> # For a single LR scheduler
            >>> with warmup_scheduler.dampening():
            >>>     lr_scheduler.step()

            >>> # To chain two LR schedulers
            >>> with warmup_scheduler.dampening():
            >>>     lr_scheduler1.step()
            >>>     lr_scheduler2.step()

            >>> # To delay an LR scheduler
            >>> iteration = warmup_scheduler.last_step + 1
            >>> with warmup_scheduler.dampening():
            >>>     if iteration >= warmup_period:
            >>>         lr_scheduler.step()
        """
        for group, lr in zip(self.optimizer.param_groups, self.lrs):
            group['lr'] = lr
        yield
        self.lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.dampen()

    def warmup_factor(self, step, **params):
        """Returns the warmup factor :math:`\\omega_{t}` at an iteration :math:`t`.

        :meth:`dampen` uses this method to get the warmup factor for each parameter group.
        It is unnecessary to explicitly call this method.

        Args:
            step (int): The index of current step.
            params (dict): The warmup parameters. For details, refer to the arguments of
                each subclass method.
        """
        raise NotImplementedError


def get_warmup_params(warmup_period, group_count):
    if isinstance(warmup_period, list):
        if len(warmup_period) != group_count:
            raise ValueError(
                'The size of warmup_period ({}) does not match the size of param_groups ({}).'.format(
                    len(warmup_period), group_count))
        for x in warmup_period:
            if not isinstance(x, int):
                raise TypeError(
                    'An element in warmup_period, {}, is not an int.'.format(
                        type(x).__name__))
            if x <= 0:
                raise ValueError(
                    'An element in warmup_period must be a positive integer, but is {}.'.format(x))
        warmup_params = [dict(warmup_period=x) for x in warmup_period]
    elif isinstance(warmup_period, int):
        if warmup_period <= 0:
            raise ValueError(
                'warmup_period must be a positive integer, but is {}.'.format(warmup_period))
        warmup_params = [dict(warmup_period=warmup_period)
                         for _ in range(group_count)]
    else:
        raise TypeError('{} ({}) is not a list nor an int.'.format(
            warmup_period, type(warmup_period).__name__))
    return warmup_params


class LinearWarmup(BaseWarmup):
    """Linear warmup schedule.

    The linear warmup schedule uses the warmup factor

        .. math::
            \\omega_{t}^{\\rm linear, \\tau} = \\min \\left\\{ 1, \\frac{1}{\\tau} \\cdot t \\right\\}

    at each iteration :math:`t`, where :math:`\\tau` is the warmup period.

    Args:
        optimizer (Optimizer): Wrapped optimizer. :class:`RAdam` is not suitable because of the
            warmup redundancy.
        warmup_period (int or list[int]): The warmup period :math:`\\tau`.
        last_step (int): The index of last step. Default: -1.

    Example:
        >>> lr_scheduler = CosineAnnealingLR(optimizer, ...)
        >>> warmup_scheduler = LinearWarmup(optimizer, warmup_period=2000)
        >>> for batch in dataloader:
        >>>     optimizer.zero_grad()
        >>>     loss = ...
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     with warmup_scheduler.dampening():
        >>>         lr_scheduler.step()

    Warning:
        The warmup schedule must not be initialized before the initialization of the learning rate schedule.
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        _check_optimizer(optimizer)
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(warmup_period, group_count)
        super(LinearWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        """Returns the warmup factor :math:`\\omega_{t}^{\\rm linear, \\tau}` at an iteration :math:`t`.

        Args:
            step (int): The index of current step.
            warmup_period (int): The warmup period :math:`\\tau`.
        """
        return min(1.0, (step+1) / warmup_period)


class ExponentialWarmup(BaseWarmup):
    """Exponential warmup schedule.

    The exponential warmup schedule uses the warmup factor

        .. math::
            \\omega_{t}^{\\rm expo, \\tau} =  1 - \\exp \\left( - \\frac{1}{\\tau} \\cdot t \\right)

    at each iteration :math:`t`, where the constant :math:`\\tau` is analogous to
    a linear warmup period.

    Args:
        optimizer (Optimizer): Wrapped optimizer. :class:`RAdam` is not suitable because of the
            warmup redundancy.
        warmup_period (int or list[int]): The constant :math:`\\tau` analogous to a linear warmup period.
        last_step (int): The index of last step. Default: -1.

    Example:
        >>> lr_scheduler = CosineAnnealingLR(optimizer, ...)
        >>> warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=1000)
        >>> for batch in dataloader:
        >>>     optimizer.zero_grad()
        >>>     loss = ...
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     with warmup_scheduler.dampening():
        >>>         lr_scheduler.step()

    Warning:
        The warmup schedule must not be initialized before the initialization of the learning rate schedule.
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        _check_optimizer(optimizer)
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(warmup_period, group_count)
        super(ExponentialWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, warmup_period):
        """Returns the warmup factor :math:`\\omega_{t}^{\\rm expo, \\tau}` at an iteration :math:`t`.

        Args:
            step (int): The index of current step.
            warmup_period (int): The constant :math:`\\tau` analogous to a linear warmup period.
        """
        return 1.0 - math.exp(-(step+1) / warmup_period)
