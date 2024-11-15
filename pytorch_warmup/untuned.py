from .base import LinearWarmup, ExponentialWarmup, _check_optimizer


class UntunedLinearWarmup(LinearWarmup):
    """Untuned linear warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    The untuned linear warmup schedule uses the warmup factor

        .. math::
            \\omega_{t}^{\\rm linear, untuned} = \\min \\left\\{ 1, \\frac{1 - \\beta_{2}}{2} \\cdot t \\right\\}

    at each iteration :math:`t`, where :math:`\\beta_{2}` is the second discount factor of Adam.
    In practice, :math:`\\omega_{t}^{\\rm linear, untuned}` is calculated as
    :math:`\\omega_{t}^{\\rm linear, \\tau}` with :math:`\\tau = \\frac{2}{1 - \\beta_{2}}`.

    Note:
        The effective warmup period is defined as

        .. centered::
            :math:`{\\cal T}(\\omega) = \\sum_{t = 1}^{\\infty} \\left( 1 - \\omega_{t} \\right)`

        for a warmup schedule :math:`\\omega = \\left\\{ \\omega_{t} \\right\\}_{t=1}^{\\infty}`.
        The warmup period :math:`\\tau` is deduced from solving approximately the rough equivalence:

        .. centered::
            :math:`{\\cal T}(\\omega^{\\rm expo, untuned}) \\approx {\\cal T}(\\omega^{{\\rm linear},
                \\tau}) \\approx \\frac{\\tau}{2}`.

    Args:
        optimizer (Optimizer): Adam optimizer or its variant:
            :class:`Adam`, :class:`AdamW`, :class:`SparseAdam`, or :class:`NAdam`.
            :class:`RAdam` is not suitable because of the warmup redundancy. This warmup
            schedule makes no sense for :class:`Adamax` as discussed in Note below.
        last_step (int): The index of last step. Default: -1.

    Note:
        This warmup schedule employs the same warmup period :math:`\\tau` for all variants of Adam. However,
        :class:`Adamax` should in principle need no linear warmup because it needs no exponential warmup.
        For further details please refer to Note in the documentation of :class:`UntunedExponentialWarmup`.
        In practice, a linear warmup may slightly improve AdaMax's performance because the initial update step
        is the same as one of the Adam optimizer.

    Example:
        >>> optimizer = AdamW(...)
        >>> lr_scheduler = CosineAnnealingLR(optimizer, ...)
        >>> warmup_scheduler = UntunedLinearWarmup(optimizer)
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

    def __init__(self, optimizer, last_step=-1):
        _check_optimizer(optimizer)

        def warmup_period_fn(beta2):
            return int(2.0 / (1.0-beta2))
        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedLinearWarmup, self).__init__(optimizer, warmup_period, last_step)


class UntunedExponentialWarmup(ExponentialWarmup):
    """Untuned exponential warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    The untuned exponential warmup schedule uses the warmup factor

        .. math::
            \\omega_{t}^{\\rm expo, untuned} =  1 - \\exp \\left( - (1 - \\beta_{2}) \\cdot t \\right)

    at each iteration :math:`t`, where :math:`\\beta_{2}` is the second discount factor of Adam.
    In practice, :math:`\\omega_{t}^{\\rm expo, untuned}` is calculated as
    :math:`\\omega_{t}^{\\rm expo, \\tau}` with :math:`\\tau = \\frac{1}{1 - \\beta_{2}}`.

    Note:
        The constant :math:`\\tau` is derived from the intuition that
        the warmup factor should be roughly equivalent to Adam's second moment bias correction term,
        :math:`1 - \\beta_{2}^{t}`.

    Note:
        The effective warmup period is defined as

        .. centered::
            :math:`{\\cal T}(\\omega) = \\sum_{t = 1}^{\\infty} \\left( 1 - \\omega_{t} \\right)`

        for a warmup schedule :math:`\\omega = \\left\\{ \\omega_{t} \\right\\}_{t=1}^{\\infty}`.
        The constant :math:`\\tau` of the untuned exponential warmup schedule is roughly equivalent to
        its effective warmup period:

        .. centered::
            :math:`{\\cal T}(\\omega^{\\rm expo, untuned}) = 1 / \\left( \\exp( 1 - \\beta_{2}) - 1 \\right) \\approx \\tau`

        for :math:`\\beta_{2}` near 1. The rough equivalence is also achieved for an exponential warmup schedule
        if its :math:`\\tau` is large enough, for example, :math:`\\tau \\ge 1`.

    Args:
        optimizer (Optimizer): Adam optimizer or its variant:
            :class:`Adam`, :class:`AdamW`, :class:`SparseAdam`, or :class:`NAdam`.
            :class:`RAdam` is not suitable because of the warmup redundancy. This warmup
            schedule makes no sense for :class:`Adamax` as discussed in Note below.
        last_step (int): The index of last step. Default: -1.

    Note:
        This warmup schedule employs the same constant :math:`\\tau` for all variants of Adam. However,
        :class:`Adamax` should in principle need no warmup because :class:`Adamax` is derived by employing
        a :math:`L^{p}` norm update rule and letting :math:`p \\rightarrow \\infty`, and the second moment bias
        correction term is :math:`1-\\beta_{2}^{pt}`, to which the warmup factor must be roughly equivalent
        in this warmup schedule derivation. In practice, an exponential warmup may slightly improve AdaMax's
        performance because the initial update step is the same as one of the Adam optimizer.

    Example:
        >>> optimizer = AdamW(...)
        >>> lr_scheduler = CosineAnnealingLR(optimizer, ...)
        >>> warmup_scheduler = UntunedExponentialWarmup(optimizer)
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

    def __init__(self, optimizer, last_step=-1):
        _check_optimizer(optimizer)

        def warmup_period_fn(beta2):
            return int(1.0 / (1.0-beta2))
        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedExponentialWarmup, self).__init__(optimizer, warmup_period, last_step)
