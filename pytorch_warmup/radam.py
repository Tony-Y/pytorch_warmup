import math
from .base import BaseWarmup, _check_optimizer


def rho_inf_fn(beta2):
    """Returns the constant of the RAdam algorithm, :math:`\\rho_{\\infty}`.

    Args:
        beta2 (float): The second discount factor of Adam, :math:`\\beta_{2}`.
    """
    return 2.0 / (1 - beta2) - 1


def rho_fn(t, beta2, rho_inf):
    """Returns the value of the function of the RAdam algorithm, :math:`\\rho_{t}`,
    at an iteration :math:`t`.

    Args:
        t (int): The iteration :math:`t`.
        beta2 (float): The second discount factor of Adam, :math:`\\beta_{2}`.
        rho_inf (float): The constant of the RAdam algorithm, :math:`\\rho_{\\infty}`.
    """
    b2t = beta2 ** t
    rho_t = rho_inf - 2 * t * b2t / (1 - b2t)
    return rho_t


def get_offset(beta2, rho_inf):
    """Returns the minimal offset :math:`\\delta`.

    Args:
        beta2 (float): The second discount factor of Adam, :math:`\\beta_{2}`.
        rho_inf (float): The constant of the RAdam algorithm, :math:`\\rho_{\\infty}`.
    """
    if not beta2 > 0.6:
        raise ValueError('beta2 ({}) must be greater than 0.6'.format(beta2))
    offset = 1
    while True:
        if rho_fn(offset, beta2, rho_inf) > 4:
            return offset
        offset += 1


class RAdamWarmup(BaseWarmup):
    """RAdam warmup schedule.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    The RAdam algorithm uses the warmup factor

        .. math::
            \\omega_{t}^{\\rm RAdam} = \\sqrt{ \\frac{ \\
            ( \\rho_{t} - 4 ) ( \\rho_{t} - 2 ) \\rho_{\\infty} }{ \\
            ( \\rho_{\\infty} - 4) (\\rho_{\\infty} - 2 ) \\rho_{t} } }

    at each iteration :math:`t` for :math:`\\rho_{t} > 4`, where

        .. math::
            \\rho_{\\infty} = \\frac{ 2 }{ 1 - \\beta_{2} } - 1

    and

        .. math::
            \\rho_{t} = \\rho_{\\infty} - \\frac{ 2 t \\cdot \\beta_{2}^{t} }{ 1 - \\beta_{2}^{t} }

    where :math:`\\beta_{2}` is the second discount factor of Adam. In the RAdam warmup schedule,
    the minimal offset :math:`\\delta` is chosen such that :math:`\\rho_{\\delta} > 4`, and then
    :math:`\\omega_{t+\\delta-1}^{\\rm RAdam}` is employed as the warmup factor at each iteration :math:`t`.
    For all practically relevant values of :math:`\\beta_{2}` (:math:`0.8 < \\beta_{2} \\le 1`),
    :math:`\\delta \\le 5` as deduced from Fact 3.1 of the paper.

    Args:
        optimizer (Optimizer): Adam optimizer or its variant:
            :class:`Adam`, :class:`AdamW`, :class:`SparseAdam`, or :class:`NAdam`.
            :class:`RAdam` is not suitable because of the warmup redundancy. This warmup
            schedule makes no sense for :class:`Adamax` and, in principle, the AMSGrad variant of
            :class:`Adam` and :class:`AdamW` as discussed in Note below. In practice, this warmup
            schedule improves the performance of the AMSGrad variant like that of the vanilla Adam.
        last_step (int): The index of last step. Default: -1.

    Note:
        This warmup schedule employs the same warmup factor for all variants of Adam. However,
        according to the RAdam theory,
        :class:`Adamax` and the AMSGrad variant of :class:`Adam` and :class:`AdamW` should
        have a different warmup factor because its :math:`\\psi(\\cdot)` function is different from one of the
        vanilla Adam, where :math:`\\psi(\\cdot)` specifies how the adaptive learning rate at :math:`t` is
        calculated. The RAdam theory derives the warmup factor :math:`\\omega_{t}` from
        :math:`\\psi(\\cdot)`. For gradients :math:`\\left\\{ g_{i} \\right\\}` viewed as i.i.d. normal random
        variables,

        .. centered::
            :math:`\\omega_{t} = \\sqrt{ C_{\\rm var} / {\\rm Var}\\left[ \\psi(g_{1}, \\dots, g_{t}) \\right] }`

        where

        .. centered::
            :math:`C_{\\rm var} = \\inf_{t} {\\rm Var}\\left[ \\psi(g_{1}, \\dots, g_{t}) \\right]`.

        (For details please refer to `On the Variance of the Adaptive Learning Rate and Beyond
        <https://arxiv.org/abs/1908.03265>`_.)

        The variance hypothesis of the RAdam theory has become questionable
        since Ma and Yarats' paper pointed out that the adaptive learning rate may not be the best medium
        of analysis for understanding the role of warmup in Adam.

    Example:
        >>> optimizer = AdamW(...)
        >>> lr_scheduler = CosineAnnealingLR(optimizer, ...)
        >>> warmup_scheduler = RAdamWarmup(optimizer)
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
        warmup_params = [
            dict(
                beta2=x['betas'][1],
                rho_inf=rho_inf_fn(x['betas'][1]),
            )
            for x in optimizer.param_groups
        ]
        for x in warmup_params:
            x['offset'] = get_offset(**x)
        super(RAdamWarmup, self).__init__(optimizer, warmup_params, last_step)

    def warmup_factor(self, step, beta2, rho_inf, offset):
        """Returns the warmup factor :math:`\\omega_{t+\\delta-1}^{\\rm RAdam}` at an iteration :math:`t`.

        Args:
            step (int): The index of current step.
            beta2 (float): The second discount factor of Adam, :math:`\\beta_{2}`.
            rho_inf (float): The constant of the RAdam algorithm, :math:`\\rho_{\\infty}`.
            offset (int): The minimal offset :math:`\\delta`.
        """
        rho = rho_fn(step+offset, beta2, rho_inf)
        numerator = (rho - 4) * (rho - 2) * rho_inf
        denominator = (rho_inf - 4) * (rho_inf - 2) * rho
        return math.sqrt(numerator/denominator)
