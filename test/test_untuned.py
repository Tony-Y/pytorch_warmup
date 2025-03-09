import unittest
import math
from typing import Union
import torch
import pytorch_warmup as warmup

from .test_base import _test_state_dict, _set_lr, _get_lr


# Type alias
WarmupClass = Union[
    type[warmup.UntunedLinearWarmup],
    type[warmup.UntunedExponentialWarmup],
    type[warmup.RAdamWarmup],
]


def _test_optimizer(self: unittest.TestCase, warmup_class: WarmupClass) -> None:
    with self.assertRaises(TypeError, msg="optimizer type") as cm:
        warmup_class(optimizer=0)  # type: ignore[arg-type]
    self.assertEqual(str(cm.exception), "0 (int) is not an Optimizer.")


class TestUntuned(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_untuned_linear(self) -> None:
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.Adam(
            [{"params": [p1]}, {"params": [p2], "lr": _set_lr(0.1)}],  # type: ignore[arg-type]
            lr=_set_lr(0.5),
            betas=(0.9, 0.7),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        print()
        for step in range(1, 11):
            lr = [_get_lr(x["lr"]) for x in optimizer.param_groups]
            print(f"{step} {lr}")
            if step < 6:
                self.assertAlmostEqual(lr[0], 0.5 * step / 6)
                self.assertAlmostEqual(lr[1], 0.1 * step / 6)
            else:
                self.assertAlmostEqual(lr[0], 0.5)
                self.assertAlmostEqual(lr[1], 0.1)
            optimizer.zero_grad()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        _test_state_dict(self, warmup_scheduler, lambda: warmup.UntunedLinearWarmup(optimizer))

        _test_optimizer(self, warmup.UntunedLinearWarmup)

    def test_untuned_exponential(self) -> None:
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.Adam(
            [{"params": [p1]}, {"params": [p2], "lr": _set_lr(0.1)}],  # type: ignore[arg-type]
            lr=_set_lr(0.5),
            betas=(0.9, 0.7),
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
        print()
        for step in range(1, 11):
            lr = [_get_lr(x["lr"]) for x in optimizer.param_groups]
            print(f"{step} {lr}")
            self.assertAlmostEqual(lr[0], 0.5 * (1 - math.exp(-step / 3)))
            self.assertAlmostEqual(lr[1], 0.1 * (1 - math.exp(-step / 3)))
            optimizer.zero_grad()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        _test_state_dict(
            self, warmup_scheduler, lambda: warmup.UntunedExponentialWarmup(optimizer)
        )

        _test_optimizer(self, warmup.UntunedExponentialWarmup)
