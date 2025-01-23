import unittest
import os
import math
import torch
from torch import Tensor
import pytorch_warmup as warmup


def _test_state_dict(self, warmup_scheduler, constructor):
    warmup_scheduler_copy = constructor()
    warmup_scheduler_copy.load_state_dict(warmup_scheduler.state_dict())
    for key in warmup_scheduler.__dict__.keys():
        if key not in ['optimizer']:
            print(key, warmup_scheduler.__dict__[key])
            self.assertAlmostEqual(warmup_scheduler.__dict__[key],
                                   warmup_scheduler_copy.__dict__[key])


def _test_optimizer(self, warmup_class):
    with self.assertRaises(TypeError, msg='optimizer type') as cm:
        warmup_class(optimizer=0, warmup_period=5)
    self.assertEqual(str(cm.exception), '0 (int) is not an Optimizer.')


def _test_get_warmup_params(self, optimizer, warmup_class):
    with self.assertRaises(ValueError, msg='warmup_period size') as cm:
        warmup_class(optimizer, warmup_period=[5])
    self.assertEqual(str(cm.exception), 'The size of warmup_period (1) does not match the size of param_groups (2).')

    with self.assertRaises(TypeError, msg='warmup_period element type') as cm:
        warmup_class(optimizer, warmup_period=[5.0, 10.0])
    self.assertEqual(str(cm.exception), 'An element in warmup_period, float, is not an int.')

    with self.assertRaises(ValueError, msg='warmup_period element range') as cm:
        warmup_class(optimizer, warmup_period=[5, 0])
    self.assertEqual(str(cm.exception), 'An element in warmup_period must be a positive integer, but is 0.')

    with self.assertRaises(ValueError, msg='warmup_period range') as cm:
        warmup_class(optimizer, warmup_period=0)
    self.assertEqual(str(cm.exception), 'warmup_period must be a positive integer, but is 0.')

    with self.assertRaises(TypeError, msg='warmup_period type') as cm:
        warmup_class(optimizer, warmup_period=5.0)
    self.assertEqual(str(cm.exception), '5.0 (float) is not a list nor an int.')


def _wrap(value):
    return torch.tensor(value)


def _unwrap(value):
    assert isinstance(value, Tensor)
    assert value.dim() == 0
    return value.item()


def _identity(value):
    return value


def _assert_naked(value):
    assert not isinstance(value, Tensor)
    return value


_set_lr, _get_lr = (_wrap, _unwrap) if 'WRAPPED_LR' in os.environ else (_identity, _assert_naked)


class TestBase(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def test_linear(self):
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.SGD([
                {'params': [p1]},
                {'params': [p2], 'lr': _set_lr(0.1)}
            ], lr=_set_lr(0.5))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=5)
        print()
        for step in range(1, 11):
            lr = [_get_lr(x['lr']) for x in optimizer.param_groups]
            print(f'{step} {lr}')
            if step < 5:
                self.assertAlmostEqual(lr[0], 0.5 * step / 5)
                self.assertAlmostEqual(lr[1], 0.1 * step / 5)
            else:
                self.assertAlmostEqual(lr[0], 0.5)
                self.assertAlmostEqual(lr[1], 0.1)
            optimizer.zero_grad()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        _test_state_dict(self, warmup_scheduler,
                         lambda: warmup.LinearWarmup(optimizer, warmup_period=10))

        _test_optimizer(self, warmup.LinearWarmup)

        _test_get_warmup_params(self, optimizer, warmup.LinearWarmup)

    def test_exponential(self):
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.SGD([
                {'params': [p1]},
                {'params': [p2], 'lr': _set_lr(0.1)}
            ], lr=_set_lr(0.5))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=5)
        print()
        for step in range(1, 11):
            lr = [_get_lr(x['lr']) for x in optimizer.param_groups]
            print(f'{step} {lr}')
            self.assertAlmostEqual(lr[0], 0.5 * (1 - math.exp(-step / 5)))
            self.assertAlmostEqual(lr[1], 0.1 * (1 - math.exp(-step / 5)))
            optimizer.zero_grad()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        _test_state_dict(self, warmup_scheduler,
                         lambda: warmup.ExponentialWarmup(optimizer, warmup_period=10))

        _test_optimizer(self, warmup.ExponentialWarmup)

        _test_get_warmup_params(self, optimizer, warmup.ExponentialWarmup)

    def test_linear_chaining(self):
        def preparation():
            p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
            p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
            optimizer = torch.optim.SGD([
                    {'params': [p1]},
                    {'params': [p2], 'lr': _set_lr(0.1)}
                    ], lr=_set_lr(0.5))
            lr_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
            return optimizer, lr_scheduler1, lr_scheduler2

        # First record undampened lrs
        optimizer, lr_scheduler1, lr_scheduler2 = preparation()
        lrs = []
        print()
        print('Undampened:')
        for step in range(1, 11):
            lr = [_get_lr(x['lr']) for x in optimizer.param_groups]
            print(f'{step} {lr}')
            lrs.append(lr)
            optimizer.zero_grad()
            optimizer.step()
            lr_scheduler1.step()
            lr_scheduler2.step()

        optimizer, lr_scheduler1, lr_scheduler2 = preparation()
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=5)
        print('Dampened:')
        for step in range(1, 11):
            lr = [_get_lr(x['lr']) for x in optimizer.param_groups]
            print(f'{step} {lr}')
            if step < 5:
                self.assertAlmostEqual(lr[0], lrs[step-1][0] * step / 5)
                self.assertAlmostEqual(lr[1], lrs[step-1][1] * step / 5)
            else:
                self.assertAlmostEqual(lr[0], lrs[step-1][0])
                self.assertAlmostEqual(lr[1], lrs[step-1][1])
            optimizer.zero_grad()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler1.step()
                lr_scheduler2.step()

        _test_state_dict(self, warmup_scheduler,
                         lambda: warmup.LinearWarmup(optimizer, warmup_period=10))

    def test_linear_wo_lr_scheduler(self):
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.SGD([
                {'params': [p1]},
                {'params': [p2], 'lr': _set_lr(0.1)}
            ], lr=_set_lr(0.5))
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=5)
        print()
        for step in range(1, 11):
            lr = [_get_lr(x['lr']) for x in optimizer.param_groups]
            print(f'{step} {lr}')
            if step < 5:
                self.assertAlmostEqual(lr[0], 0.5 * step / 5)
                self.assertAlmostEqual(lr[1], 0.1 * step / 5)
            else:
                self.assertAlmostEqual(lr[0], 0.5)
                self.assertAlmostEqual(lr[1], 0.1)
            optimizer.zero_grad()
            optimizer.step()
            with warmup_scheduler.dampening():
                pass

        _test_state_dict(self, warmup_scheduler,
                         lambda: warmup.LinearWarmup(optimizer, warmup_period=10))
