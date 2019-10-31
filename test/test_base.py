import unittest
import math
import torch
import pytorch_warmup as warmup

class TestBase(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def test_linear(self):
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.SGD([
                {'params': [p1]},
                {'params': [p2], 'lr': 0.1}
            ], lr=0.5)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=5)
        for step in range(1,11):
            lr = [x['lr'] for x in optimizer.param_groups]
            print(f'{step} {lr}')
            if step < 5:
                self.assertAlmostEqual(lr[0], 0.5 * step / 5)
                self.assertAlmostEqual(lr[1], 0.1 * step / 5)
            else:
                self.assertAlmostEqual(lr[0], 0.5)
                self.assertAlmostEqual(lr[1], 0.1)
            optimizer.zero_grad()
            optimizer.step()
            lr_scheduler.step()
            warmup_scheduler.dampen()

    def test_exponetial(self):
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.SGD([
                {'params': [p1]},
                {'params': [p2], 'lr': 0.1}
            ], lr=0.5)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=5)
        for step in range(1,11):
            lr = [x['lr'] for x in optimizer.param_groups]
            print(f'{step} {lr}')
            self.assertAlmostEqual(lr[0], 0.5 * (1 - math.exp(-step / 5)))
            self.assertAlmostEqual(lr[1], 0.1 * (1 - math.exp(-step / 5)))
            optimizer.zero_grad()
            optimizer.step()
            lr_scheduler.step()
            warmup_scheduler.dampen()
