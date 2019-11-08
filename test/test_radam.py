import unittest
import torch
import pytorch_warmup as warmup

from .test_base import _test_state_dict


class TestRAdam(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def test_radam(self):
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.Adam([
                {'params': [p1]},
                {'params': [p2], 'lr': 0.1}
            ], lr=0.5, betas=(0.9, 0.7))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
        for step in range(1, 11):
            lr = [x['lr'] for x in optimizer.param_groups]
            print(f'{step} {lr}')
            self.assertLess(lr[0], 0.5)
            self.assertLess(lr[1], 0.1)
            optimizer.zero_grad()
            optimizer.step()
            lr_scheduler.step()
            warmup_scheduler.dampen()

        _test_state_dict(self, warmup_scheduler,
                         lambda: warmup.RAdamWarmup(optimizer))
