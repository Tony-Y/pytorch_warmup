import unittest
import math
import torch
import pytorch_warmup as warmup

from .test_base import _test_state_dict, _set_lr, _get_lr
from .test_untuned import _test_optimizer


# The expression of each warmup factor
# offset: 6
# beta2: 7/10
# rho_inf: 17/3
ewf = {
    1: 3*math.sqrt(55795775461652507765)/126458170465,
    2: 3*math.sqrt(118975550786574877912913153615)/2269508758199815,
    3: 3*math.sqrt(364009685132320107701159663753)/2992977113632385,
    4: 3*math.sqrt(258572826689968392763003617038979)/68225651323259287,
    5: 3*math.sqrt(668289519821298522824847043230807053)/3138599717744915303,
    6: 3*math.sqrt(60431582784117573249154657184784100939048735)/27879860688339331112605,
    7: 3*math.sqrt(3668869686599344602586804992292010752258094185)/207030521845988349697045,
    8: 3*math.sqrt(38610293903545800493859693214989542002076301625518651)/648745826249577848268496415,
    9: 3*math.sqrt(12026080263946093429637752207887183661294840713819813)/353035787321509409011021039,
    10: 3*math.sqrt(456865593113897246792694328842050091932272202605586577311)/67546148486329926220639511801,
}


class TestRAdam(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def test_radam(self):
        p1 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        p2 = torch.nn.Parameter(torch.arange(10, dtype=torch.float32).to(self.device))
        optimizer = torch.optim.Adam([
                {'params': [p1]},
                {'params': [p2], 'lr': _set_lr(0.1)}
            ], lr=_set_lr(0.5), betas=(0.9, 0.7))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
        print()
        for step in range(1, 11):
            lr = [_get_lr(x['lr']) for x in optimizer.param_groups]
            print(f'{step} {lr}')
            self.assertAlmostEqual(lr[0], 0.5 * ewf[step])
            self.assertAlmostEqual(lr[1], 0.1 * ewf[step])
            optimizer.zero_grad()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

        _test_state_dict(self, warmup_scheduler,
                         lambda: warmup.RAdamWarmup(optimizer))

        _test_optimizer(self, warmup.RAdamWarmup)
