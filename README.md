# A PyTorch Extension for Learning Rate Warmup

This library contains PyTorch implementations of the warmup schedules described in [On the adequacy of untuned warmup for adaptive optimization](https://arxiv.org/abs/1910.04209).

<p align="center"><img src="examples/plots/figs/warmup_schedule.png" alt="Warmup schedule" width="400"/></p>

## Installation

Make sure you have Python 3.6+ and PyTorch 1.1 or newer. Then, run the following command:

```
python setup.py install
```

## Usage

### Sample Codes

The scheduled learning rate is dampened by the multiplication of the warmup factor:

<p align="center"><img src="examples/emnist/figs/learning_rate.png" alt="Learning rate" width="400"/></p>

#### Approach 1

When the learning rate schedule uses the global iteration number, the untuned linear warmup can be used as follows:

```python
import torch
import pytorch_warmup as warmup

optimizer = torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
num_steps = len(dataloader) * num_epochs
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
for epoch in range(1,num_epochs+1):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        warmup_scheduler.dampen()
```

#### Approach 2

When the learning rate schedule uses the epoch number, the warmup schedule can be used as follows (for PyTorch 1.2 or above):

```python
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs//3], gamma=0.1)
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
warmup_scheduler.last_step = -1 # initialize the step counter
for epoch in range(1,num_epochs+1):
    for batch in dataloader:
        lr_scheduler.step(epoch-1)
        warmup_scheduler.dampen()
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()
```

The user warning about calling `lr_scheduler.step()` before `optimizer.step()` may be ignored.

### Warmup Schedules

#### Manual Warmup

The warmup factor `w(t)` depends on the warmup period, which must manually be specified, for `LinearWarmup` and `ExponentialWarmup`.

##### Linear

`w(t) = min(1, t / warmup_period)`

```python
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=2000)
```

##### Exponential

`w(t) = 1 - exp(-t / warmup_period)`

```python
warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=1000)
```

#### Untuned Warmup

The warmup period is given by a function of Adam's `beta2` parameter for `UntunedLinearWarmup` and `UntunedExponentialWarmup`.

##### Linear

`warmup_period = 2 / (1 - beta2)`

```python
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
```

##### Exponential

`warmup_period = 1 / (1 - beta2)`

```python
warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
```

#### RAdam Warmup

The warmup factor depends on Adam's `beta2` parameter for `RAdamWarmup`. Please see the original paper for the details.

```python
warmup_scheduler = warmup.RAdamWarmup(optimizer)
```

### Apex's Adam

The Apex library provides an Adam optimizer tuned for CUDA devices, [FusedAdam](https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedAdam). The FusedAdam optimizer can be used with the warmup schedulers. For example:

```python
optimizer = apex.optimizers.FusedAdam(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
```


## License

MIT License

Copyright (c) 2019 Takenori Yamamoto
