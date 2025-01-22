# A PyTorch Extension for Learning Rate Warmup

This library contains PyTorch implementations of the warmup schedules described in [On the adequacy of untuned warmup for adaptive optimization](https://arxiv.org/abs/1910.04209).

<p align="center"><img src="https://github.com/Tony-Y/pytorch_warmup/raw/master/examples/plots/figs/warmup_schedule.png" alt="Warmup schedule" width="400"/></p>

[![Python package](https://github.com/Tony-Y/pytorch_warmup/workflows/Python%20package/badge.svg)](https://github.com/Tony-Y/pytorch_warmup/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/pytorch-warmup.svg)](https://pypi.python.org/pypi/pytorch-warmup/)
[![PyPI license](https://img.shields.io/pypi/l/pytorch-warmup.svg)](https://github.com/Tony-Y/pytorch_warmup/blob/master/LICENSE)
[![Python versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)

## Installation

Make sure you have Python 3.9+ and PyTorch 1.9+ or 2.x. Then, run the following command in the project directory:

```shell
python -m pip install .
```

or install the latest version from the Python Package Index:

```shell
pip install -U pytorch_warmup
```

## Examples

* [CIFAR10](https://github.com/Tony-Y/pytorch_warmup/tree/master/examples/cifar10) -
 A sample script to train a ResNet model on the CIFAR10 dataset using an optimization algorithm with a warmup schedule.
 Its README presents ResNet20 results obtained using each of AdamW, NAdamW, AMSGradW, and AdaMax
 together with each of various warmup schedules.
 In addition, there is a ResNet performance comparison (up to ResNet110) obtained using the SGD algorithm
 with a linear warmup schedule.
* [EMNIST](https://github.com/Tony-Y/pytorch_warmup/tree/master/examples/emnist) -
 A sample script to train a CNN model on the EMNIST dataset using the AdamW algorithm with a warmup schedule.
 Its README presents a result obtained using the AdamW algorithm with each of the untuned linear and exponential warmup,
 and the RAdam warmup.
* [Plots](https://github.com/Tony-Y/pytorch_warmup/tree/master/examples/plots) -
 A script to plot effective warmup periods as a function of &beta;&#8322;, and warmup schedules over time.

## Usage

The [documentation](https://tony-y.github.io/pytorch_warmup/master/) provides more detailed information on this library, unseen below. 

### Sample Codes

The scheduled learning rate is dampened by the multiplication of the warmup factor:

<p align="center"><img src="https://github.com/Tony-Y/pytorch_warmup/raw/master/examples/emnist/figs/learning_rate.png" alt="Learning rate" width="400"/></p>

#### Approach 1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tony-Y/colab-notebooks/blob/master/PyTorch_Warmup_Approach1_chaining.ipynb)

When the learning rate schedule uses the global iteration number, the untuned linear warmup can be used
together with `Adam` or its variant (`AdamW`, `NAdam`, etc.) as follows:

```python
import torch
import pytorch_warmup as warmup

optimizer = torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    # This sample code uses the AdamW optimizer.
num_steps = len(dataloader) * num_epochs
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    # The LR schedule initialization resets the initial LR of the optimizer.
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    # The warmup schedule initialization dampens the initial LR of the optimizer.
for epoch in range(1,num_epochs+1):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()
        with warmup_scheduler.dampening():
            lr_scheduler.step()
```

> [!Warning]
> Note that the warmup schedule must not be initialized before the initialization of the learning rate schedule.

If you want to use the learning rate schedule *chaining*, which is supported for PyTorch 1.4 or above, you may simply write a code of learning rate schedulers as a suite of the `with` statement:

```python
lr_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
for epoch in range(1,num_epochs+1):
    for batch in dataloader:
        ...
        optimizer.step()
        with warmup_scheduler.dampening():
            lr_scheduler1.step()
            lr_scheduler2.step()
```

If you want to start the learning rate schedule after the end of the linear warmup, delay it by the warmup period:

```python
warmup_period = 2000
num_steps = len(dataloader) * num_epochs - warmup_period
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
for epoch in range(1,num_epochs+1):
    for batch in dataloader:
        ...
        optimizer.step()
        with warmup_scheduler.dampening():
            if warmup_scheduler.last_step + 1 >= warmup_period:
                lr_scheduler.step()
```

#### Approach 2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tony-Y/colab-notebooks/blob/master/PyTorch_Warmup_Approach2_chaining.ipynb)

When the learning rate schedule uses the epoch number, the warmup schedule can be used as follows:

```python
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs//3], gamma=0.1)
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
for epoch in range(1,num_epochs+1):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()
        if i < len(dataloader)-1:
            with warmup_scheduler.dampening():
                pass
    with warmup_scheduler.dampening():
        lr_scheduler.step()
```

This code can be rewritten more compactly:

```python
for epoch in range(1,num_epochs+1):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()
        with warmup_scheduler.dampening():
            if i + 1 == len(dataloader):
                lr_scheduler.step()
```

#### Approach 3

When you use `CosineAnnealingWarmRestarts`, the warmup schedule can be used as follows:

```python
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
warmup_period = 2000
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
iters = len(dataloader)
warmup_epochs = ... # for example, (warmup_period + iters - 1) // iters
for epoch in range(epochs+warmup_epochs):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        optimizer.step()
        with warmup_scheduler.dampening():
            if epoch >= warmup_epochs:
                lr_scheduler.step(epoch-warmup_epochs + i / iters)
```

### Warmup Schedules

#### Manual Warmup

In `LinearWarmup` and `ExponentialWarmup`, the warmup factor `w(t)` depends on the warmup period that must manually be specified.

##### Linear

`w(t) = min(1, t / warmup_period)`

```python
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=2000)
```

For details please refer to [LinearWarmup](https://tony-y.github.io/pytorch_warmup/master/manual_warmup.html#pytorch_warmup.base.LinearWarmup) in the documentation.

##### Exponential

`w(t) = 1 - exp(-t / warmup_period)`

```python
warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=1000)
```

For details please refer to [ExponentialWarmup](https://tony-y.github.io/pytorch_warmup/master/manual_warmup.html#pytorch_warmup.base.ExponentialWarmup) in the documentation.

#### Untuned Warmup

In `UntunedLinearWarmup` and `UntunedExponentialWarmup`, the warmup period is determined by a function of Adam's `beta2` parameter.

##### Linear

`warmup_period = 2 / (1 - beta2)`

```python
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
```

For details please refer to [UntunedLinearWarmup](https://tony-y.github.io/pytorch_warmup/master/untuned_warmup.html#pytorch_warmup.untuned.UntunedLinearWarmup) in the documentation.

##### Exponential

`warmup_period = 1 / (1 - beta2)`

```python
warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
```

For details please refer to [UntunedExponentialWarmup](https://tony-y.github.io/pytorch_warmup/master/untuned_warmup.html#pytorch_warmup.untuned.UntunedExponentialWarmup) in the documentation.

#### RAdam Warmup

In `RAdamWarmup`, the warmup factor `w(t)` is a complicated function depending on Adam's `beta2` parameter.

```python
warmup_scheduler = warmup.RAdamWarmup(optimizer)
```

For details please refer to [RAdamWarmup](https://tony-y.github.io/pytorch_warmup/master/radam_warmup.html#pytorch_warmup.radam.RAdamWarmup) in the documentation, or
"[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)."

### Apex's Adam

The Apex library provides an Adam optimizer tuned for CUDA devices, [FusedAdam](https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedAdam). The FusedAdam optimizer can be used together with any one of the warmup schedules above. For example:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tony-Y/colab-notebooks/blob/master/PyTorch_Warmup_FusedAdam.ipynb)

```python
optimizer = apex.optimizers.FusedAdam(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
```

### Compiled Optimizers

[Benchmarking results](https://dev-discuss.pytorch.org/t/performance-comparison-between-torch-compile-and-apex-optimizers/2023)
show that the complied Adam outperforms the Apex's Adam.

> [!Warning]
> PyTorch 2.3 or later is required for using the compiled optimizer with
> a warmup scheduler and/or LR schedulers.
> PyTorch-Warmup 0.2 or earlier is incompatible with the complied optimizer.

You can compile the Adam optimizer as follows:

```python
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(0.001).to(device))
opt_step = torch.compile(optimizer.step, mode="reduce-overhead")
```

> [!Important]
> Wrap the learning rate in a `Tensor`, or `torch.compile` will recompile
> as the value of the learning rate changes.

Then, the compiled version `opt_step` have to be invoked instead of `optimizer.step`:

```python
for epoch in range(1,num_epochs+1):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = ...
        loss.backward()
        opt_step()
        with warmup_scheduler.dampening():
            lr_scheduler.step()
```

You can also compile other built-in optimizers in the way shown above.

> [!Note]
> When using the compiled SGD with momentum, its momentum buffer is needed
> to be initialized manually. You can find sample code in the CIFAR10 exmaple.

In practice, you may compile it together with other PyTorch code as follows:

```python
@torch.compile(mode="reduce-overhead")
def train_iter_fn(batch):
    optimizer.zero_grad()
    loss = ...
    loss.backward()
    optimizer.step()

for epoch in range(1,num_epochs+1):
    for batch in dataloader:
        train_iter_fn(batch)
        with warmup_scheduler.dampening():
            lr_scheduler.step()
```

`torch.compile` skips `lr_scheduler.step` even if it were invoked within `train_iter_fn`.
Likewise, you should not compile `warmup_scheduler.dampening`.
You may also use `torch.compiler.disable` to have `torch.compile` skip a function
updating the learning rate as follows:

```python
@torch.compiler.disable
def update_lr_fn():
    with warmup_scheduler.dampening():
        lr_scheduler.step()

@torch.compile(mode="reduce-overhead")
def train_iter_fn(batch):
    optimizer.zero_grad()
    loss = ...
    loss.backward()
    optimizer.step()
    update_lr_fn()

for epoch in range(1,num_epochs+1):
    for batch in dataloader:
        train_iter_fn(batch)
```

## License

MIT License

&copy; 2019-2025 Takenori Yamamoto
