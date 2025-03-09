import argparse
from typing import Union
import matplotlib.pyplot as plt
import torch
from pytorch_warmup import RAdamWarmup, UntunedExponentialWarmup, UntunedLinearWarmup


WarmupClass = Union[type[RAdamWarmup], type[UntunedExponentialWarmup], type[UntunedLinearWarmup]]


def get_rates(warmup_cls: WarmupClass, beta2: float, max_step: int) -> list[float]:
    rates: list[float] = []
    p = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
    optimizer = torch.optim.Adam([{"params": p}], lr=1.0, betas=(0.9, beta2))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    warmup_scheduler = warmup_cls(optimizer)
    for step in range(1, max_step + 1):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()
        warmup_scheduler.dampen()
    return rates


parser = argparse.ArgumentParser(description="Warmup schedule")
# fmt: off
parser.add_argument("--output", type=str, default="none",
                    choices=["none", "png", "pdf"],
                    help="Output file type (default: none)")
# fmt: on
args = parser.parse_args()

beta2 = 0.999
max_step = 3000

plt.plot(
    range(1, max_step + 1),
    get_rates(RAdamWarmup, beta2, max_step),
    label="RAdam",
)
plt.plot(
    range(1, max_step + 1),
    get_rates(UntunedExponentialWarmup, beta2, max_step),
    label="Untuned Exponential",
)
plt.plot(
    range(1, max_step + 1),
    get_rates(UntunedLinearWarmup, beta2, max_step),
    label="Untuned Linear",
)
plt.legend()
plt.title("Warmup Schedule")
plt.xlabel("Iteration")
plt.ylabel(r"Warmup factor $(\omega_t)$")
if args.output == "none":
    plt.show()
else:
    plt.savefig(f"warmup_schedule.{args.output}")
