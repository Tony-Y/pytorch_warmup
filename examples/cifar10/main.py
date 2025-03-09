import sys
import os
import time
import argparse
from typing import Optional, TextIO, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import pytorch_warmup as warmup
from pytorch_warmup import BaseWarmup

try:
    import resnet
except ImportError:
    sys.exit("Download resnet.py from https://github.com/akamaster/pytorch_resnet_cifar10")

import torch.backends.cudnn as cudnn

cudnn.benchmark = True


architecture_names = ["resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet1202"]
algorithm_names = ["sgd", "adamw", "amsgradw", "nadamw", "adamax", "radamw"]
warmup_names = ["linear", "exponential", "radam", "none"]


# Type alias
Model = torch.nn.Module
Device = torch.device
DataLoaderKwargs = dict[str, Union[int, str, bool]]
OptimKwargs = dict[str, Union[float, bool, tuple[float, float], Tensor]]


class Config(argparse.Namespace):
    arch: str
    batch_size: int
    test_batch_size: int
    epochs: int
    milestones: list[int]
    algorithm: str
    lr: float
    weight_decay: float
    beta2: float
    warmup: str
    warmup_period: int
    workers: int
    seed: int
    log_interval: int
    output: str
    save_model: bool
    no_progress: bool
    no_gpu: bool
    compile: bool


def check_pytorch_version(algorithm: str) -> None:
    major, minor, patch = map(int, torch.__version__.split("+")[0].split("."))
    if major == 0 or (major == 1 and minor < 12):
        sys.exit("This script requires PyTorch 1.12+ or 2.x.")

    if algorithm == "nadamw" and (major == 1 or (major == 2 and minor < 1)):
        sys.exit("[Error] The NAdamW optimization algorithm requires PyTorch 2.1 or later.")
    elif algorithm == "radamw" and (major == 1 or (major == 2 and minor < 3)):
        sys.exit("[Error] The RAdamW optimization algorithm requires PyTorch 2.3 or later.")


def get_lr(optimizer: Optimizer) -> float:
    lr: Union[float, Tensor] = optimizer.param_groups[0]["lr"]
    return lr.item() if isinstance(lr, Tensor) else lr


def train_iter_loss_fn(
    optimizer: Optimizer,
    model: Model,
    data: Tensor,
    target: Tensor,
) -> Tensor:
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    return loss


def update_lr_fn(
    lr_scheduler: LRScheduler,
    warmup_scheduler: BaseWarmup,
) -> None:
    with warmup_scheduler.dampening():
        lr_scheduler.step()


def train(
    config: Config,
    model: Model,
    device: Device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    warmup_scheduler: BaseWarmup,
    epoch: int,
    history: TextIO,
) -> None:
    since = time.time()
    model.train()
    progress = tqdm(total=len(train_loader), disable=config.no_progress)
    progress.set_description(f"[train] Epoch {epoch}")
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        lr = get_lr(optimizer)
        data, target = data.to(device), target.to(device)
        batch_loss = train_iter_loss_fn(optimizer, model, data, target)
        update_lr_fn(lr_scheduler, warmup_scheduler)
        loss = batch_loss.item()
        train_loss += loss
        batch_step = batch_idx + 1
        if batch_step % config.log_interval == 0:
            step = warmup_scheduler.last_step
            history.write(f"{epoch},{step},{loss:g},{lr:g}\n")
        if not progress.disable:
            if batch_step % 10 == 0:
                progress.set_postfix_str(f"loss={loss:.2f}, lr={lr:5.4f}")
            progress.update()
    progress.close()

    train_loss /= len(train_loader)
    elapsed = time.time() - since
    print(f"[train] Epoch {epoch}: Elapsed Time: {elapsed:.3f} sec, Ave. Loss: {train_loss:.4f}")


def test_iter_loss_fn(model: Model, data: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    output = model(data)
    loss = F.cross_entropy(output, target, reduction="sum")
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max of unnormalized logits
    correct = pred.eq(target.view_as(pred)).sum()
    return loss, correct


@torch.inference_mode()
def test(
    config: Config,
    model: Model,
    device: Device,
    test_loader: DataLoader,
    epoch: int,
    evaluation: TextIO,
) -> float:
    since = time.time()
    model.eval()
    progress = tqdm(test_loader, disable=config.no_progress)
    progress.set_description(f"[test]  Epoch {epoch}")
    test_loss = 0.0
    correct = 0
    for data, target in progress:
        data, target = data.to(device), target.to(device)
        batch_loss, batch_correct = test_iter_loss_fn(model, data, target)
        test_loss += batch_loss.item()  # sum up batch loss
        correct += batch_correct.item()  # type: ignore[assignment]  # sum up batch correct

    test_loss /= len(test_loader.dataset)  # type: ignore[operator,arg-type]
    test_acc: float = 100.0 * correct / len(test_loader.dataset)  # type: ignore[operator,arg-type]
    elapsed = time.time() - since
    print(
        f"[test]  Epoch {epoch}: Elapsed Time: {elapsed:.3f} sec, "
        f"Ave. Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%"
    )
    evaluation.write(f"{epoch},{test_loss:g},{test_acc:.2f}\n")
    evaluation.flush()
    return test_acc


def gpu_device() -> Device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def dataloader_options(device: Device, workers: int) -> DataLoaderKwargs:
    if device.type == "cpu":
        return {}

    kwargs: DataLoaderKwargs = dict(num_workers=workers, pin_memory=True)
    if workers > 0:
        if device.type == "mps":
            kwargs.update(dict(multiprocessing_context="forkserver", persistent_workers=True))
        else:
            kwargs.update(dict(persistent_workers=True))
    return kwargs


def optimization_algorithm(config: Config, model: Model, device: Device) -> Optimizer:
    name = config.algorithm
    lr = torch.tensor(config.lr).to(device) if config.compile else config.lr
    kwargs: OptimKwargs = dict(lr=lr, weight_decay=config.weight_decay)
    if name == "sgd":
        kwargs["momentum"] = 0.9
    else:
        kwargs["betas"] = (0.9, config.beta2)

    if name == "sgd":
        return optim.SGD(model.parameters(), **kwargs)  # type: ignore[arg-type]
    elif name == "adamw":
        return optim.AdamW(model.parameters(), **kwargs)  # type: ignore[arg-type]
    elif name == "amsgradw":
        return optim.AdamW(model.parameters(), amsgrad=True, **kwargs)  # type: ignore[arg-type]
    elif name == "nadamw":
        return optim.NAdam(model.parameters(), decoupled_weight_decay=True, **kwargs)  # type: ignore[arg-type]
    elif name == "adamax":
        return optim.Adamax(model.parameters(), **kwargs)  # type: ignore[arg-type]
    elif name == "radamw":
        return optim.RAdam(model.parameters(), decoupled_weight_decay=True, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"unknown optimization algorithm: {name}")


def warmup_schedule(optimizer: Optimizer, name: str, period: int) -> BaseWarmup:
    if name == "linear":
        if period == 0:
            return warmup.UntunedLinearWarmup(optimizer)
        else:
            return warmup.LinearWarmup(optimizer, period)
    elif name == "exponential":
        if period == 0:
            return warmup.UntunedExponentialWarmup(optimizer)
        else:
            return warmup.ExponentialWarmup(optimizer, period)
    elif name == "radam":
        return warmup.RAdamWarmup(optimizer)
    elif name == "none":
        return warmup.LinearWarmup(optimizer, 1)
    else:
        raise ValueError(f"unknown warmup schedule: {name}")


def compile_functions() -> None:
    global train_iter_loss_fn
    global test_iter_loss_fn
    train_iter_loss_fn = torch.compile(train_iter_loss_fn, mode="reduce-overhead")
    test_iter_loss_fn = torch.compile(test_iter_loss_fn, mode="reduce-overhead")


def init_momentum_buffer(optimizer: Optimizer) -> None:
    for group in optimizer.param_groups:
        if group["momentum"] != 0:
            for p in group["params"]:
                state = optimizer.state[p]
                if state.get("momentum_buffer") is None:
                    state["momentum_buffer"] = torch.zeros_like(p.data)


def main(args: Optional[list[str]] = None) -> None:
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Example")
    # fmt: off
    parser.add_argument("-r", "--arch", type=str, default="resnet20", metavar="ARCH",
                        choices=architecture_names,
                        help="ResNet architecture for CIFAR10: " +
                             " | ".join(architecture_names) + " (default: resnet20)")
    parser.add_argument("-b", "--batch-size", type=int, default=128, metavar="BS",
                        help="input batch size for training (default: 128)")
    parser.add_argument("-c", "--test-batch-size", type=int, default=1000, metavar="BS",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("-e", "--epochs", type=int, default=186, metavar="NE",
                        help="number of epochs to train (default: 186)")
    parser.add_argument("-m", "--milestones", type=int, nargs="+", default=[81, 122], metavar="M",
                        help="MultiStepLR's milestones in epoch (default: [81, 122])")
    parser.add_argument("-a", "--algorithm", type=str, default="sgd", metavar="ALGO",
                        choices=algorithm_names,
                        help="optimization algorithm: " +
                             " | ".join(algorithm_names) + " (default: sgd)")
    parser.add_argument("-l", "--lr", type=float, default=0.1, metavar="LR",
                        help="base learning rate (default: 0.1)")
    parser.add_argument("-d", "--weight-decay", type=float, default=0.0001, metavar="WD",
                        help="weight decay (default: 0.0001)")
    parser.add_argument("-g", "--beta2", type=float, default=0.999, metavar="B2",
                        help="Adam's beta2 parameter (default: 0.999)")
    parser.add_argument("-w", "--warmup", type=str, default="none", metavar="WU",
                        choices=warmup_names,
                        help="warmup schedule: " +
                             " | ".join(warmup_names) + " (default: none)")
    parser.add_argument("-t", "--warmup-period", type=int, default=0, metavar="TAU",
                        help="linear warmup period or exponential warmup constant. " +
                             "Set 0 to use the untuned linear or exponential warmup. (default: 0)")
    parser.add_argument("-n", "--workers", type=int, default=0, metavar="NW",
                        help="number of dataloader workers for GPU training (default: 0)")
    parser.add_argument("-s", "--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("-i", "--log-interval", type=int, default=10, metavar="I",
                        help="how many batches to wait before logging training status")
    parser.add_argument("-o", "--output", default="output", metavar="PATH",
                        help="path to output directory (default: output)")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="for saving the best model")
    parser.add_argument("--no-progress", action="store_true", default=False,
                        help="disable progress bar")
    parser.add_argument("--no-gpu", action="store_true", default=False,
                        help="disable GPU training. " +
                             "As default, an MPS or CUDA device will be used if available.")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="optimize PyTorch code using TorchDynamo, AOTAutograd, and TorchInductor")
    # fmt: on
    config = parser.parse_args(args, namespace=Config())

    check_pytorch_version(config.algorithm)

    print(config)
    device = torch.device("cpu") if config.no_gpu else gpu_device()
    print(f"Device: {device.type}")

    torch.manual_seed(config.seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    kwargs = dataloader_options(device, config.workers)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs,  # type: ignore[arg-type]
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=config.test_batch_size,
        shuffle=False,
        **kwargs,  # type: ignore[arg-type]
    )

    output_dir = config.output
    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        sys.exit(f"[Error] File exists: {output_dir}")

    history = open(os.path.join(output_dir, "history.csv"), "w")
    history.write("epoch,step,loss,lr\n")

    evaluation = open(os.path.join(output_dir, "evaluation.csv"), "w")
    evaluation.write("epoch,loss,accuracy\n")

    model = resnet.__dict__[config.arch]().to(device)

    optimizer = optimization_algorithm(config, model, device)

    steps_per_epoch = len(train_loader)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[i * steps_per_epoch for i in config.milestones]
    )

    warmup_scheduler = warmup_schedule(optimizer, name=config.warmup, period=config.warmup_period)

    if config.compile:
        if config.algorithm == "sgd":
            init_momentum_buffer(optimizer)
        compile_functions()

    best_acc = 0.0
    best_epoch = 0
    print()
    for epoch in range(1, config.epochs + 1):
        train(config, model, device, train_loader, optimizer, lr_scheduler,
              warmup_scheduler, epoch, history)  # fmt: skip
        cur_acc = test(config, model, device, test_loader, epoch, evaluation)

        if cur_acc > best_acc:
            best_acc = cur_acc
            best_epoch = epoch
            if config.save_model:
                torch.save(
                    model.state_dict(), os.path.join(output_dir, f"cifar10_{config.arch}.pt")
                )

    print(f"The best accuracy: {best_acc:.2f}% (epoch {best_epoch})")

    history.close()
    evaluation.close()


if __name__ == "__main__":
    main()
