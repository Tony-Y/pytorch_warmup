import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import pytorch_warmup as warmup
import os
import sys
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, lr_scheduler,
          warmup_scheduler, epoch, history):
    since = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        lr = optimizer.param_groups[0]['lr']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        with warmup_scheduler.dampening():
            lr_scheduler.step()
        if (batch_idx+1) % args.log_interval == 0:
            loss = loss.item()
            step = warmup_scheduler.last_step
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} LR: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader) * len(data),
                100. * (batch_idx+1) / len(train_loader), loss, lr))
            history.write(f'{epoch},{step},{loss:g},{lr:g}\n')
    print('Train Elapsed Time: {:.3f} sec'.format(time.time()-since))


def test(args, model, device, test_loader, epoch, evaluation):
    since = time.time()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    evaluation.write(f'{epoch},{test_loss:g},{test_acc:.2f}\n')
    evaluation.flush()
    print('Test Elapsed Time: {:.3f} sec\n'.format(time.time()-since))


def mps_is_available():
    try:
        return torch.backends.mps.is_available()
    except AttributeError:
        return False


def gpu_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif mps_is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def dataloader_options(device, workers):
    if device.type == 'cpu':
        return {}

    kwargs = dict(num_workers=workers, pin_memory=True)
    if workers > 0:
        if device.type == 'mps':
            kwargs.update(dict(multiprocessing_context="forkserver", persistent_workers=True))
        else:
            kwargs.update(dict(persistent_workers=True))
    return kwargs


def warmup_schedule(optimizer, name):
    if name == 'linear':
        return warmup.UntunedLinearWarmup(optimizer)
    elif name == 'exponential':
        return warmup.UntunedExponentialWarmup(optimizer)
    elif name == 'radam':
        return warmup.RAdamWarmup(optimizer)
    elif name == 'none':
        return warmup.LinearWarmup(optimizer, 1)


def main(args=None):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch EMNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='base learning rate (default: 0.01)')
    parser.add_argument('--lr-min', type=float, default=1e-5, metavar='LM',
                        help='minimum learning rate (default: 1e-5)')
    parser.add_argument('--wd', type=float, default=0.01, metavar='WD',
                        help='weight decay (default: 0.01)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help="Adam's beta2 parameter (default: 0.999)")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--warmup', type=str, default='linear',
                        choices=['linear', 'exponential', 'radam', 'none'],
                        help='warmup schedule')
    parser.add_argument('--workers', type=int, default=0, metavar='N',
                        help='number of dataloader workers for GPU training (default: 0)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for saving the current model')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disable GPU training. ' +
                             'As default, an MPS or CUDA device will be used if available.')
    args = parser.parse_args(args)

    print(args)
    device = torch.device('cpu') if args.no_gpu else gpu_device()
    print(f'Device: {device.type}')

    torch.manual_seed(args.seed)

    kwargs = dataloader_options(device, args.workers)
    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('data', 'balanced', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1751,), (0.3332,))
                        ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('data', 'balanced', train=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1751,), (0.3332,))
                        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    output_dir = f'output_{args.warmup}'
    try:
        os.makedirs(output_dir, exist_ok=False)
    except FileExistsError:
        sys.exit(f'[Error] File exists: {output_dir}')

    history = open(os.path.join(output_dir, 'history.csv'), 'w')
    history.write('epoch,step,loss,lr\n')

    evaluation = open(os.path.join(output_dir, 'evaluation.csv'), 'w')
    evaluation.write('epoch,loss,accuracy\n')

    model = Net().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, args.beta2),
                            weight_decay=args.wd)
    num_steps = len(train_loader) * args.epochs
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=args.lr_min)
    warmup_scheduler = warmup_schedule(optimizer, args.warmup)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, lr_scheduler,
              warmup_scheduler, epoch, history)
        test(args, model, device, test_loader, epoch, evaluation)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(output_dir, "emnist_cnn.pt"))

    history.close()
    evaluation.close()


if __name__ == '__main__':
    main()
