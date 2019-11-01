import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import pytorch_warmup as warmup
import os


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
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        lr = optimizer.param_groups[0]['lr']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        warmup_scheduler.dampen()
        if (batch_idx+1) % args.log_interval == 0:
            loss = loss.item()
            step = warmup_scheduler.last_step
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} LR: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader) * len(data),
                100. * (batch_idx+1) / len(train_loader), loss, lr))
            history.write(f'{epoch},{step},{loss},{lr}\n')


def test(args, model, device, test_loader, epoch, evaluation):
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    evaluation.write(f'{epoch},{test_loss},{test_acc}\n')
    evaluation.flush()


def main():
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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--warmup', type=str, default='linear',
                        choices=['linear', 'exponential', 'radam', 'none'],
                        help='warmup schedule')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('.data', 'balanced', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1751,), (0.3332,))
                        ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.EMNIST('.data', 'balanced', train=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1751,), (0.3332,))
                        ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    output_dir = args.warmup
    os.makedirs(output_dir, exist_ok=True)

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
    if args.warmup == 'linear':
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif args.warmup == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif args.warmup == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
    elif args.warmup == 'none':
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, lr_scheduler,
              warmup_scheduler, epoch, history)
        test(args, model, device, test_loader, epoch, evaluation)

    if (args.save_model):
        torch.save(model.state_dict(), os.path.join(output_dir, "emnist_cnn.pt"))

    history.close()
    evaluation.close()


if __name__ == '__main__':
    main()
