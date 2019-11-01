import argparse
import matplotlib.pyplot as plt
from pytorch_warmup import *
import numpy as np


def untuned_exponential_period(beta2):
    return 1.0 / (np.exp(1.0 - beta2) - 1.0)


def untuned_linear_period(beta2):
    return 1.0 / (1.0 - beta2) - 0.5


def warmup_factor(step, beta2, rho_inf, offset):
    rho = rho_fn(step+offset, beta2, rho_inf)
    numerator = (rho - 4) * (rho - 2) * rho_inf
    denominator = (rho_inf - 4) * (rho_inf - 2) * rho
    return np.sqrt(numerator/denominator)


def radam_period_fn(beta2):
    rho_inf = rho_inf_fn(beta2)
    offset = get_offset(beta2, rho_inf)
    steps = np.arange(1, 101)
    w = warmup_factor(steps, beta2, rho_inf, offset)
    total_sum = np.sum(1-w)
    t = 1
    while True:
        steps = np.arange(100*t+1, 100*(t+1)+1)
        w = warmup_factor(steps, beta2, rho_inf, offset)
        partial_sum = np.sum(1-w)
        if partial_sum < 0.1:
            break
        total_sum += partial_sum
        t += 1
    return total_sum


def radam_period(beta2):
    return [radam_period_fn(x) for x in beta2]


parser = argparse.ArgumentParser(description='Effective warmup period')
parser.add_argument('--output', type=str, default='none',
                    choices=['none', 'png', 'pdf'],
                    help='Output file type (default: none)')
args = parser.parse_args()

beta2 = np.arange(0.99, 0.9999, 0.0001)
plt.plot(beta2, untuned_exponential_period(beta2), label='Untuned Exponential')
plt.plot(beta2, untuned_linear_period(beta2), linestyle=':', label='Untuned Linear')
plt.plot(beta2, radam_period(beta2), linestyle='--', label='RAdam')
plt.xlim(0.990, 1.00)
plt.ylim(100, 10000)
plt.yscale('log')
plt.legend()
plt.title('Effective Warmup Period')
plt.xlabel(r'$\beta_{2}$')
plt.ylabel(r'${\cal T}(\omega)$')
if args.output == 'none':
    plt.show()
else:
    plt.savefig(f'warmup_period.{args.output}')
