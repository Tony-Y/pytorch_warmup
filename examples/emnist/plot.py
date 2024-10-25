import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main(args=None):
    parser = argparse.ArgumentParser(description='Training History Plot')
    parser.add_argument('path', metavar='PATH',
                        help='path to the output directory of the training script')
    parser.add_argument('--output', type=str, default='none',
                        choices=['none', 'png', 'pdf'],
                        help='output file type (default: none)')
    args = parser.parse_args(args)

    output_dir = args.path

    df_hist = pd.read_csv(os.path.join(output_dir, "history.csv"))
    df_eval = pd.read_csv(os.path.join(output_dir, "evaluation.csv"))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), layout="constrained")

    df_hist.plot(x="step", y="lr", ax=ax1, legend=False)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Learning rate')

    min_loss = df_hist.loss.min()
    df_hist.plot(x="step", y="loss", ylim=(0, 1) if min_loss < 0.8 else None, ax=ax2, legend=False)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')

    df_eval.plot(x="epoch", y="accuracy", marker="o", ax=ax3, legend=False)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    max_idx = df_eval.accuracy.argmax()
    max_epoch = df_eval.epoch[max_idx]
    max_acc = df_eval.accuracy[max_idx]
    min_acc = df_eval.accuracy.min()
    ax3.axvline(x=max_epoch, color='red', ls=':')
    ax3.text(x=max_epoch, y=min_acc, s=f'Max: {max_acc:.2f}% @ {max_epoch} ', ha='right', va='bottom')
    ax3.plot([0], [max_acc])

    if args.output == 'none':
        plt.show()
    else:
        file_path = os.path.join(output_dir, f'fig_history.{args.output}')
        plt.savefig(file_path)
        print(file_path)


if __name__ == '__main__':
    main()
