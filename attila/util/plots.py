import random
import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt


def get_figsize(n_rows, n_cols):
    row_size = 4    # heigth
    column_size = 10    # width

    return (n_cols * column_size, n_rows * row_size)


def get_figa(n_rows, n_cols):
    fig, ax = plt.subplots(n_rows, n_cols, figsize=get_figsize(n_rows, n_cols))
    return fig, ax


def plot_sample(X, y, cmap='nipy_spectral', ix=None, out_folder=None):
    if ix is None:
        ix = random.randint(0, len(X) - 1)

    fig, ax = get_figa(1, 3)

    im = ax[0].imshow(X[ix, ..., 0], cmap=cmap)
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title('image')

    ax[1].hist(X[ix, ...].ravel(), bins=256)
    ax[1].set_title('histogram of image')

    ax[2].imshow(y[ix].squeeze(), cmap='gray')
    ax[2].set_title('mask')

    fig.suptitle('sample #{}'.format(ix))
    if out_folder:
        fig.savefig(out_folder / 'sample_{}.png'.format(ix))

    plt.close(fig)

    return ix


def plot_history(experiments, last=None, out_folder=None):
    if last is None:
        last = 0

    n_cols = 2
    n_rows = int(np.ceil(len(experiments) / n_cols))
    fig, ax = get_figa(n_rows, n_cols)

    def _plot_key(ax, key, results, color, scale=None, find_min=False, find_max=False):
        training = results[key]
        validation = results['val_{}'.format(key)]

        if scale:
            ax.set_ylim(scale)

        ax.plot(training, label='training {}'.format(key), color=color)
        ax.plot(validation, '--', label='validation {}'.format(key), color=color)

        if find_min:
            ax.plot(np.argmin(validation), np.min(validation), marker='x', color='r')

        if find_max:
            ax.plot(np.argmax(validation), np.max(validation), marker='x', color='r')

    def _plot_results(results, ax, title):
        _plot_key(ax, 'loss', results, 'C1', scale=[0, 0.05], find_min=True)
        # ax.legend()

        ax = ax.twinx()    # instantiate a second axes that shares the same x-axis

        _plot_key(ax, 'mean_IoU', results, 'C0', scale=[0.9, 1], find_max=True)
        _plot_key(ax, 'DSC', results, 'C2', scale=[0.9, 1], find_max=True)

        # ax.legend()

        ax.set_title(title)

    for a, experiment in zip(ax.ravel(), experiments):
        history = experiment['history']
        results = {
            k: history[k][-last:]
            for k in history.keys()
        }

        _plot_results(results, a, experiment['name'])

    if out_folder:
        fig.savefig(out_folder / 'history.png')
        # tikzplotlib.save(out_folder / 'history.pgf')


def plot_preds(X, y, preds, cmap, title=None, out_folder=None):
    for ix in range(len(preds)):
        fig, ax = get_figa(1, 2)

        ax[0].imshow(X[ix, ..., 0], cmap=cmap)
        ax[0].set_title('input image (sample #{})'.format(ix))

        ax[1].imshow(preds[ix].squeeze(), cmap='gray')
        ax[1].contour(y[ix].squeeze(), colors='yellow', levels=[0.5])
        ax[1].set_title('prediction (ground truth as contour)')

        if title:
            fig.suptitle(title)

        if out_folder:
            fig.savefig(out_folder / '{}.pgf'.format(ix))
