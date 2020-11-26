import random
import numpy as np
from matplotlib import pyplot as plt


def get_figsize(n_rows, n_cols):
    row_size = 4  # heigth
    column_size = 10  # width

    return (n_cols * column_size, n_rows * row_size)


def get_figa(n_rows, n_cols):
    fig, ax = plt.subplots(n_rows, n_cols, figsize=get_figsize(n_rows, n_cols))
    return fig, ax


def get_mask(masks, ix):
    background = masks[ix, ..., 0]
    foreground = masks[ix, ..., 1]
    borders = masks[ix, ..., 2]

    return background, foreground, borders


def plot_sample(X, y, cmap='magma', ix=None, out_folder=None):
    if ix is None:
        ix = random.randint(0, len(X) - 1)


    img = X[ix, ...].squeeze()

    im = plt.imshow(img, cmap=cmap)
    plt.gcf().colorbar(im, ax=plt.gca())
    if out_folder:
        plt.gcf().savefig(out_folder / 'sample_input_{}.png'.format(ix))
        plt.close()

    plt.gca().hist(img.ravel(), bins=256)
    if out_folder:
        plt.gcf().savefig(out_folder / 'sample_hist_{}.png'.format(ix))
        plt.close()

    _, foreground, borders = get_mask(y, ix)
    plt.gca().imshow(foreground, cmap='gray')
    plt.gca().contour(borders, colors='red', levels=[0.5])
    if out_folder:
        plt.gcf().savefig(out_folder / 'sample_mask_{}.png'.format(ix))
        plt.close()

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
        _plot_key(ax, 'loss', results, 'C1', scale=[0, 0.02], find_min=True)
        # ax.legend() 

        ax = ax.twinx()  # instantiate a second axes that shares the same x-axis

        _plot_key(ax, 'batch_metric-mean_IoU', results, 'C0', scale=[0.5, 1], find_max=True)
        _plot_key(ax, 'batch_metric-mean_DSC', results, 'C2', scale=[0.5, 1], find_max=True)

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
        plt.close()


def plot_preds(X, y, preds, cmap, title=None, out_folder=None):
    pred_count = preds.shape[0]
    how_many = min(pred_count, 8)

    for _ in range(how_many):
        ix = random.randint(0, len(X) - 1)

        plt.imshow(X[ix, ..., 0], cmap=cmap)
        if out_folder:
            plt.gcf().savefig(out_folder / 'ground_truth_{}.png'.format(ix))
            plt.close()

        _, ground_truth_foreground, ground_truth_borders = get_mask(y, ix)
        ground_truth = ground_truth_foreground + ground_truth_borders

        _, pred_foreground, pred_borders = get_mask(preds, ix)

        plt.imshow(pred_foreground, cmap='gray')
        plt.contour(ground_truth, colors='red', levels=[0.5])
        if out_folder:
            plt.gcf().savefig(out_folder / 'pred_{}.png'.format(ix))
            plt.close()
