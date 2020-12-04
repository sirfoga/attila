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
    foreground = masks[ix, ..., 0]
    borders = masks[ix, ..., 1]

    return foreground, borders


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

    foreground, borders = get_mask(y, ix)
    plt.gca().imshow(foreground, cmap='gray')
    plt.gca().contour(borders, colors='red', levels=[0.5])
    if out_folder:
        plt.gcf().savefig(out_folder / 'sample_mask_{}.png'.format(ix))
        plt.close()

    return ix


def plot_history(history, last=None, out_folder=None):
    if last is None:
        last = 0

    fig, ax = plt.subplots()

    def _plot_key(ax, key, results, color, scale=None, find_min=False, find_max=False):
        training = results[key]
        validation = results['val_{}'.format(key)]

        if scale:
            adjusted_scale = scale.copy()

            if find_min:  # we're interested in the min => loss curve
                p_min = np.percentile(training + validation, [10])

                if p_min > adjusted_scale[1]:  # will not be seen
                    adjusted_scale[1] = p_min

            if find_max:  # we're interested in the max => metric curve
                p_max = np.percentile(training + validation, [90])

                if p_max < adjusted_scale[0]:  # will not be seen
                    adjusted_scale[0] = p_max

            ax.set_ylim(adjusted_scale)  # fixed_scale

        ax.plot(training, label='training {}'.format(key), color=color)
        ax.plot(validation, '--', label='validation {}'.format(key), color=color)

        if find_min:
            ax.plot(np.argmin(validation), np.min(validation), marker='x', color='r')

        if find_max:
            ax.plot(np.argmax(validation), np.max(validation), marker='x', color='r')

    def _plot_results(results, ax):
        _plot_key(ax, 'loss', results, 'C1', scale=[0, 0.02], find_min=True)

        ax = ax.twinx()  # instantiate a second axes that shares the same x-axis

        _plot_key(ax, 'attila_metrics_mean_IoU', results, 'C0', scale=[0.96, 1], find_max=True)
        _plot_key(ax, 'attila_metrics_DSC', results, 'C2', scale=[0.96, 1], find_max=True)


    results = {
        k: history[k][-last:]
        for k in history.keys()
    }

    _plot_results(results, ax)

    if out_folder:
        fig.savefig(out_folder / 'history.png')
        plt.close()


def extract_preds(X, y, preds, ixs):
    out = []

    for ix in ixs:
        ground_truth_foreground, ground_truth_borders = get_mask(y, ix)
        ground_truth = ground_truth_foreground + ground_truth_borders

        pred_foreground, _ = get_mask(preds, ix)

        out.append(
            (X[ix, ..., 0], ground_truth, pred_foreground)
        )

    return out

def plot_preds(X, y, preds, ixs, cmap, title=None, out_folder=None):
    # todo use `extract_preds`
    # for ix in ixs:
    #     plt.imshow(X[ix, ..., 0], cmap=cmap)
    #     if out_folder:
    #         plt.gcf().savefig(out_folder / 'input_{}.png'.format(ix))
    #         plt.close()


    #     plt.imshow(pred_foreground, cmap='gray')
    #     plt.contour(ground_truth, colors='red', levels=[0.5])
    #     if out_folder:
    #         plt.gcf().savefig(out_folder / 'pred_{}.png'.format(ix))
    #         plt.close()
    pass
