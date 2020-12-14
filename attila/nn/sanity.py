import numpy as np


def gen_img(shape, px_generator):
    n_channels = shape[2]  # assuming channels last
    assert n_channels == px_generator().shape[0]

    x = np.zeros(shape)

    for r in range(shape[0]):
        for c in range(shape[1]):
            x[r, c, :] = px_generator()

    return x


def gen_random_img(shape):
    def _px_generator(n_channels):
        def _f():
            x = np.zeros(n_channels)
            x[np.random.randint(n_channels)] = 1.0  # one-hot
            return x
        return _f

    n_channels = shape[2]  # assuming channels last
    return gen_img(shape, _px_generator(n_channels))


def gen_ch_img(shape, ch_index):
    n_channels = shape[2]  # assuming channels last

    def _px_generator(ch_index):
        def _f():
            x = np.zeros(n_channels)
            x[ch_index] = 1.0  # one-hot
            return x
        return _f

    return gen_img(shape, _px_generator(ch_index))


def gen_background_img(shape):
    return gen_ch_img(shape, -1)


def gen_foreground_img(shape):
    return gen_ch_img(shape, 0)


def do_sanity_check(samples, checks, config, f_out='   - {}: {:.3f} +- {:.2f}'):
    def _compute_checks(samples, y_pred):
        for check in checks:
            res = [
                check(sample, img)
                for sample in samples
            ]
            yield (np.mean(res), np.std(res))

    img_shape = (config.getint('image', 'height'), config.getint('image', 'width'))
    n_channels = config.getint('image', 'n classes')
    imgs = [
        (gen_random_img((*img_shape, n_channels)), 'RANDOM'),
        (gen_background_img((*img_shape, n_channels)), 'ALL BLACKS'),
        (gen_foreground_img((*img_shape, n_channels)), 'FOREGROUND'),
    ]

    print('=== SANITY CHECKS on {} samples ==='.format(len(samples)))
    for img, title in imgs:
        print(' == {}'.format(title))
        for check, res in zip(checks, _compute_checks(samples, img)):
            print(f_out.format(check.__name__, *res))


def get_steps_per_epoch(X, bs, multi=1.0):
    out = len(X) / bs * multi
    return int(np.ceil(out))
