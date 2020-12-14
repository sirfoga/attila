import numpy as np
from PIL import Image, ImageFilter

from attila.util.f import apply_f


def blur(img):
    im = Image.fromarray(np.uint8(img * 255))
    im = im.filter(ImageFilter.BLUR)
    return np.array(im)


def make_noise(img):
    mu, std = 0, 20
    noise = np.random.normal(mu, std, img.shape)
    return np.clip(img + noise, 0, 255)


from matplotlib import pyplot as plt


def do_augment(img):
    plt.gca().hist(img.ravel(), bins=256)  # breakpoint

    _u = np.random.uniform(0.0, 1.0)
    if _u < 0.33:
        return make_noise(img)
    elif _u < 0.66:
        return blur(img)
    
    return img  # do nothing
