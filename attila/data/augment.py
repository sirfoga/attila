import numpy as np
from PIL import Image, ImageFilter

from attila.util.f import apply_f


def blur(img):
    """ img has 3 channels, last dim is for color channels """
    
    im = np.uint8(img[..., 0] * 255)  # 3D -> 2D
    im = Image.fromarray(im, mode='L')  # [0, 1] -> [0, 255]
    im = im.filter(ImageFilter.BLUR)
    im = np.divide(np.array(im), 255)  # [0, 255] -> [0, 1]

    return np.expand_dims(im, -1)  # restore 3D axis


def make_noise(img):
    """ img has 3 channels, last dim is for color channels """

    mu, std = 0, 0.2
    noise = np.random.normal(mu, std, img.shape)
    return np.clip(img + noise, 0.0, 1.0)


def do_augment(img):
    _u = np.random.uniform(0.0, 1.0)
    
    if _u < 0.33:
        return make_noise(img)
    elif _u < 0.66:
        return blur(img)
    
    return img  # do nothing
