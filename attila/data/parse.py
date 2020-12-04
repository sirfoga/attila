from tifffile import imread
import numpy as np

from attila.data.transform import rm_percentiles_transformation, normalize_transformation, crop_center_transformation, add_dim, do_transformations, img2channels


def load_tiff(f):
    return imread(f)


def load_image(f):
    return load_tiff(f)


def load_mask(f):
    return load_tiff(f)


def get_data(imgs_path, masks_path, extension='.tif'):
    list_imgs = [
        f
        for f in imgs_path.iterdir()
        if str(f).endswith(extension)
    ]

    images = []
    masks = []

    for img_path in list_imgs:
        img = load_image(img_path).squeeze()

        mask_path = str(img_path).replace(str(imgs_path), str(masks_path)).replace('img_', 'mask_')
        mask = load_mask(mask_path)

        images.append(np.array(img))
        masks.append(np.array(mask))

    return images, masks


def parse_data(raw, img_shape):
    (X, y) = raw

    base_transformations = [
        np.array,  # just in case parser did not np.array-ed
        rm_percentiles_transformation(2, 98),  # threshold outliers
        normalize_transformation((0, 1)),
        crop_center_transformation(img_shape),
    ]

    X = do_transformations(
        X,
        base_transformations + [add_dim()]
    )

    y = do_transformations(
        y,
        base_transformations + [img2channels()]
    )

    return X, y
