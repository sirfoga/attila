def get_data(imgs_path, masks_path, im_size):
    """ Get and resize train images and masks """

    im_width, im_height = im_size
    list_imgs = next(os.walk(imgs_path))[2]
    X = np.zeros((len(list_imgs), im_width, im_height, 1), dtype=np.float32)
    y = np.zeros((len(list_imgs), im_width, im_height, 1), dtype=np.float32)
    
    for i, img_name in enumerate(list_imgs):
        mask_name = img_name.replace('img_', 'mask_')
        
        # load images
        img = img_to_array(load_img(imgs_path + '/' + img_name, grayscale=True))
        img = resize(img, (im_width, im_height, 1), mode='constant', preserve_range=True)

        # load masks
        mask = img_to_array(load_img(masks_path + '/' + mask_name, grayscale=True))
        mask = resize(mask, (im_width, im_height, 1), mode='constant', preserve_range=True)

        # save images
        X[i, ..., 0] = img.squeeze() / 255
        y[i] = mask / 255

    return X, y

# get data
X, y = get_data(data_path + '/images', data_path + '/masks', (128, 128))
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)  # split train and test
