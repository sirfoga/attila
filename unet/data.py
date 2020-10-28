def crop_center(cropx,cropy):
    def _f(img):
        (y,x,_) = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]
    return _f

# todo: random crop
