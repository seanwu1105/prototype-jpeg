from matplotlib import pyplot as plt
import numpy as np

import prototype_jpeg
from prototype_jpeg.utils import rgb2ycbcr

def main():
    with open('tests/images/rgb/Lena.raw', 'rb') as img_file:
        original_img = np.fromfile(img_file, dtype=np.uint8)
    original_img.shape = (512, 512, 3)
    results = rgb2ycbcr(*(original_img[:, :, idx] for idx in range(3)))
    show_raw_images(results.values(), (512, 512), grey_level=True)

def show_raw_images(images, size, titles=None, grey_level=False):
    if titles is None:
        titles = range(len(images))
    _, axarr = plt.subplots(1, len(images))

    for idx, img in enumerate(images):
        if isinstance(img, str):
            with open(img, 'rb') as img_file:
                arr = np.fromfile(img_file, dtype=np.uint8)
        else:
            arr = np.array(img)
        arr.shape = size if grey_level else (*size, 3)
        axarr[idx].set_title(titles[idx])
        axarr[idx].imshow(arr, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()
