import collections

from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct


R, G, B = 'r', 'g', 'b'
Y, CB, CR = 'y', 'cb', 'cr'


def show_raw_images(images, sizes, titles=None, grey_level=False):
    if titles is None:
        titles = range(len(images))
    _, axarr = plt.subplots(1, len(images))

    for idx, (img, size, title) in enumerate(zip(images, sizes, titles)):
        if isinstance(img, str):
            with open(img, 'rb') as img_file:
                arr = np.fromfile(img_file, dtype=np.uint8)
        else:
            arr = np.array(img)
        arr.shape = size if grey_level else (*size, 3)
        if len(images) == 1:
            axarr.set_title(title)
            axarr.imshow(arr, cmap='gray', vmin=0, vmax=255)
        else:
            axarr[idx].set_title(title)
            axarr[idx].imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.show()


def rgb2ycbcr(r, g, b):
    """Convert RGB to YCbCr.

    The range of R, G, B should be [0, 255]. The range of Y, Cb, Cr is [0, 255],
    [-128, 127], [-128, 127] respectively.

    Arguments:
        r {np.ndarray} -- Red Layer.
        g {np.ndarray} -- Green Layer.
        b {np.ndarray} -- Blue Layer.

    Returns:
        OrderDict -- An ordered dictionary containing Y, Cb, Cr layers.
    """

    return collections.OrderedDict((
        (Y, + 0.299 * r + 0.587 * g + 0.114 * b),
        (CB, - 0.168736 * r - 0.331264 * g + 0.5 * b),
        (CR, + 0.5 * r - 0.418688 * g - 0.081312 * b)
    ))


def ycbcr2rgb(y, cb, cr):
    """Convert YCbCr to RGB.

    The range of Y, Cb, Cr should be [0, 255], [-128, 127], [-128, 127]
    respectively. The range of R, G, B is [0, 255].

    Arguments:
        y {np.ndarray} -- Luminance Layer.
        cb {np.ndarray} -- Chrominance (Cb) Layer.
        cr {np.ndarray} -- Chrominance (Cr) Layer.

    Returns:
        dict -- A dictionary containing R, G, B layers.
    """

    return collections.OrderedDict((
        (R, y + 1.402 * cr),
        (G, y - 0.344136 * cb - 0.714136 * cr),
        (B, y + 1.772 * cb)
    ))


def downsample(arr, mode):
    """Downsample an 2D array.

    Arguments:
        arr {2d numpy array} -- The target array.
        mode {1 or 2} -- Downsample ratio (4:mode).

    Returns:
        2d numpy array -- Downsampled array.
    """

    if mode == 4:
        return arr
    return arr[::3 - mode, ::2]


def upsample(arr, mode):
    """Upsample an 2D array.

    Arguments:
        arr {2d numpy array} -- The target array.
        mode {1 or 2} -- Upsample ratio (4:mode).

    Returns:
        2d numpy array -- Upsampled array.
    """

    if mode == 4:
        return arr
    return arr.repeat(3 - mode, axis=0).repeat(2, axis=1)


def block_slice(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.

    `reshape` will raise a `ValueError` if `nrows` or `ncols` doesn't evenly
    divide the shape.
    """
    h, _ = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def block_combine(arr, nrows, ncols):
    """Combine a list of blocks (m * n) into nrows * ncols 2D matrix.

    Arguments:
        arr {3D np.array} -- A list of blocks in the format:
            arr[# of block][block row size][block column size]
        nrows {int} -- The target row size after combination.
        ncols {int} -- The target column size after combination.

    Returns:
        2D np.array -- Combined matrix.

    Raise:
        ValueError -- The size of `arr` is not equal to `nrows * ncols`.
    """

    if arr.size != nrows * ncols:
        raise ValueError('The size of arr should be equal to nrows * ncols')

    _, block_nrows, block_ncols = arr.shape

    return (arr.reshape(nrows // block_nrows, -1, block_nrows, block_ncols)
            .swapaxes(1, 2)
            .reshape(nrows, ncols))


def dct2d(arr):
    return dct(dct(arr, norm='ortho', axis=0), norm='ortho', axis=1)


def idct2d(arr):
    return idct(idct(arr, norm='ortho', axis=0), norm='ortho', axis=1)


def quantize(block, block_type, quality=50, inverse=False):
    if block_type == Y:
        quantization_table = LUMINANCE_QUANTIZATION_TABLE
    else:  # Cb or Cr (LUMINANCE)
        quantization_table = CHROMINANCE_QUANTIZATION_TABLE
    factor = 5000 / quality if quality < 50 else 200 - 2 * quality
    if inverse:
        return block * (quantization_table * factor / 100)
    return block / (quantization_table * factor / 100)


LUMINANCE_QUANTIZATION_TABLE = np.array((
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 36, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99)
))

CHROMINANCE_QUANTIZATION_TABLE = np.array((
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99)
))
