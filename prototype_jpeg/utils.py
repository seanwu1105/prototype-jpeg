import numpy as np


def rgb2ycbcr(r, g, b):
    return {
        'y':  0   + 0.299    * r + 0.587    * g + 0.114    * b,
        'cb': 128 - 0.168736 * r - 0.331264 * g + 0.5      * b,
        'cr': 128 + 0.5      * r - 0.418688 * g - 0.081312 * b
    }


def ycbcr2rgb(y, cb, cr):
    return {
        'r': y                         + 1.402    * (cr - 128),
        'g': y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128),
        'b': y + 1.772    * (cb - 128)
    }


def downsample(arr, mode):
    if mode == 4:
        return arr
    return arr[::3 - mode, ::2]


def upsample(arr, mode):
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
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
