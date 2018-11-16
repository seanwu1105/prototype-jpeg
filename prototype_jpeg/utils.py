import numpy as np


def rgb2ycbcr(r, g, b):
    return {
        'y': + 0.299 * r + 0.587 * g + 0.114 * b,
        'cb': - 0.168736 * r - 0.331264 * g + 0.5 * b,
        'cr': + 0.5 * r - 0.418688 * g - 0.081312 * b
    }


def ycbcr2rgb(y, cb, cr):
    return {
        'r': y + 1.402 * cr,
        'g': y - 0.344136 * cb - 0.714136 * cr,
        'b': y + 1.772 * cb
    }


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
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def quantize(block, block_type, quality=50):
    if block_type == 'y' or block_type == 'luminance':
        quantization_table = np.array([
            [],
        ])
    else:
        quantization_table = np.array([
            [],
        ])
    return block
