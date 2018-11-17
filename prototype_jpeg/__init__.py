import numpy as np
from scipy.fftpack import dct

from .utils import (rgb2ycbcr, ycbcr2rgb, downsample, block_slice, dct2d,
                    idct2d, quantize)


#############################################################
# Compress Algorithm:                                       #
#       Color Space Conversion                              #
#       Subsampling (Chrominance)                           #
#       Level Offset (Luminance)                            #
#       For each color layer:                               #
#           Pad layer to 8n * 8n                            #
#           8*8 Slicing                                     #
#           For each slicing:                               #
#               DCT                                         #
#               Quantization (Luminance and Chrominance)    #
#               Entropy Coder (Luminance and Chrominance)   #
#############################################################

# Header:
#   Image Size
#       - Get info about which subsampling mode is used
#       - Get info about how many rows and cols are padded to 8N * 8N

# Improvements:
#   Multiprocessing for different blocks, DC and AC VLC


def compress(img_arr, size, quality=50, grey_level=False, subsampling_mode=1):
    img_arr.shape = size if grey_level else (*size, 3)

    if not grey_level:
        # Color Space Conversion with Level Offset
        data = rgb2ycbcr(*(img_arr[:, :, idx] for idx in range(3)))

        # Subsampling
        data['cb'] = downsample(data['cb'], subsampling_mode)
        data['cr'] = downsample(data['cr'], subsampling_mode)

        # Level Offset
        data['y'] = data['y'] - 128

        for key, layer in data.items():
            nrows, ncols = layer.shape

            # Pad Layers to 8N * 8N
            data[key] = np.pad(
                layer,
                (
                    (0, (nrows // 8 + 1) * 8 - nrows if nrows % 8 else 0),
                    (0, (ncols // 8 + 1) * 8 - ncols if ncols % 8 else 0)
                ),
                mode='constant'
            )

            # Block Slicing
            data[key] = block_slice(data[key], 8, 8)

            for idx, block in enumerate(data[key]):
                # 2D DCT
                data[key][idx] = dct2d(block)

                # Quantization
                data[key][idx] = quantize(data[key][idx], key, quality=quality)

    else:
        pass

    return data


def extract(filename):
    pass
    # For IDCT: https://stackoverflow.com/questions/34890585/in-scipy-why-doesnt-idctdcta-equal-to-a
