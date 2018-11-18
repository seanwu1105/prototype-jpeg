import numpy as np
from scipy.fftpack import dct

from .utils import (rgb2ycbcr, ycbcr2rgb, downsample, block_slice, dct2d,
                    idct2d, quantize)
from codec import encode, decode


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
#       Write Header                                        #
#############################################################

# Header:
#   Image Size
#       - Get info about which subsampling mode is used for extraction
#       - Get info about how many rows and cols are padded to 8N * 8N for
#         extraction
#   Is Grey Level
#   Quality Factor

# Improvements:
#   Multiprocessing for different blocks, DC and AC VLC


def compress(img_arr, size, quality=50, grey_level=False, subsampling_mode=1):
    img_arr.shape = size if grey_level else (*size, 3)

    if not grey_level:
        # Color Space Conversion (w/o Level Offset)
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

        # Entropy Encoder
        return encode(data)
    
    # Grey Level Image
    raise NotImplementedError('Grey level image is not yet implemented.')


def extract(byte_seq):
    # TODO: Read Header
    grey_level = False
    quality = 50

    if not grey_level:
        # TODO: Entropy Decoder
        data = decode(byte_seq)
        #   Do something to get decoded data having the following format:
        #   data = {
        #       'y': array_of_blocks,
        #       'cb': array_of_blocks,
        #       'cr': array_of_blocks
        #   }

        for key, layer in data.items():
            for idx, block in enumerate(layer):
                # Inverse Quantization
                layer[idx] = quantize(block, key, quality=quality, inverse=True)

                # 2D IDCT
                layer[idx] = idct2d(layer[idx])

            # Combine the blocks into original image
            # data[key] = block_combine(layer)

        # Inverse Level Offset
        # data['y'] = data['y'] + 128

        # Upsampling

        # Color Space Conversion

        # Clip Image
        # XXX: This could be done after combine the blocks into original image
        # to speed up the decoding process a little bit, but this would require
        # further calculation about the "before upsampling padding sizes".



    return data
    # For IDCT: https://stackoverflow.com/questions/34890585/in-scipy-why-doesnt-idctdcta-equal-to-a
