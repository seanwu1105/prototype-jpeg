import math

import numpy as np
from scipy.fftpack import dct

from .utils import (rgb2ycbcr, ycbcr2rgb, downsample, upsample, block_slice,
                    block_combine, dct2d, idct2d, quantize)
from .codec import encode, decode


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
#       - Get info about how many rows and cols are padded to 8N * 8N for
#         extraction
#   Is Grey Level
#   Subsampling Mode (This cannot be identified by image size as padding
#       process increase the possibilities)
#   Quality Factor

# Improvements:
#   Multiprocessing for different blocks, DC and AC VLC


def compress(byte_seq, size, quality=50, grey_level=False, subsampling_mode=1):
    img_arr = np.fromfile(byte_seq, dtype=np.uint8).reshape(
        size if grey_level else (*size, 3)
    )

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

                # Quantization and Rounding
                data[key][idx] = np.rint(quantize(
                    data[key][idx],
                    key,
                    quality=quality)
                )

        # Entropy Encoder
        return encode(data)

    # Grey Level Image
    raise NotImplementedError('Grey level image is not yet implemented.')


def extract(byte_seq):
    # TODO: Read Header
    size = (512, 512)
    grey_level = False
    quality = 50
    subsampling_mode = 1

    # Calculate the size after subsampling.
    if subsampling_mode == 4:
        subsampled_size = size
    else:
        subsampled_size = (
            size[0] if subsampling_mode == 2 else school_round(size[0] / 2),
            school_round(size[1] / 2)
        )

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
                layer[idx] = quantize(
                    block,
                    key,
                    quality=quality,
                    inverse=True
                )

                # 2D IDCT.
                layer[idx] = idct2d(layer[idx])

            # Calculate the size after subsampling and padding.
            if key == 'y':
                padded_size = ((s // 8 + 1) * 8 if s % 8 else s for s in size)
            else:
                padded_size = ((s // 8 + 1) * 8 if s % 8 else s
                               for s in subsampled_size)
            # Combine the blocks into original image
            data[key] = block_combine(layer, *padded_size)

        # Clip Padded Image
        data['y'] = data['y'][:size[0], :size[1]]
        data['cb'] = data['cb'][:subsampled_size[0], :subsampled_size[1]]
        data['cr'] = data['cr'][:subsampled_size[0], :subsampled_size[1]]

        # Inverse Level Offset
        data['y'] = data['y'] + 128

        # Upsampling and Clipping
        data['cb'] = upsample(data['cb'], subsampling_mode)[:size[0], :size[1]]
        data['cr'] = upsample(data['cr'], subsampling_mode)[:size[0], :size[1]]

        # Color Space Conversion
        data = ycbcr2rgb(**data)

        # Rounding, Clipping and Flatten
        data = {k: np.rint(np.clip(v, 0, 255)).flatten()
                for k, v in data.items()}

        # Combine layers into signle raw data.
        data = (np.dstack((data['r'], data['g'], data['b']))
                .flatten()
                .astype(np.uint8))
    return data


def school_round(val):
    if float(val) % 1 >= 0.5:
        return math.ceil(val)
    return round(val)
