import logging
import math
import os
import time

from bitarray import bitarray, bits2bytes
import numpy as np

from .codec import Encoder, Decoder, DC, AC, LUMINANCE, CHROMINANCE
from .utils import (rgb2ycbcr, ycbcr2rgb, downsample, upsample, block_slice,
                    block_combine, dct2d, idct2d, quantize, Y, CB, CR)


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


def compress(f, size, quality=50, grey_level=False, subsampling_mode=1):
    start_time = time.perf_counter()
    logging.getLogger(__name__).info('Original file size: '
                                     f'{os.fstat(f.fileno()).st_size} Bytes')

    if quality <= 0 or quality > 95:
        raise ValueError('Quality should within (0, 95].')

    img_arr = np.fromfile(f, dtype=np.uint8).reshape(
        size if grey_level else (*size, 3)
    )

    if grey_level:
        data = {Y: img_arr.astype(float)}

    else:  # RGB
        # Color Space Conversion (w/o Level Offset)
        data = rgb2ycbcr(*(img_arr[:, :, idx] for idx in range(3)))

        # Subsampling
        data[CB] = downsample(data[CB], subsampling_mode)
        data[CR] = downsample(data[CR], subsampling_mode)

    # Level Offset
    data[Y] = data[Y] - 128

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

        # Rounding
        data[key] = np.rint(data[key]).astype(int)

    if grey_level:
        # Entropy Encoder
        encoded = Encoder(data[Y], LUMINANCE).encode()

        # Combine grey level data as binary in the order:
        #   DC, AC
        order = (encoded[DC], encoded[AC])

    else:  # RGB
        # Entropy Encoder
        encoded = {
            LUMINANCE: Encoder(data[Y], LUMINANCE).encode(),
            CHROMINANCE: Encoder(
                np.vstack((data[CB], data[CR])),
                CHROMINANCE
            ).encode()
        }

        # Combine RGB data as binary in the order:
        #   LUMINANCE.DC, LUMINANCE.AC, CHROMINANCE.DC, CHROMINANCE.AC
        order = (encoded[LUMINANCE][DC], encoded[LUMINANCE][AC],
                 encoded[CHROMINANCE][DC], encoded[CHROMINANCE][AC])

    bits = bitarray(''.join(order))

    logging.getLogger(__name__).info(
        'Time elapsed: %.4f seconds' % (time.perf_counter() - start_time)
    )
    return {
        'data': bits,
        'header': {
            'size': size,
            'grey_level': grey_level,
            'quality': quality,
            'subsampling_mode': subsampling_mode,
            # Remaining bits length is the fake filled bits for 8 bits as a
            # byte.
            'remaining_bits_length': bits2bytes(len(bits)) * 8 - len(bits),
            'data_slice_lengths': tuple(len(d) for d in order)
        }
    }


def extract(f, header):
    def school_round(val):
        if float(val) % 1 >= 0.5:
            return math.ceil(val)
        return round(val)

    start_time = time.perf_counter()
    logging.getLogger(__name__).info('Compressed file size: '
                                     f'{os.fstat(f.fileno()).st_size} Bytes')

    bits = bitarray()
    bits.fromfile(f)
    bits = bits.to01()

    # Read Header
    size = header['size']
    grey_level = header['grey_level']
    quality = header['quality']
    subsampling_mode = header['subsampling_mode']
    remaining_bits_length = header['remaining_bits_length']
    dsls = header['data_slice_lengths']  # data_slice_lengths

    # Calculate the size after subsampling.
    if subsampling_mode == 4:
        subsampled_size = size
    else:
        subsampled_size = (
            size[0] if subsampling_mode == 2 else school_round(size[0] / 2),
            school_round(size[1] / 2)
        )

    # Preprocessing Byte Sequence:
    #   1. Remove Remaining (Fake Filled) Bits.
    #   2. Slice Bits into Dictionary Data Structure for `Decoder`.

    if remaining_bits_length:
        bits = bits[:-remaining_bits_length]

    if grey_level:
        # The order of dsls (grey level) is:
        #   DC, AC
        sliced = {
            DC: bits[:dsls[0]],
            AC: bits[dsls[0]:]
        }
    else:  # RGB
        # The order of dsls (RGB) is:
        #   LUMINANCE.DC, LUMINANCE.AC, CHROMINANCE.DC, CHROMINANCE.AC
        sliced = {
            LUMINANCE: {
                DC: bits[:dsls[0]],
                AC: bits[dsls[0]:dsls[0] + dsls[1]]
            },
            CHROMINANCE: {
                DC: bits[dsls[0] + dsls[1]:dsls[0] + dsls[1] + dsls[2]],
                AC: bits[dsls[0] + dsls[1] + dsls[2]:]
            }
        }

    # Huffman Decoding
    if grey_level:
        data = {Y: Decoder(sliced, LUMINANCE).decode()}
    else:
        cb, cr = np.split(Decoder(
            sliced[CHROMINANCE],
            CHROMINANCE
        ).decode(), 2)
        data = {
            Y: Decoder(sliced[LUMINANCE], LUMINANCE).decode(),
            CB: cb,
            CR: cr
        }

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
        if key == Y:
            padded_size = ((s // 8 + 1) * 8 if s % 8 else s for s in size)
        else:
            padded_size = ((s // 8 + 1) * 8 if s % 8 else s
                           for s in subsampled_size)
        # Combine the blocks into original image
        data[key] = block_combine(layer, *padded_size)

    # Inverse Level Offset
    data[Y] = data[Y] + 128

    # Clip Padded Image
    data[Y] = data[Y][:size[0], :size[1]]
    if not grey_level:
        data[CB] = data[CB][:subsampled_size[0], :subsampled_size[1]]
        data[CR] = data[CR][:subsampled_size[0], :subsampled_size[1]]

        # Upsampling and Clipping
        data[CB] = upsample(data[CB], subsampling_mode)[:size[0], :size[1]]
        data[CR] = upsample(data[CR], subsampling_mode)[:size[0], :size[1]]

        # Color Space Conversion
        data = ycbcr2rgb(**data)

    # Rounding, Clipping and Flatten
    for k, v in data.items():
        data[k] = np.rint(np.clip(v, 0, 255)).flatten()

    logging.getLogger(__name__).info(
        'Time elapsed: %.4f seconds' % (time.perf_counter() - start_time)
    )
    # Combine layers into signle raw data.
    return (np.dstack((data.values()))
            .flatten()
            .astype(np.uint8))
