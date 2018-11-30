import math

from bitarray import bitarray, bits2bytes
import numpy as np
from scipy.fftpack import dct

from .codec import Encoder, Decoder, DC, AC, LUMINANCE, CHROMINANCE
from .utils import (rgb2ycbcr, ycbcr2rgb, downsample, upsample, block_slice,
                    block_combine, dct2d, idct2d, quantize)


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

                # Quantization
                data[key][idx] = quantize(data[key][idx], key, quality=quality)

            # Rounding
            data[key] = np.rint(data[key]).astype(int)
        # Entropy Encoder
        encoded = Encoder(data).encode()

        # Combine data as binary.
        bits = bitarray(''.join(d for c in encoded.values()
                                for d in c.values()))
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
                # The order of data slice lengths is:
                #   (DC.LUM, DC.CHR, AC.LUM, AC.CHR)
                'data_slice_lengths': (
                    len(encoded[DC][LUMINANCE]),
                    len(encoded[DC][CHROMINANCE]),
                    len(encoded[AC][LUMINANCE]),
                    len(encoded[AC][CHROMINANCE])
                )
            }
        }

    # # Grey Level Image

    # # Level Offset
    # data = img_arr - 128

    # # Pad 2D Data to 8N * 8N
    # nrows, ncols = data.shape
    # data = np.pad(
    #     data,
    #     (
    #         (0, (nrows // 8 + 1) * 8 - nrows if nrows % 8 else 0),
    #         (0, (ncols // 8 + 1) * 8 - ncols if ncols % 8 else 0)
    #     ),
    #     mode='constant'
    # )

    # # Block Slicing
    # data = block_slice(data, 8, 8)

    # for idx, block in enumerate(data):
    #     # 2D DCT
    #     data[idx] = dct2d(block)

    #     # Quantization
    #     data[idx] = quantize(data[idx], 'y', quality=quality)

    # # Rounding
    # data = np.rint(data).astype(int)
    # raise Exception('fuck') -------------------------------------------
    # # Entropy Encoder
    # encoded = Encoder(data).encode()

    # # Combine data as binary.
    # bits = bitarray(''.join(d for c in encoded.values()
    #                         for d in c.values()))
    # return {
    #     'data': bits,
    #     'header': {
    #         'size': size,
    #         'grey_level': grey_level,
    #         'quality': quality,
    #         'subsampling_mode': subsampling_mode,
    #         # Remaining bits length is the fake filled bits for 8 bits as a
    #         # byte.
    #         'remaining_bits_length': bits2bytes(len(bits)) * 8 - len(bits),
    #         # The order of data slice lengths is:
    #         #   (DC.LUM, DC.CHR, AC.LUM, AC.CHR)
    #         'data_slice_lengths': (
    #             len(encoded[DC][LUMINANCE]),
    #             len(encoded[DC][CHROMINANCE]),
    #             len(encoded[AC][LUMINANCE]),
    #             len(encoded[AC][CHROMINANCE])
    #         )
    #     }
    # }


def extract(byte_seq, header):
    bits = bitarray()
    bits.fromfile(byte_seq)
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

    if not grey_level:  # RGB Image
        # Preprocessing Byte Sequence:
        #   1. Remove Remaining (Fake Filled) Bits.
        #   2. Slice Bits into Dictionary Data Structure for `Decoder`.
        #   data_slice_lengths = (DC.LUM, DC.CHR, AC.LUM, AC.CHR)
        bits = bits[:-remaining_bits_length]
        data = {
            DC: {
                LUMINANCE: bits[:dsls[0]],
                CHROMINANCE: bits[dsls[0]:dsls[0] + dsls[1]]
            },
            AC: {
                LUMINANCE: bits[dsls[0] + dsls[1]:dsls[0] + dsls[1] + dsls[2]],
                CHROMINANCE: bits[dsls[0] + dsls[1] + dsls[2]:]
            }
        }

        data = Decoder(data).decode()
        #   The decoded data having the following format:
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
        data = (np.dstack((data.values()))
                .flatten()
                .astype(np.uint8))

    else:  # Grey Level Image
        #   Only LUMINANCE encoding table is used.
        #   data_slice_lengths = (DC: int, AC: int)
        raise NotImplementedError()
    return data


def school_round(val):
    if float(val) % 1 >= 0.5:
        return math.ceil(val)
    return round(val)
