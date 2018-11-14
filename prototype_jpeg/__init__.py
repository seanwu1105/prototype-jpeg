import numpy as np

from .utils import rgb2ycbcr, ycbcr2rgb, downsample, block_slice


#############################################################
# Compress Algorithm:                                       #
#       Color Space Conversion                              #
#       Subsampling (Luminance and Chrominance)             #
#       For each color layer:                               #
#           Pad layer to 8n * 8n                            #
#           8*8 Slicing                                     #
#           For each slicing:                               #
#               DCT                                         #
#               Quantization (Luminance and Chrominance)    #
#               Entropy Coder (Luminance and Chrominance)   #
#############################################################

# Header:
#   Image Size (use this to identify which subsampling mode is used when decoding)

# Improvements:
#   Multiprocessing for different blocks, DC and AC VLC


def compress(img_arr, size, quality=50, grey_level=False, subsampling_mode=1):
    img_arr.shape = size if grey_level else (*size, 3)

    if not grey_level:
        # Color Space Conversion with Level Offset
        ycbcr = rgb2ycbcr(*(img_arr[:, :, idx] for idx in range(3)))

        # Subsampling
        ycbcr['cb'] = downsample(ycbcr['cb'], subsampling_mode)
        ycbcr['cr'] = downsample(ycbcr['cr'], subsampling_mode)

        for key, layer in ycbcr.items():
            nrows, ncols = layer.shape

            # Pad Layers to 8N * 8N
            ycbcr[key] = np.pad(
                layer,
                (
                    (0, (nrows // 8 + 1) * 8 - nrows if nrows % 8 else 0),
                    (0, (ncols // 8 + 1) * 8 - ncols if ncols % 8 else 0)
                ),
                mode='constant'
            )

            # Block Slicing
            ycbcr[key] = block_slice(layer, 8, 8)

    else:
        pass

    return ycbcr


def extract(filename):
    pass
