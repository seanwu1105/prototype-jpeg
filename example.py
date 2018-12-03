import logging

import numpy as np

from prototype_jpeg import compress, extract
from prototype_jpeg.utils import show_raw_images, psnr


logging.basicConfig(level=logging.INFO)


def example():
    specs = ({
        'fn': 'tests/images/rgb/Baboon.raw',
        'size': (512, 512),
        'grey_level': False,
        'quality': 50,
        'subsampling_mode': 1
    }, {
        'fn': 'tests/images/rgb/Lena.raw',
        'size': (512, 512),
        'grey_level': False,
        'quality': 50,
        'subsampling_mode': 1
    }, {
        'fn': 'tests/images/rgb/Baboon.raw',
        'size': (512, 512),
        'grey_level': False,
        'quality': 5,
        'subsampling_mode': 1
    }, {
        'fn': 'tests/images/rgb/Lena.raw',
        'size': (512, 512),
        'grey_level': False,
        'quality': 5,
        'subsampling_mode': 1
    }, {
        'fn': 'tests/images/grey_level/Baboon.raw',
        'size': (512, 512),
        'grey_level': True,
        'quality': 50,
        'subsampling_mode': 1
    }, {
        'fn': 'tests/images/grey_level/Lena.raw',
        'size': (512, 512),
        'grey_level': True,
        'quality': 50,
        'subsampling_mode': 1
    })
    for spec in specs:
        with open(spec['fn'], 'rb') as raw_file:
            original = np.fromfile(raw_file, dtype=np.uint8)
            raw_file.seek(0)
            compressed = compress(
                raw_file,
                size=spec['size'],
                grey_level=spec['grey_level'],
                quality=spec['quality'],
                subsampling_mode=spec['subsampling_mode']
            )
        with open('compressed.protojpg', 'wb') as compressed_file:
            compressed['data'].tofile(compressed_file)
        header = compressed['header']

        with open('compressed.protojpg', 'rb') as compressed_file:
            extracted = extract(
                compressed_file,
                header={
                    'size': header['size'],
                    'grey_level': header['grey_level'],
                    'quality': header['quality'],
                    'subsampling_mode': header['subsampling_mode'],
                    'remaining_bits_length': header['remaining_bits_length'],
                    'data_slice_lengths': header['data_slice_lengths']
                }
            )

        logging.getLogger(__name__).info(
            'PSNR: %.4f' % psnr(original, extracted)
        )
        show_raw_images(
            (original, extracted),
            (spec['size'], spec['size']),
            (spec['fn'], 'Compressed and Extracted'),
            grey_level=spec['grey_level']
        )


if __name__ == '__main__':
    example()
