import tempfile
import unittest

from prototype_jpeg import __version__, compress, extract


def test_version():
    assert __version__ == '0.1.0'


class TestCompressAndExtract(unittest.TestCase):
    def test_rgb(self):
        compress_and_extract({
            'fn': 'tests/images/rgb/Baboon.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 50,
            'subsampling_mode': 1
        })
        compress_and_extract({
            'fn': 'tests/images/rgb/Lena.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 50,
            'subsampling_mode': 1
        })

    def test_rgb_quality_95(self):
        compress_and_extract({
            'fn': 'tests/images/rgb/Baboon.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 95,
            'subsampling_mode': 1
        })
        compress_and_extract({
            'fn': 'tests/images/rgb/Lena.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 95,
            'subsampling_mode': 1
        })

    def test_rgb_quality_1(self):
        compress_and_extract({
            'fn': 'tests/images/rgb/Baboon.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 1,
            'subsampling_mode': 1
        })
        compress_and_extract({
            'fn': 'tests/images/rgb/Lena.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 1,
            'subsampling_mode': 1
        })

    def test_rgb_subsampling_mode_2(self):
        compress_and_extract({
            'fn': 'tests/images/rgb/Baboon.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 50,
            'subsampling_mode': 2
        })
        compress_and_extract({
            'fn': 'tests/images/rgb/Lena.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 50,
            'subsampling_mode': 2
        })

    def test_rgb_subsampling_mode_4(self):
        compress_and_extract({
            'fn': 'tests/images/rgb/Baboon.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 50,
            'subsampling_mode': 4
        })
        compress_and_extract({
            'fn': 'tests/images/rgb/Lena.raw',
            'size': (512, 512),
            'grey_level': False,
            'quality': 50,
            'subsampling_mode': 4
        })

    def test_grey_level(self):
        compress_and_extract({
            'fn': 'tests/images/grey_level/Baboon.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 50,
            'subsampling_mode': 1
        })
        compress_and_extract({
            'fn': 'tests/images/grey_level/Lena.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 50,
            'subsampling_mode': 1
        })

    def test_grey_level_quality_95(self):
        compress_and_extract({
            'fn': 'tests/images/grey_level/Baboon.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 95,
            'subsampling_mode': 1
        })
        compress_and_extract({
            'fn': 'tests/images/grey_level/Lena.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 95,
            'subsampling_mode': 1
        })

    def test_grey_level_quality_1(self):
        compress_and_extract({
            'fn': 'tests/images/grey_level/Baboon.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 1,
            'subsampling_mode': 1
        })
        compress_and_extract({
            'fn': 'tests/images/grey_level/Lena.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 1,
            'subsampling_mode': 1
        })

    def test_grey_level_subsampling_mode_2(self):
        compress_and_extract({
            'fn': 'tests/images/grey_level/Baboon.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 50,
            'subsampling_mode': 2
        })
        compress_and_extract({
            'fn': 'tests/images/grey_level/Lena.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 50,
            'subsampling_mode': 2
        })

    def test_grey_level_subsampling_mode_4(self):
        compress_and_extract({
            'fn': 'tests/images/grey_level/Baboon.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 50,
            'subsampling_mode': 4
        })
        compress_and_extract({
            'fn': 'tests/images/grey_level/Lena.raw',
            'size': (512, 512),
            'grey_level': True,
            'quality': 50,
            'subsampling_mode': 4
        })


def compress_and_extract(spec):
    with open(spec['fn'], 'rb') as raw_file:
        compressed = compress(
            raw_file,
            size=spec['size'],
            grey_level=spec['grey_level'],
            quality=spec['quality'],
            subsampling_mode=spec['subsampling_mode']
        )
    header = compressed['header']
    with tempfile.TemporaryFile() as compressed_file:
        compressed['data'].tofile(compressed_file)
        compressed_file.seek(0)
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
    return extracted
