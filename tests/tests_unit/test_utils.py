import unittest

import numpy as np

from prototype_jpeg.utils import (rgb2ycbcr, ycbcr2rgb, downsample, upsample,
                                  block_slice, block_combine, dct2d, idct2d, quantize)


class TestColorSpaceConversion(unittest.TestCase):
    def test_rgb2ycbcr(self):
        test_input = np.linspace(0, 255, 24, dtype=int).reshape(2, 4, 3)
        expect = {
            'y': np.array([
                [8.965, 41.965, 74.965, 107.965],
                [141.965, 174.965, 207.965, 241.079]
            ]),
            'cb': np.array([
                [7.356096, 7.356096, 7.356096, 7.356096],
                [7.356096, 7.356096, 7.356096, 7.856096]
            ]),
            'cr': np.array([
                [-6.394432, -6.394432, -6.394432, -6.394432],
                [-6.394432, -6.394432, -6.394432, -6.475744]
            ])
        }
        result = rgb2ycbcr(*(test_input[:, :, i] for i in range(3)))
        for key in ('y', 'cb', 'cr'):
            np.testing.assert_array_almost_equal(
                result[key],
                expect[key],
                decimal=0
            )

    def test_ycbcr2rgb(self):
        test_input = np.linspace(0, 255, 24).reshape(2, 4, 3)
        expect = {
            'r': np.array([
                [31.08782609, 110.98043478, 190.87304348, 270.76565217],
                [350.65826087, 430.55086957, 510.44347826, 590.33608696]
            ]),
            'g': np.array([
                [-19.65061043, -21.58878783, -23.52696522, -25.46514261],
                [-27.40332, -29.34149739, -31.27967478, -33.21785217]
            ]),
            'b': np.array([
                [19.64608696, 111.84521739, 204.04434783, 296.24347826],
                [388.4426087, 480.64173913, 572.84086957, 665.04]
            ])
        }
        result = ycbcr2rgb(*(test_input[:, :, i] for i in range(3)))
        for key in ('r', 'g', 'b'):
            np.testing.assert_array_almost_equal(
                result[key],
                expect[key],
                decimal=0
            )

    def test_color_space_conversion(self):
        test_input = {
            k: np.linspace(0, 255, 24).reshape(2, 4, 3)[:, :, i]
            for i, k in enumerate('rgb')
        }
        expect = ycbcr2rgb(**rgb2ycbcr(**test_input))
        for key in ('r', 'g', 'b'):
            np.testing.assert_array_almost_equal(
                test_input[key],
                expect[key],
                decimal=0
            )


class TestSampling(unittest.TestCase):
    def test_downsample(self):
        cases = ({
            'input': np.linspace(0, 255, 16, dtype=int).reshape(4, 4),
            'mode': 1,
            'expect': np.array([
                [0, 34],
                [136, 170]
            ])
        }, {
            'input': np.linspace(0, 255, 16, dtype=int).reshape(4, 4),
            'mode': 2,
            'expect': np.array([
                [0, 34],
                [68, 102],
                [136, 170],
                [204, 238]
            ])
        }, {
            'input': np.linspace(0, 255, 6, dtype=int).reshape(2, 3),
            'mode': 1,
            'expect': np.array([[0, 102]])
        }, {
            'input': np.linspace(0, 255, 6, dtype=int).reshape(2, 3),
            'mode': 2,
            'expect': np.array([
                [0, 102],
                [153, 255]
            ])
        })
        for case in cases:
            result = downsample(case['input'], case['mode'])
            np.testing.assert_array_equal(result, case['expect'])

    def test_upsample(self):
        cases = ({
            'input': np.linspace(0, 255, 4, dtype=int).reshape(2, 2),
            'mode': 1,
            'expect': np.array([
                [0, 0, 85, 85],
                [0, 0, 85, 85],
                [170, 170, 255, 255],
                [170, 170, 255, 255]
            ])
        }, {
            'input': np.linspace(0, 255, 8, dtype=int).reshape(4, 2),
            'mode': 2,
            'expect': np.array([
                [0, 0, 36, 36],
                [72, 72, 109, 109],
                [145, 145, 182, 182],
                [218, 218, 255, 255]
            ])
        }, {
            'input': np.linspace(0, 255, 6, dtype=int).reshape(2, 3),
            'mode': 1,
            'expect': np.array([
                [0, 0, 51, 51, 102, 102],
                [0, 0, 51, 51, 102, 102],
                [153, 153, 204, 204, 255, 255],
                [153, 153, 204, 204, 255, 255]
            ])
        }, {
            'input': np.linspace(0, 255, 6, dtype=int).reshape(2, 3),
            'mode': 2,
            'expect': np.array([
                [0, 0, 51, 51, 102, 102],
                [153, 153, 204, 204, 255, 255]
            ])
        })
        for case in cases:
            result = upsample(case['input'], case['mode'])
            np.testing.assert_array_equal(result, case['expect'])


class TestBlockSlicingAndCombination(unittest.TestCase):
    def test_block_slice(self):
        test_input = np.arange(16).reshape(4, 4)
        expect = np.array([
            [[0, 1],
             [4, 5]],
            [[2, 3],
             [6, 7]],
            [[8, 9],
             [12, 13]],
            [[10, 11],
             [14, 15]],
        ])
        np.testing.assert_array_equal(block_slice(test_input, 2, 2), expect)

    def test_block_slice_individable(self):
        """Test the block_slice function when the shape cannot be divided evenly
        by nrows or ncols.
        """
        test_input = np.arange(16).reshape(4, 4)
        with self.assertRaises(ValueError):
            block_slice(test_input, 3, 3)

    def test_block_combine(self):
        test_input = np.array([
            [[0, 1],
             [4, 5]],
            [[2, 3],
             [6, 7]],
            [[8, 9],
             [12, 13]],
            [[10, 11],
             [14, 15]],
        ])
        expect = np.arange(16).reshape(4, 4)
        np.testing.assert_array_equal(block_combine(test_input, 4, 4), expect)

    def test_block_combine_inequal_number_of_elements(self):
        test_input = np.array([
            [[0, 1],
             [4, 5]],
            [[2, 3],
             [6, 7]],
            [[8, 9],
             [12, 13]],
            [[10, 11],
             [14, 15]],
        ])
        with self.assertRaises(ValueError):
            block_combine(test_input, 4, 3)

    def test_invertible(self):
        test_input = np.arange(64).reshape(8, 8)
        np.testing.assert_array_equal(
            test_input,
            block_combine(block_slice(test_input, 2, 2), 8, 8)
        )


class TestDCT2D(unittest.TestCase):
    def test_dct2d(self):
        test_input = np.array([
            [139, 144, 149, 153, 155, 155, 155, 155],
            [144, 151, 153, 156, 159, 156, 156, 156],
            [150, 155, 160, 163, 158, 156, 156, 156],
            [159, 161, 162, 160, 160, 159, 159, 159],
            [159, 160, 161, 162, 162, 155, 155, 155],
            [161, 161, 161, 161, 160, 157, 157, 157],
            [162, 162, 161, 163, 162, 157, 157, 157],
            [162, 162, 161, 161, 163, 158, 158, 158]
        ])
        expect = np.array([
            [1260, -1, -12, -5, 2, -2, -3, 1],
            [-23, -17, -6, -3, -3, -0, 0, -1],
            [-11, -9, -2, 2, 0, -1, -1, -0],
            [-7, -2, 0, 1, 1, -0, -0, 0],
            [-1, -1, 1, 2, -0, -1, 1, 1],
            [2, -0, 2, -0, -1, 1, 1, -1],
            [-1, -0, -0, -1, -0, 2, 1, -1],
            [-3, 2, -4, -2, 2, 1, -1, -0]
        ])
        np.testing.assert_array_almost_equal(
            np.rint(dct2d(test_input)), expect
        )

    def test_idct2d(self):
        test_input = np.array([
            [1260, -1, -12, -5, 2, -2, -3, 1],
            [-23, -17, -6, -3, -3, -0, 0, -1],
            [-11, -9, -2, 2, 0, -1, -1, -0],
            [-7, -2, 0, 1, 1, -0, -0, 0],
            [-1, -1, 1, 2, -0, -1, 1, 1],
            [2, -0, 2, -0, -1, 1, 1, -1],
            [-1, -0, -0, -1, -0, 2, 1, -1],
            [-3, 2, -4, -2, 2, 1, -1, -0]
        ])
        expect = np.array([
            [139, 145, 149, 153, 155, 155, 155, 154],
            [144, 151, 153, 156, 159, 156, 156, 156],
            [151, 155, 160, 163, 158, 156, 156, 156],
            [159, 161, 162, 160, 160, 160, 159, 159],
            [159, 160, 161, 162, 162, 155, 155, 155],
            [161, 161, 161, 162, 160, 157, 157, 158],
            [162, 163, 161, 163, 162, 157, 157, 157],
            [162, 162, 161, 161, 163, 159, 158, 158]
        ])
        np.testing.assert_array_almost_equal(
            np.rint(idct2d(test_input)),
            expect
        )

    def test_invertible(self):
        test_input = np.linspace(0, 255, 64, dtype=int).reshape(8, 8)
        np.testing.assert_almost_equal(test_input, idct2d(dct2d(test_input)))


class TestQuantization(unittest.TestCase):
    def test_quantize(self):
        test_input = np.array([
            [236, -1, -12, -5, 2, -2, -3, 1],
            [-23, -17, -6, -3, -3, 0, 0, -1],
            [-11, -9, -2, 2, 0, -1, -1, 0],
            [-7, -2, 0, 1, 1, 0, 0, 0],
            [-1, -1, 1, 2, 0, -1, 1, 1],
            [2, 0, 2, 0, -1, 1, 1, -1],
            [-1, 0, 0, -1, 0, 2, 1, -1],
            [-3, 2, -4, -2, 2, 1, -1, 0]
        ])
        expect = np.array([
            [15, 0, -1, 0, 0, 0, 0, 0],
            [-2, -1, 0, 0, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        np.testing.assert_almost_equal(
            quantize(
                test_input,
                'y',
                quality=50,
                inverse=False
            ),
            expect,
            decimal=0
        )

    def test_inverse_quantize(self):
        test_input = np.array([
            [15, 0, -1, 0, 0, 0, 0, 0],
            [-2, -1, 0, 0, 0, 0, 0, 0],
            [-1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        expect = np.array([
            [240, 0, -10, 0, 0, 0, 0, 0],
            [-24, -12, 0, 0, 0, 0, 0, 0],
            [-14, -13, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        np.testing.assert_almost_equal(
            quantize(
                test_input,
                'y',
                quality=50,
                inverse=True
            ),
            expect,
            decimal=0
        )

    def test_invertible(self):
        test_input = np.array([
            [236, -1, -12, -5, 2, -2, -3, 1],
            [-23, -17, -6, -3, -3, 0, 0, -1],
            [-11, -9, -2, 2, 0, -1, -1, 0],
            [-7, -2, 0, 1, 1, 0, 0, 0],
            [-1, -1, 1, 2, 0, -1, 1, 1],
            [2, 0, 2, 0, -1, 1, 1, -1],
            [-1, 0, 0, -1, 0, 2, 1, -1],
            [-3, 2, -4, -2, 2, 1, -1, 0]
        ])
        np.testing.assert_array_almost_equal(
            test_input,
            quantize(
                quantize(test_input, 'y', inverse=False),
                'y', inverse=True
            )
        )
