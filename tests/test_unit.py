import unittest

import numpy as np

from prototype_jpeg.utils import (rgb2ycbcr, ycbcr2rgb, downsample,
                                  upsample)


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
