import unittest

import numpy as np

from prototype_jpeg.utils import (rgb2ycbcr, ycbcr2rgb, downsample,
                                  upsample)


class TestColorSpaceConversion(unittest.TestCase):
    def test_rgb2ycbcr(self):
        test_input = np.linspace(0, 255, 24, dtype=int).reshape(2, 4, 3)
        expect = {
            'y': np.array([
                [8.965,  41.965,  74.965, 107.965],
                [141.965, 174.965, 207.965, 241.079]
            ]),
            'cb': np.array([
                [135.356096, 135.356096, 135.356096, 135.356096],
                [135.356096, 135.356096, 135.356096, 135.856096]
            ]),
            'cr': np.array([
                [121.605568, 121.605568, 121.605568, 121.605568],
                [121.605568, 121.605568, 121.605568, 121.524256]
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
                [-148.36817391,  -68.47556522,   11.41704348,   91.30965217],
                [171.20226087,  251.09486957,  330.98747826,  410.88008696]
            ]),
            'g': np.array([
                [115.80820557, 113.87002817, 111.93185078, 109.99367339],
                [108.055496, 106.11731861, 104.17914122, 102.24096383]
            ]),
            'b': np.array([
                [-207.16991304, -114.97078261,  -22.77165217,   69.42747826],
                [161.6266087,  253.82573913,  346.02486957,  438.224]
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
