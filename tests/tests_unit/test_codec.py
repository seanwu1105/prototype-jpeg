import collections
import itertools
import unittest
from unittest import mock

import numpy as np

from prototype_jpeg.codec import (
    Encoder, Decoder, decode_huffman, encode_huffman, encode_differential,
    decode_differential, iter_zig_zag, inverse_iter_zig_zag, encode_run_length,
    decode_run_length, EOB, ZRL, DC, AC, LUMINANCE, CHROMINANCE,
    HUFFMAN_CATEGORY_CODEWORD
)


class TestEncoder(unittest.TestCase):
    def test_init_diff_dc(self):
        data = collections.OrderedDict((
            ('y', np.array([
                [[14, 1, 0, -1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]],
                [[44, -2, -2, 0, 0, 0, 0, 0],
                 [3, 0, -1, 0, 0, 0, 0, 0],
                 [-1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]],
                [[6, 0, 1, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
            ])),
            ('cb', np.array([
                [[-14, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]],
                [[6, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 1, 0, 0, 0, 0, 0, 0],
                 [1, -1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
            ])),
            ('cr', np.array([
                [[22, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]],
                [[11, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]],
                [[17, -1, 0, 0, 0, 0, 0, 0],
                 [-1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
            ]))
        ))
        expect = {
            LUMINANCE: (14, 30, -38),
            CHROMINANCE: (-14, 20, -6, 22, -11, 6)
        }
        self.assertDictEqual(Encoder(data).diff_dc, expect)

    def test_init_run_length_ac(self):
        data = collections.OrderedDict((
            ('y', np.array([
                [[14, 1, 0, -1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]],
                [[14, 1, 0, -1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, -99]]
            ])),
            ('cb', np.array([
                [[-14, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 99, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
            ])),
            ('cr', np.array([
                [[17, -1, 0, 0, 0, 0, 0, 0],
                 [-1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
            ]))
        ))
        expect = {
            LUMINANCE: [
                (0, 1), (0, 1), (3, -1), EOB,
                (0, 1), (0, 1), (3, -1), ZRL, ZRL, ZRL, (8, -99), EOB
            ],
            CHROMINANCE: [
                (1, 1), ZRL, (0, 99), EOB,
                (0, -1), (0, -1), (1, 1), EOB
            ]
        }
        self.assertDictEqual(Encoder(data).run_length_ac, expect)

    def test_encode(self):
        test_diff_dc = {
            LUMINANCE: (63, 2, -7, 3),
            CHROMINANCE: (15, 7)
        }
        test_run_length_ac = {
            LUMINANCE: [
                (0, -1), (2, -1), (0, 2), EOB,
                (1, -2), ZRL, (1, -1), EOB,
                (2, -1), ZRL, (0, -1), EOB,
                ZRL, ZRL, (1, 1), EOB
            ],
            CHROMINANCE: [
                (0, 1), ZRL, ZRL, (2, -1), EOB,
                EOB
            ]
        }
        expect = {
            DC: {
                LUMINANCE: '1110111111 01110 100000 01111'.replace(' ', ''),
                CHROMINANCE: '11101111 110111'.replace(' ', '')
            },
            AC: {
                LUMINANCE: ''.join((
                    '000', '111000', '0110', '1010',
                    '1101101', '11111111001', '11000', '1010',
                    '111000', '11111111001', '000', '1010',
                    '11111111001', '11111111001', '11001', '1010'
                )),
                CHROMINANCE: ''.join((
                    '011', '1111111010', '1111111010', '110100', '00',
                    '00'
                ))
            }
        }
        with mock.patch.object(Encoder, '__init__') as mock_Encoder_init:
            mock_Encoder_init.return_value = None
            encoder = Encoder(None)
            encoder.diff_dc = test_diff_dc
            encoder.run_length_ac = test_run_length_ac
            self.assertDictEqual(encoder.encode(), expect)


class TestDecoder(unittest.TestCase):
    def test_dc(self):
        test_instance = Decoder({
            DC: {
                LUMINANCE: '1110111111 01110 100000 01111'.replace(' ', ''),
                CHROMINANCE: '11101111 110111'.replace(' ', '')
            },
            AC: {
                LUMINANCE: ''.join((
                    '000', '111000', '0110', '1010',
                    '1101101', '11111111001', '11000', '1010',
                    '111000', '11111111001', '000', '1010',
                    '11111111001', '11111111001', '11001', '1010'
                )),
                CHROMINANCE: ''.join((
                    '011', '1111111010', '1111111010', '110100', '00',
                    '00'
                ))
            }
        })
        expect = {
            LUMINANCE: (63, 65, 58, 61),
            CHROMINANCE: (15, 22)
        }
        self.assertDictEqual(test_instance.dc, expect)

    def test_ac(self):
        test_instance = Decoder({
            DC: {
                LUMINANCE: '1110111111 01110 100000 01111'.replace(' ', ''),
                CHROMINANCE: '11101111 110111'.replace(' ', '')
            },
            AC: {
                LUMINANCE: ''.join((
                    '000', '111000', '0110', '1010',
                    '1101101', '11111111001', '11000', '1010',
                    '111000', '11111111001', '000', '1010',
                    '11111111001', '11111111001', '11001', '1010'
                )),
                CHROMINANCE: ''.join((
                    '011', '1111111010', '1111111010', '110100', '00',
                    '00'
                ))
            }
        })
        expect = {
            LUMINANCE: (
                (-1, 0, 0, -1, 2),
                (0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1),
                (0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1),
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
            ),
            CHROMINANCE: (
                (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1),
                ()
            )
        }
        self.assertDictEqual(test_instance.ac, expect)


class TestHuffmanCoding(unittest.TestCase):
    def test_encode_diff_dc_luminance_codeword(self):
        test_categories = (
            0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047,
            1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047
        )
        expect_categories = (
            '00', '0100', '01100', '100000', '1010000',
            '11000000', '1110000000', '111100000000',
            '11111000000000', '1111110000000000', '111111100000000000',
            '11111111000000000000', '0101', '01111', '100111',
            '1011111', '11011111', '1110111111', '111101111111',
            '11111011111111', '1111110111111111', '111111101111111111',
            '11111111011111111111'
        )
        for diff_dc, expect in zip(test_categories, expect_categories):
            self.assertEqual(encode_huffman(diff_dc, LUMINANCE),
                             expect)
        test_diff_values = (-3, -2, -1, 0, 1, 2, 3)
        expect_diff_values = (
            '01100', '01101', '0100',
            '00', '0101', '01110', '01111'
        )
        for diff_dc, expect in zip(test_diff_values, expect_diff_values):
            self.assertEqual(encode_huffman(diff_dc, LUMINANCE),
                             expect)

    def test_encode_diff_dc_chrominance_codeword(self):
        test_categories = (
            0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047,
            1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047
        )
        expect_categories = (
            '00', '010', '1000', '110000', '11100000',
            '1111000000', '111110000000', '11111100000000',
            '1111111000000000', '111111110000000000',
            '11111111100000000000', '1111111111000000000000', '011',
            '1011', '110111', '11101111', '1111011111',
            '111110111111', '11111101111111', '1111111011111111',
            '111111110111111111', '11111111101111111111',
            '1111111111011111111111'
        )
        for diff_dc, expect in zip(test_categories, expect_categories):
            self.assertEqual(encode_huffman(diff_dc, CHROMINANCE),
                             expect)

        test_diff_values = (-3, -2, -1, 0, 1, 2, 3)
        expect_diff_values = (
            '1000', '1001', '010',
            '00', '011', '1010', '1011'
        )
        for diff_dc, expect in zip(test_diff_values, expect_diff_values):
            self.assertEqual(encode_huffman(diff_dc, CHROMINANCE),
                             expect)

    def test_encode_diff_dc_out_of_range(self):
        test_inputs = (-2048, 2048)
        for val in test_inputs:
            with self.assertRaises(ValueError):
                encode_huffman(val, LUMINANCE)
            with self.assertRaises(ValueError):
                encode_huffman(val, CHROMINANCE)

    def test_encode_run_length_ac_luminance_codeword(self):
        test_inputs = ((1, -2), (0, -1), (2, -1), ZRL, (15, -1023), EOB)
        expects = (
            '1101101', '000', '111000', '11111111001',
            '11111111111111100000000000', '1010'
        )
        for run_length_ac, expect in zip(test_inputs, expects):
            self.assertEqual(encode_huffman(run_length_ac, LUMINANCE),
                             expect)

    def test_encode_run_length_ac_chrominance_codeword(self):
        test_inputs = ((1, -2), (0, 1), ZRL, (10, 1023), EOB)
        expects = (
            '11100101', '011', '1111111010',
            '11111111110100011111111111', '00'
        )
        for run_length_ac, expect in zip(test_inputs, expects):
            self.assertEqual(encode_huffman(run_length_ac, CHROMINANCE),
                             expect)

    def test_encode_run_length_ac_out_of_range(self):
        test_inputs = ((1, 0), (0, -1024), (0, 1024))
        for val in test_inputs:
            with self.assertRaises(ValueError):
                encode_huffman(val, LUMINANCE)
            with self.assertRaises(ValueError):
                encode_huffman(val, CHROMINANCE)

    def test_encode_run_length_ac_cannot_find_in_table(self):
        test_inputs = ((16, 1), (-1, 1))
        for val in test_inputs:
            with self.assertRaises(KeyError):
                encode_huffman(val, LUMINANCE)
            with self.assertRaises(KeyError):
                encode_huffman(val, CHROMINANCE)

    def test_decode_diff_dc_luminance_codeword(self):
        test_0 = '00'
        expect_0 = [0]
        self.assertSequenceEqual(
            list(decode_huffman(test_0, DC, LUMINANCE)),
            expect_0
        )
        test_27_len = ''.join(('1011111', '11111011111111'))
        expect_27_len = [15, 255]
        self.assertSequenceEqual(
            list(decode_huffman(test_27_len, DC, LUMINANCE)),
            expect_27_len
        )
        test_32_len = ''.join(('11111111011111111111', '111100000000'))
        expect_32_len = [2047, -127]
        self.assertSequenceEqual(
            list(decode_huffman(test_32_len, DC, LUMINANCE)),
            expect_32_len
        )

    def test_decode_diff_dc_chrominance_codeword(self):
        test_input = '1011'
        expect = [3]
        self.assertSequenceEqual(
            list(decode_huffman(test_input, DC, CHROMINANCE)),
            expect
        )

    def test_decode_run_length_ac_luminance_codeword(self):
        test_EOB = '1010'
        expect_EOB = [EOB]
        self.assertSequenceEqual(
            list(decode_huffman(test_EOB, AC, LUMINANCE)),
            expect_EOB
        )
        test_ZRL = '11111111001'
        expect_ZRL = [ZRL]
        self.assertSequenceEqual(
            list(decode_huffman(test_ZRL, AC, LUMINANCE)),
            expect_ZRL
        )
        test_26_len = '11111111111111100000000000'
        expect_26_len = [(15, -1023)]
        self.assertSequenceEqual(
            list(decode_huffman(test_26_len, AC, LUMINANCE)),
            expect_26_len
        )
        test_32_len = ''.join(('1111000111111', '1111111111010001000'))
        expect_32_len = [(0, 63), (11, -7)]
        self.assertSequenceEqual(
            list(decode_huffman(test_32_len, AC, LUMINANCE)),
            expect_32_len
        )

    def test_decode_run_length_ac_chrominance_codeword(self):
        test_EOB = '00'
        expect_EOB = [EOB]
        self.assertSequenceEqual(
            list(decode_huffman(test_EOB, AC, CHROMINANCE)),
            expect_EOB
        )
        test_ZRL = '1111111010'
        expect_ZRL = [ZRL]
        self.assertSequenceEqual(
            list(decode_huffman(test_ZRL, AC, CHROMINANCE)),
            expect_ZRL
        )
        test_26_len = '10111'
        expect_26_len = [(1, 1)]
        self.assertSequenceEqual(
            list(decode_huffman(test_26_len, AC, CHROMINANCE)),
            expect_26_len
        )

    def test_decode_cannot_find_in_table(self):
        test_input = '011011'
        with self.assertRaises(KeyError):
            tuple(decode_huffman(test_input, DC, LUMINANCE))
        test_input = '1011111'
        with self.assertRaises(KeyError):
            tuple(decode_huffman(test_input, DC, CHROMINANCE))
        test_input = '1000111'
        with self.assertRaises(KeyError):
            tuple(decode_huffman(test_input, AC, LUMINANCE))
        test_input = '100111'
        with self.assertRaises(KeyError):
            tuple(decode_huffman(test_input, AC, CHROMINANCE))

    def test_decode_error_fixed_code(self):
        test_input = '11010'
        with self.assertRaises(IndexError):
            tuple(decode_huffman(test_input, DC, LUMINANCE))
        test_input = '11010'
        with self.assertRaises(IndexError):
            tuple(decode_huffman(test_input, DC, CHROMINANCE))
        test_input = '11010'
        with self.assertRaises(IndexError):
            tuple(decode_huffman(test_input, AC, LUMINANCE))
        test_input = '11010'
        with self.assertRaises(IndexError):
            tuple(decode_huffman(test_input, AC, CHROMINANCE))


class TestDifferentialCoding(unittest.TestCase):
    def test_differential_encode(self):
        test_input = np.linspace(-128, 127, 10, dtype=int)
        expect = [-128, 29, 28, 28, 29, 27, 29, 28, 28, 29]
        self.assertSequenceEqual(encode_differential(test_input), expect)

    def test_differential_decode(self):
        test_input = [-128, 29, 28, 28, 29, 27, 29, 28, 28, 29]
        expect = np.linspace(-128, 127, 10, dtype=int).tolist()
        self.assertSequenceEqual(decode_differential(test_input), expect)

    def test_invertible(self):
        test_input = np.linspace(-128, 127, 10, dtype=int).tolist()
        self.assertSequenceEqual(
            test_input,
            decode_differential(encode_differential(test_input))
        )


class TestZigZag(unittest.TestCase):
    def test_zig_zag_scan_4x4(self):
        test_input = np.arange(16).reshape(4, 4)
        expect = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        self.assertSequenceEqual(list(iter_zig_zag(test_input)), expect)

    def test_zig_zag_scan_8x8(self):
        test_input = np.arange(64).reshape(8, 8)
        expect = [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12,
            19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
            42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
        ]
        self.assertSequenceEqual(list(iter_zig_zag(test_input)), expect)

    def test_inverse_zig_zag_scan(self):
        test_input = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        expect = np.arange(16).reshape(4, 4)
        np.testing.assert_array_equal(inverse_iter_zig_zag(test_input), expect)

    def test_invertible(self):
        test_input = np.arange(64).reshape(8, 8)
        np.testing.assert_array_equal(
            test_input,
            inverse_iter_zig_zag(list(iter_zig_zag(test_input)))
        )


class TestRunLengthCoding(unittest.TestCase):
    def test_run_length_encode(self):
        test_input_1 = (0, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0,
                        0, 0)
        expect_1 = [(1, -2), (0, -1), (0, -1), (0, -1),
                    ZRL, (11, -1), (0, -1), EOB]
        self.assertSequenceEqual(encode_run_length(test_input_1), expect_1)
        test_input_2 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99)
        expect_2 = [(0, 1), ZRL, (0, 99), EOB]
        self.assertSequenceEqual(encode_run_length(test_input_2), expect_2)

    def test_run_length_decode(self):
        test_input = [(1, -2), ZRL, (0, -1), ZRL, (2, -1), (0, -1), EOB]
        expect = (0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, -1, -1)
        result = decode_run_length(test_input)
        self.assertSequenceEqual(result, expect)

        test_EOB = [EOB]
        expect_EOB = ()
        result_EOB = decode_run_length(test_EOB)
        self.assertSequenceEqual(result_EOB, expect_EOB)

    def test_invertible(self):
        """Run length codec is invertible iff there is no trailing zero when
        encode.
        """

        test_input = (0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1,
                      -1)
        self.assertSequenceEqual(
            test_input,
            decode_run_length(encode_run_length(test_input))
        )


class TestHuffmanCategoryCodewordTable(unittest.TestCase):
    def test_dc_no_same_keys(self):
        for layer in (LUMINANCE, CHROMINANCE):
            test_input = HUFFMAN_CATEGORY_CODEWORD[DC][layer].keys()
            expect = {i for i in range(11 + 1)}
            self.assertSetEqual(set(test_input), expect)

    def test_ac_no_same_keys(self):
        for layer in (LUMINANCE, CHROMINANCE):
            test_input = HUFFMAN_CATEGORY_CODEWORD[AC][layer].keys()
            expect = set(itertools.product(range(15 + 1), range(1, 10 + 1)))
            expect.update((EOB, ZRL))
            self.assertSetEqual(set(test_input), expect)

    def test_dc_luminance_uniqueness(self):
        self.assertTrue(is_unique_decodable(
            HUFFMAN_CATEGORY_CODEWORD[DC][LUMINANCE].values()
        ))

    def test_dc_chrominance_uniqueness(self):
        self.assertTrue(is_unique_decodable(
            HUFFMAN_CATEGORY_CODEWORD[DC][CHROMINANCE].values()
        ))

    def test_ac_luminance_uniqueness(self):
        self.assertTrue(is_unique_decodable(
            HUFFMAN_CATEGORY_CODEWORD[AC][LUMINANCE].values()
        ))

    def test_ac_chrominance_uniqueness(self):
        self.assertTrue(is_unique_decodable(
            HUFFMAN_CATEGORY_CODEWORD[AC][CHROMINANCE].values()
        ))


def is_unique_decodable(codewords):
    # Step 1. Examine all paris of codewords to see if any codeword is a
    #         prefix of another.
    # Step 2. Whenever we find such a pair, add the dangling suffix to the
    #         list (unless it's already added).
    # Step 3. Repeat step 2. and 3. until
    #           Get a dangling suffix that is a codeword in the original
    #               list. --> NOT uniquely decodable.
    #           No more unique dangling suffix --> uniquely decodable.
    original = set(codewords)
    if len(codewords) != len(original):
        return False
    added = set()
    new_unique_dangling_suffix = True

    while new_unique_dangling_suffix:
        new_unique_dangling_suffix = False
        for pair in itertools.combinations(original.union(added), 2):
            pair = sorted(pair, key=len)
            if len(pair[0]) != len(pair[1]) and pair[1].startswith(pair[0]):
                dangling_suffix = pair[1].replace(pair[0], '', 1)
                if dangling_suffix in original:
                    return False
                if dangling_suffix not in added:
                    new_unique_dangling_suffix = True
                    added.add(dangling_suffix)
    return True
