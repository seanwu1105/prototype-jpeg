import itertools
import unittest

from bitarray import bitarray as b
import numpy as np

from prototype_jpeg.codec import (
    Encoder, Decoder, encode_huffman, encode_differential, decode_differential,
    iter_zig_zag, inverse_iter_zig_zag, encode_run_length, decode_run_length,
    EOB, DC, AC, LUMINANCE, CHROMINANCE, HUFFMAN_CATEGORY_CODEWORD
)


class TestEncoder(unittest.TestCase):
    def test_init_diff_dc(self):
        data = {
            'y': np.array([
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
            ]),
            'cb': np.array([
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
            ]),
            'cr': np.array([
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
            ])
        }
        expect = [14, 30, -38, -20, 20, -6, 22, -11, 6]
        self.assertSequenceEqual(Encoder(data).diff_dc, expect)

    def test_init_run_length_ac(self):
        data = {
            'y': np.array([
                [[14, 1, 0, -1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
            ]),
            'cb': np.array([
                [[-14, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
            ]),
            'cr': np.array([
                [[17, -1, 0, 0, 0, 0, 0, 0],
                 [-1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]]
            ])
        }
        expect = [
            [(0, 1), (0, 1), (3, -1), EOB],
            [(1, 1), EOB],
            [(0, -1), (0, -1), (1, 1), EOB]
        ]
        self.assertSequenceEqual(Encoder(data).run_length_ac, expect)

    def test_encode(self):
        # XXX: Remember to test multiple ZRL!
        pass


class TestDecoder(unittest.TestCase):
    # XXX: Remember to test ZRL!
    pass


class TestHuffmanEncoding(unittest.TestCase):
    def test_encode_diff_dc_luminance_codeword(self):
        test_categories = (
            0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047,
            1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047
        )
        expect_categories = (
            b('00'), b('0100'), b('01100'), b('100000'), b('1010000'),
            b('11000000'), b('1110000000'), b('111100000000'),
            b('11111000000000'), b('1111110000000000'), b('111111100000000000'),
            b('11111111000000000000'), b('0101'), b('01111'), b('100111'),
            b('1011111'), b('11011111'), b('1110111111'), b('111101111111'),
            b('11111011111111'), b('1111110111111111'), b('111111101111111111'),
            b('11111111011111111111')
        )
        for codeword, expect in zip(test_categories, expect_categories):
            self.assertEqual(encode_huffman(codeword, DC, LUMINANCE),
                             expect)
        test_diff_values = (-3, -2, -1, 0, 1, 2, 3)
        expect_diff_values = (
            b('01100'), b('01101'), b('0100'),
            b('00'), b('0101'), b('01110'), b('01111')
        )
        for codeword, expect in zip(test_diff_values, expect_diff_values):
            self.assertEqual(encode_huffman(codeword, DC, LUMINANCE),
                             expect)

    def test_encode_diff_dc_chrominance_codeword(self):
        test_categories = (
            0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047,
            1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047
        )
        expect_categories = (
            b('00'), b('010'), b('1000'), b('110000'), b('11100000'),
            b('1111000000'), b('111110000000'), b('11111100000000'),
            b('1111111000000000'), b('111111110000000000'),
            b('11111111100000000000'), b('1111111111000000000000'), b('011'),
            b('1011'), b('110111'), b('11101111'), b('1111011111'),
            b('111110111111'), b('11111101111111'), b('1111111011111111'),
            b('111111110111111111'), b('11111111101111111111'),
            b('1111111111011111111111')
        )
        for codeword, expect in zip(test_categories, expect_categories):
            self.assertEqual(encode_huffman(codeword, DC, CHROMINANCE),
                             expect)

        test_diff_values = (-3, -2, -1, 0, 1, 2, 3)
        expect_diff_values = (
            b('1000'), b('1001'), b('010'),
            b('00'), b('011'), b('1010'), b('1011')
        )
        for codeword, expect in zip(test_diff_values, expect_diff_values):
            self.assertEqual(encode_huffman(codeword, DC, CHROMINANCE),
                             expect)

    def test_encode_diff_dc_out_of_range(self):
        test_inputs = (-2048, 2048)
        for val in test_inputs:
            with self.assertRaises(ValueError):
                encode_huffman(val, DC, LUMINANCE)


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
        self.assertSequenceEqual(iter_zig_zag(test_input), expect)

    def test_zig_zag_scan_8x8(self):
        test_input = np.arange(64).reshape(8, 8)
        expect = [
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12,
            19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
            42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
        ]
        self.assertSequenceEqual(iter_zig_zag(test_input), expect)

    def test_inverse_zig_zag_scan(self):
        test_input = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        expect = np.arange(16).reshape(4, 4)
        np.testing.assert_array_equal(inverse_iter_zig_zag(test_input), expect)

    def test_invertible(self):
        test_input = np.arange(64).reshape(8, 8)
        np.testing.assert_array_equal(
            test_input,
            inverse_iter_zig_zag(iter_zig_zag(test_input))
        )


class TestRunLengthCoding(unittest.TestCase):
    def test_run_length_encode(self):
        test_input = (0, -2, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0)
        expect = [(1, -2), (0, -1), (0, -1), (0, -1), (2, -1), (0, -1), EOB]
        self.assertSequenceEqual(encode_run_length(test_input), expect)

    def test_run_length_decode(self):
        test_input = [(1, -2), (0, -1), (0, -1),
                      (0, -1), (2, -1), (0, -1), EOB]
        consumed_expect = (0, -2, -1, -1, -1, 0, 0, -1, -1)
        result = decode_run_length(test_input)
        self.assertSequenceEqual(tuple(result), consumed_expect)

    def test_invertible(self):
        """Run length codec is invertible iff there is no trailing zero when 
        encode.
        """

        test_input = (0, -2, -1, -1, -1, 0, 0, -1, -1)
        self.assertSequenceEqual(
            test_input,
            decode_run_length(encode_run_length(test_input))
        )


class TestHuffmanCategoryCodewordTableUniqueness(unittest.TestCase):
    def test_dc_luminance(self):
        self.assertTrue(test_unique_decodable(
            HUFFMAN_CATEGORY_CODEWORD[DC][LUMINANCE]
        ))

    def test_dc_chrominance(self):
        self.assertTrue(test_unique_decodable(
            HUFFMAN_CATEGORY_CODEWORD[DC][CHROMINANCE]
        ))

    def test_ac_luminance(self):
        self.assertTrue(test_unique_decodable(
            HUFFMAN_CATEGORY_CODEWORD[AC][LUMINANCE].values()
        ))

    def test_ac_chrominance(self):
        self.assertTrue(test_unique_decodable(
            HUFFMAN_CATEGORY_CODEWORD[AC][CHROMINANCE].values()
        ))


def test_unique_decodable(codewords):
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
