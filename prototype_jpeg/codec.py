import itertools

from bitarray import bitarray as b
import numpy as np


EOB = (0, 0)
ZRL = (15, 0)
DC = 'DC'
AC = 'AC'
LUMINANCE = 'luminance'
CHROMINANCE = 'chrominance'


class Encoder:
    def __init__(self, data):
        self.data = data

        # The differential DC in blocks order: 'y', 'cb', 'cr'.
        self.diff_dc = encode_differential(self.dc)

        # The run-length-encoded AC in blocks order: 'y', 'cb', 'cr'. Every
        #   sublist is the run-length-encoded AC of one block.
        self.run_length_ac = [encode_run_length(iter_zig_zag(b)[1:])
                              for l in self.data.values()
                              for b in l]

    @property
    def dc(self):
        return tuple(b[0][0] for l in self.data.values() for b in l)

    def encode(self):
        # TODO: encode DCC and ACC and return a dict.
        pass


class Decoder:
    def __init__(self, byte_seq):
        pass

    def decode(self):
        # TODO: decode DCC and ACC and return a dict.
        pass
        # Next 16 bits searching:
        # try:
        #   HUFFMAN_CATEGORY_CODEWORD[value_type][layer_type].index(category_bits)
        # except:
        #   Remove one bits and search again till find matching.
        # After found, consume the matching bits (category code word) and DIFF
        # value code word (bits with size, index of category).


def encode_huffman(val, value_type, layer_type):
    if val <= -2048 or val >= 2048:
        raise ValueError('Differential DC should be within [-2047, 2047].')
    size, diff_idx = index_2d(HUFFMAN_CATEGORIES, val)
    if size == 0:
        return b(HUFFMAN_CATEGORY_CODEWORD[value_type][layer_type][size])
    return b(HUFFMAN_CATEGORY_CODEWORD[value_type][layer_type][size]
             + '{:0{padding}b}'.format(diff_idx, padding=size))


def encode_differential(seq):
    return tuple(
        (item - seq[idx - 1]) if idx else item
        for idx, item in enumerate(seq)
    )


def decode_differential(seq):
    return tuple(itertools.accumulate(seq))


def encode_run_length(seq):
    groups = [(len(tuple(group)), key)
              for key, group in itertools.groupby(seq)]
    ret = []
    borrow = False
    if groups[-1][1] == 0:
        del groups[-1]
    for idx, (length, key) in enumerate(groups):
        if borrow == True:
            length -= 1
            borrow = False
        if length == 0:
            continue
        if key == 0:
            ret.append((length, groups[idx + 1][1]))
            borrow = True
        else:
            ret.extend(((0, key), ) * length)
    return ret + [EOB]


def decode_run_length(seq):
    # Remove the last element as the last created by EOB would always be a `0`.
    return tuple(item for l, k in seq for item in [0] * l + [k])[:-1]


def iter_zig_zag(data):
    if data.shape[0] != data.shape[1]:
        raise ValueError('The shape of input array should be square.')
    ret = []
    x, y = 0, 0
    for _ in np.nditer(data):
        ret.append(data[y][x])
        if (x + y) % 2 == 1:
            x, y = move_zig_zag_idx(x, y, data.shape[0])
        else:
            y, x = move_zig_zag_idx(y, x, data.shape[0])
    return ret


def inverse_iter_zig_zag(seq):
    if not (len(seq) ** 0.5).is_integer():
        raise ValueError('The length of input sequence should be perfect '
                         'square.')
    size = int(len(seq) ** 0.5)
    ret = np.empty((size, size), dtype=int)
    x, y = 0, 0
    for value in seq:
        ret[y][x] = value
        if (x + y) % 2 == 1:
            x, y = move_zig_zag_idx(x, y, size)
        else:
            y, x = move_zig_zag_idx(y, x, size)
    return ret


def move_zig_zag_idx(i, j, size):
    if j < (size - 1):
        return (max(0, i - 1), j + 1)
    return (i + 1, j)


def index_2d(table, target):
    for i, row in enumerate(table):
        for j, element in enumerate(row):
            if target == element:
                return (i, j)
    raise ValueError('Cannot find the target value in the table.')


HUFFMAN_CATEGORIES = (
    (0, ),
    (-1, 1),
    (-3, -2, 2, 3),
    (*range(-7, -4 + 1), *range(4, 7 + 1)),
    (*range(-15, -8 + 1), *range(8, 15 + 1)),
    (*range(-31, -16 + 1), *range(16, 31 + 1)),
    (*range(-63, -32 + 1), *range(32, 63 + 1)),
    (*range(-127, -64 + 1), *range(64, 127 + 1)),
    (*range(-255, -128 + 1), *range(128, 255 + 1)),
    (*range(-511, -256 + 1), *range(256, 511 + 1)),
    (*range(-1023, -512 + 1), *range(512, 1023 + 1)),
    (*range(-2047, -1024 + 1), *range(1024, 2047 + 1)),
    (*range(-4095, -2048 + 1), *range(2048, 4095 + 1)),
    (*range(-8191, -4096 + 1), *range(4096, 8191 + 1)),
    (*range(-16383, -8192 + 1), *range(8192, 16383 + 1)),
    (*range(-32767, -16384 + 1), *range(16384, 32767 + 1))
)

HUFFMAN_CATEGORY_CODEWORD = {
    DC: {
        LUMINANCE: (
            '00',
            '010',
            '011',
            '100',
            '101',
            '110',
            '1110',
            '11110',
            '111110',
            '1111110',
            '11111110',
            '111111110'
        ),
        CHROMINANCE: (
            '00',
            '01',
            '10',
            '110',
            '1110',
            '11110',
            '111110',
            '1111110',
            '11111110',
            '111111110',
            '1111111110',
            '11111111110'
        )
    },
    AC: {
        LUMINANCE: {
            EOB: '1010',

            (0, 1):  '00',
            (0, 2):  '01',
            (0, 3):  '100',
            (0, 4):  '1011',
            (0, 5):  '11010',
            (0, 6):  '1111000',
            (0, 7):  '11111000',
            (0, 8):  '1111110110',
            (0, 9):  '1111111110000010',
            (0, 10): '1111111110000011',

            (1, 1):  '1100',
            (1, 2):  '11011',
            (1, 3):  '1111001',
            (1, 4):  '111110110',
            (1, 5):  '11111110110',
            (1, 6):  '1111111110000100',
            (1, 7):  '1111111110000101',
            (1, 8):  '1111111110000110',
            (1, 9):  '1111111110000111',
            (1, 10): '1111111110001000',

            (2, 1):  '11100',
            (2, 2):  '11111001',
            (2, 3):  '1111110111',
            (2, 4):  '111111110100',
            (2, 5):  '1111111110001001',
            (2, 6):  '1111111110001010',
            (2, 7):  '1111111110001011',
            (2, 8):  '1111111110001100',
            (2, 9):  '1111111110001101',
            (2, 10): '1111111110001110',

            (3, 1):  '',
            (3, 2):  '',
            (3, 3):  '',
            (3, 4):  '',
            (3, 5):  '',
            (3, 6):  '',
            (3, 7):  '',
            (3, 8):  '',
            (3, 9):  '',
            (3, 10): '',

            (4, 1):  '',
            (4, 2):  '',
            (4, 3):  '',
            (4, 4):  '',
            (4, 5):  '',
            (4, 6):  '',
            (4, 7):  '',
            (4, 8):  '',
            (4, 9):  '',
            (4, 10): '',

            (5, 1):  '',
            (5, 2):  '',
            (5, 3):  '',
            (5, 4):  '',
            (5, 5):  '',
            (5, 6):  '',
            (5, 7):  '',
            (5, 8):  '',
            (5, 9):  '',
            (5, 10): '',

            (6, 1):  '',
            (6, 2):  '',
            (6, 3):  '',
            (6, 4):  '',
            (6, 5):  '',
            (6, 6):  '',
            (6, 7):  '',
            (6, 8):  '',
            (6, 9):  '',
            (6, 10): '',

            (7, 1):  '',
            (7, 2):  '',
            (7, 3):  '',
            (7, 4):  '',
            (7, 5):  '',
            (7, 6):  '',
            (7, 7):  '',
            (7, 8):  '',
            (7, 9):  '',
            (7, 10): '',

            (8, 1):  '',
            (8, 2):  '',
            (8, 3):  '',
            (8, 4):  '',
            (8, 5):  '',
            (8, 6):  '',
            (8, 7):  '',
            (8, 8):  '',
            (8, 9):  '',
            (8, 10): '',

            (9, 1):  '',
            (9, 2):  '',
            (9, 3):  '',
            (9, 4):  '',
            (9, 5):  '',
            (9, 6):  '',
            (9, 7):  '',
            (9, 8):  '',
            (9, 9):  '',
            (9, 10): '',

            (10, 1):  '',
            (10, 2):  '',
            (10, 3):  '',
            (10, 4):  '',
            (10, 5):  '',
            (10, 6):  '',
            (10, 7):  '',
            (10, 8):  '',
            (10, 9):  '',
            (10, 10): '',

            (11, 1):  '',
            (11, 2):  '',
            (11, 3):  '',
            (11, 4):  '',
            (11, 5):  '',
            (11, 6):  '',
            (11, 7):  '',
            (11, 8):  '',
            (11, 9):  '',
            (11, 10): '',

            (12, 1):  '',
            (12, 2):  '',
            (12, 3):  '',
            (12, 4):  '',
            (12, 5):  '',
            (12, 6):  '',
            (12, 7):  '',
            (12, 8):  '',
            (12, 9):  '',
            (12, 10): '',

            (13, 1):  '',
            (13, 2):  '',
            (13, 3):  '',
            (13, 4):  '',
            (13, 5):  '',
            (13, 6):  '',
            (13, 7):  '',
            (13, 8):  '',
            (13, 9):  '',
            (13, 10): '',

            (14, 1):  '',
            (14, 2):  '',
            (14, 3):  '',
            (14, 4):  '',
            (14, 5):  '',
            (14, 6):  '',
            (14, 7):  '',
            (14, 8):  '',
            (14, 9):  '',
            (14, 10): '',

            (15, 1):  '',
            (15, 2):  '',
            (15, 3):  '',
            (15, 4):  '',
            (15, 5):  '',
            (15, 6):  '',
            (15, 7):  '',
            (15, 8):  '',
            (15, 9):  '',
            (15, 10): '',

            ZRL: '11111111001'
        },
        CHROMINANCE: {
            EOB: '00',

            (0, 1):  '',
            (0, 2):  '',
            (0, 3):  '',
            (0, 4):  '',
            (0, 5):  '',
            (0, 6):  '',
            (0, 7):  '',
            (0, 8):  '',
            (0, 9):  '',
            (0, 10): '',

            (1, 1):  '',
            (1, 2):  '',
            (1, 3):  '',
            (1, 4):  '',
            (1, 5):  '',
            (1, 6):  '',
            (1, 7):  '',
            (1, 8):  '',
            (1, 9):  '',
            (1, 10): '',

            (2, 1):  '',
            (2, 2):  '',
            (2, 3):  '',
            (2, 4):  '',
            (2, 5):  '',
            (2, 6):  '',
            (2, 7):  '',
            (2, 8):  '',
            (2, 9):  '',
            (2, 10): '',

            (3, 1):  '',
            (3, 2):  '',
            (3, 3):  '',
            (3, 4):  '',
            (3, 5):  '',
            (3, 6):  '',
            (3, 7):  '',
            (3, 8):  '',
            (3, 9):  '',
            (3, 10): '',

            (4, 1):  '',
            (4, 2):  '',
            (4, 3):  '',
            (4, 4):  '',
            (4, 5):  '',
            (4, 6):  '',
            (4, 7):  '',
            (4, 8):  '',
            (4, 9):  '',
            (4, 10): '',

            (5, 1):  '',
            (5, 2):  '',
            (5, 3):  '',
            (5, 4):  '',
            (5, 5):  '',
            (5, 6):  '',
            (5, 7):  '',
            (5, 8):  '',
            (5, 9):  '',
            (5, 10): '',

            (6, 1):  '',
            (6, 2):  '',
            (6, 3):  '',
            (6, 4):  '',
            (6, 5):  '',
            (6, 6):  '',
            (6, 7):  '',
            (6, 8):  '',
            (6, 9):  '',
            (6, 10): '',

            (7, 1):  '',
            (7, 2):  '',
            (7, 3):  '',
            (7, 4):  '',
            (7, 5):  '',
            (7, 6):  '',
            (7, 7):  '',
            (7, 8):  '',
            (7, 9):  '',
            (7, 10): '',

            (8, 1):  '',
            (8, 2):  '',
            (8, 3):  '',
            (8, 4):  '',
            (8, 5):  '',
            (8, 6):  '',
            (8, 7):  '',
            (8, 8):  '',
            (8, 9):  '',
            (8, 10): '',

            (9, 1):  '',
            (9, 2):  '',
            (9, 3):  '',
            (9, 4):  '',
            (9, 5):  '',
            (9, 6):  '',
            (9, 7):  '',
            (9, 8):  '',
            (9, 9):  '',
            (9, 10): '',

            (10, 1):  '',
            (10, 2):  '',
            (10, 3):  '',
            (10, 4):  '',
            (10, 5):  '',
            (10, 6):  '',
            (10, 7):  '',
            (10, 8):  '',
            (10, 9):  '',
            (10, 10): '',

            (11, 1):  '',
            (11, 2):  '',
            (11, 3):  '',
            (11, 4):  '',
            (11, 5):  '',
            (11, 6):  '',
            (11, 7):  '',
            (11, 8):  '',
            (11, 9):  '',
            (11, 10): '',

            (12, 1):  '',
            (12, 2):  '',
            (12, 3):  '',
            (12, 4):  '',
            (12, 5):  '',
            (12, 6):  '',
            (12, 7):  '',
            (12, 8):  '',
            (12, 9):  '',
            (12, 10): '',

            (13, 1):  '',
            (13, 2):  '',
            (13, 3):  '',
            (13, 4):  '',
            (13, 5):  '',
            (13, 6):  '',
            (13, 7):  '',
            (13, 8):  '',
            (13, 9):  '',
            (13, 10): '',

            (14, 1):  '',
            (14, 2):  '',
            (14, 3):  '',
            (14, 4):  '',
            (14, 5):  '',
            (14, 6):  '',
            (14, 7):  '',
            (14, 8):  '',
            (14, 9):  '',
            (14, 10): '',

            (15, 1):  '',
            (15, 2):  '',
            (15, 3):  '',
            (15, 4):  '',
            (15, 5):  '',
            (15, 6):  '',
            (15, 7):  '',
            (15, 8):  '',
            (15, 9):  '',
            (15, 10): '',

            ZRL: '1111111010'
        }
    }
}
