# [Prototype JPEG](https://github.com/seanwu1105/prototype-jpeg)

[![pipeline status](https://gitlab.com/seanwu1105/prototype-jpeg/badges/master/pipeline.svg)](https://gitlab.com/seanwu1105/prototype-jpeg/commits/master)
[![coverage report](https://gitlab.com/seanwu1105/prototype-jpeg/badges/master/coverage.svg)](https://gitlab.com/seanwu1105/prototype-jpeg/commits/master)
[![Requirements Status](https://requires.io/github/seanwu1105/prototype-jpeg/requirements.svg?branch=master)](https://requires.io/github/seanwu1105/prototype-jpeg/requirements/?branch=master)

A prototype JPEG compressor in Python.

![preview](https://i.imgur.com/KwD7c1u.jpg "preview")

## Get Started

### Installation

Clone this repositroy.

``` bash
git clone https://github.com/seanwu1105/prototype-jpeg
```

Open the root directory.

``` bash
cd prototype_jpeg
```

Install the dependencies.

``` bash
pip install -r requirements.txt
```

### Uses

Show the compressed file results.

``` bash
python main.py
```

Or to run some examples,

``` bash
python example.py
```

Inside `example.py`, you could find how to compress a file with following script.

``` python
with open(spec['fn'], 'rb') as raw_file:
    original = np.fromfile(raw_file, dtype=np.uint8)
    raw_file.seek(0)
    compressed = compress(
        raw_file,
        # Set image spec.
        size=spec['size'],
        grey_level=spec['grey_level'],
        quality=spec['quality'],
        subsampling_mode=spec['subsampling_mode']
    )
with open('compressed.protojpg', 'wb') as compressed_file:
    compressed['data'].tofile(compressed_file)
header = compressed['header']  # Get compressed header (metadata).
```

And extract a compressed file.

``` python
with open('compressed.protojpg', 'rb') as compressed_file:
    extracted = extract(
        compressed_file,
        # Set extract spec (metadata).
        header={
            'size': header['size'],
            'grey_level': header['grey_level'],
            'quality': header['quality'],
            'subsampling_mode': header['subsampling_mode'],
            'remaining_bits_length': header['remaining_bits_length'],
            'data_slice_lengths': header['data_slice_lengths']
        }
    )
```

### Image Spec

You can set the following image compression spec for compression and extraction.

|        Spec        |        Type       | Details                                                                                                                                                 |
|:------------------:|:-----------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------|
|       `size`       | `tuple(int, int)` | The size of image.                                                                                                                                      |
|    `grey_level`    |       `bool`      | Grey level (`True`) or RGB (`False`) image.                                                                                                             |
|      `quality`     |       `int`       | Baseline JPEG quality factor.                                                                                                                           |
| `subsampling_mode` |  `1`, `2` or `4`  | Subsampling modes. Luminance:Chrominance = 4:`subsampling_mode`. The `subsampling_mode` in spec will not have any effect if `grey_level` is set `True`. |

Compressing process would automatically generate the following 2 more items in the header. You should put these items into `extract()` as well.

|           Spec          |              Type              | Details                                                                                                              |
|:-----------------------:|:------------------------------:|----------------------------------------------------------------------------------------------------------------------|
| `remaining_bits_length` |              `int`             | The length of remaining (appending) bits to fill up to `8n` bits for minimal byte size in order to save into a file. |
|   `data_slice_lengths`  | `tuple(int, int, [, int, int])` | Record the length of luminance and chrominance DC/AC bits for extracting process.                                    |

## Development

### Tests

Use the following command to run unit and integrating tests.

``` bash
nose2 -v --with-coverage tests
```

### Dependencies

* [bidict](https://bidict.readthedocs.io/en/master/)
* [bitarray](https://pypi.org/project/bitarray/)
* [Matplotlib](https://matplotlib.org/)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)

## Compressed File Results

The PSNR gets higher, compressing/extracting time elapsing gets longer and file size gets smaller with higher QF.

![analysis](https://i.imgur.com/Qsjm11x.png "analysis")
> For further details, see [this report](https://docs.google.com/document/d/1psiuwJxcRnHfEcb9GTTmYd2JbMJ2z6h0kz8B4l03k_0/edit?usp=sharing).

## Algorithms and Implementation

### Bits Storage

Since the algorithm (especially for the extracting procedure) require many times to remove the last element in an iterable, we use strings to save any bit sequence (including the data of Huffman table) for both developing and run-time speed. Actually, lists could be a better choice regarding deletion [performance](https://gist.github.com/GLaDOS1105/6c348e8e3c02d3a07fe583e15cffe96a). However, strings have better performance and cleaner codes for the conversion between integer (decimal number) and bit sequence.

> Saving bit sequence into strings or lists could lead to memory insufficiency as one bit need a byte (character or boolean in Python) to store.

### RGB and Grey Level

The compression will treat grey level images as a luminance (Y) layer in RGB images. Thus, it would not be subsampled. Namely, the `subsampling_mode` in spec will not have any effect if `grey_level` is set `True`.

### Level Offset

After the color space conversion, Cb and Cr would be in the range `[-128, 127]` but Y in `[0, 255]`. Hence, layer Y would minus `128` in order to have the same range.

### Subsampling Modes

Only chrominance (Cb and Cr) would run through subsampling process. The following is the example of subsampled pixels.

![subsampled pixel](https://i.imgur.com/NcrtglH.png "subsampled pixel")

``` python
def downsample(arr, mode):
    """Downsample an 2D array.
    Arguments:
        arr {2d numpy array} -- The target array.
        mode {1 or 2 or 4} -- Downsample ratio (4:mode).
    Returns:
        2d numpy array -- Downsampled array.
    """
    if mode not in {1, 2, 4}:
        raise ValueError(f'Mode ({mode}) must be 1, 2 or 4.')
    if mode == 4:
        return arr
    return arr[::3 - mode, ::2]
```

### Slicing

In order to slice a 2D mn pixel array into 3D `k * 8 * 8` blocks sequence **evenly**, the 2D pixel would be padded up to `8N * 8N` with `0`.

``` python
# Pad Layers to 8N * 8N
data[key] = np.pad(
    layer,
    (
        (0, (nrows // 8 + 1) * 8 - nrows if nrows % 8 else 0),
        (0, (ncols // 8 + 1) * 8 - ncols if ncols % 8 else 0)
    ),
    mode='constant'
)
```

``` python
def block_slice(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    `reshape` will raise a `ValueError` if `nrows` or `ncols` doesn't evenly
    divide the shape.
    """
    h, _ = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))
```

### Quality Factor

The range of quality factor QF is `(0, 95]` because:

If `QF = 0`, by the definition of quantization, `Q = S * f(QF) / 100`, where `Q` is quantization table, `S` is default quantization table (for baseline JPEG)

![default quantization table](https://i.imgur.com/FkV6cUA.png "default quantization table")

and

![function](https://latex.codecogs.com/png.latex?%5Clarge%20f%28QF%29%20%3D%20%5Cbegin%7Bcases%7D%20%5Cfrac%7B5000%7D%7BQF%7D%2C%20%26%20%5Ctext%7Bif%20%7D%20QF%20%3C%2050%5C%5C200%20-%202QF%2C%20%26%20%5Ctext%7Bif%20%7D%20QF%20%5Cgeq%2050%20%5Cend%7Bcases%7D "f(QF) function")

it would cause division-by-zero error.

If `QF < 0`, the quantization table would have negative elements.

If there is an element smaller than `1` in quantization table, after the division in quantization process, the corresponding element in the result image block would become larger, which violates the goal of quantization. Assume `QF = 95`, `Q = S / 10`. The minimal element in `S` is `10`, which would be `1` in `Q` after the division, and if `QF > 95`, the minimal element in `Q` would be smaller than `1`. Thus, QF should always be smaller than or equal to `95`.

### Baseline JPEG Huffman Tables

The baseline JPEG Huffman table could be found in [http://dirac.epucfe.eu](http://dirac.epucfe.eu/projets/wakka.php?wiki=P14AB08/download&file=P14AB08_JPEG_ALGORITH_BASELINE_ON_EMBEDDED_SYSTEMS.pdf), which has the same Huffman coding for DC and AC luminance, as well as DC chrominance in the class slides. However, that Huffman table also includes the AC chrominance coding, which could decrease the compressed file size significantly. You can find the complete table in `/prototype_jpeg/codec.py`.

``` python
CHROMINANCE: bidict({
    EOB: '00',  # (0, 0)
    ZRL: '1111111010',  # (F, 0)
    (0, 1):  '01',
    (0, 2):  '100',
    (0, 3):  '1010',
    (0, 4):  '11000',
    (0, 5):  '11001',
    (0, 6):  '111000',
    (0, 7):  '1111000',
    (0, 8):  '111110100',
    (0, 9):  '1111110110',
    (0, 10): '111111110100',
    # …
})
```

Furthermore, for the high performance, we save the Huffman table as a **bidirectional hash table** since every key and value in the Huffman table is **unique**. For the uniqueness (unique decodable) test, you can find the test in `/tests/tests_unit/test_codec.py`.
