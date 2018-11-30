from matplotlib import pyplot as plt
import numpy as np

from prototype_jpeg import compress, extract


def main():
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
    })
    for spec in specs:
        with open(spec['fn'], 'rb') as raw_file:
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
        show_raw_images(
            (spec['fn'], extracted),
            ((512, 512), (512, 512)),
            ('Original', 'Compressed and Extracted')
        )


def show_raw_images(images, sizes, titles=None, grey_level=False):
    if titles is None:
        titles = range(len(images))
    _, axarr = plt.subplots(1, len(images))

    for idx, (img, size, title) in enumerate(zip(images, sizes, titles)):
        if isinstance(img, str):
            with open(img, 'rb') as img_file:
                arr = np.fromfile(img_file, dtype=np.uint8)
        else:
            arr = np.array(img)
        arr.shape = size if grey_level else (*size, 3)
        if len(images) == 1:
            axarr.set_title(title)
            axarr.imshow(arr, cmap='gray', vmin=0, vmax=255)
        else:
            axarr[idx].set_title(title)
            axarr[idx].imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == '__main__':
    main()
