import logging
from matplotlib import pyplot as plt
import numpy as np

from prototype_jpeg import compress, extract
from prototype_jpeg.utils import psnr
from tests.test_integration import compress_and_extract

logging.basicConfig(level=logging.INFO)


def read_img(fn):
    with open(fn, 'rb') as f:
        ret = np.fromfile(f, dtype=np.uint8)
    return ret


filenames = (
    'tests/images/rgb/Baboon.raw',
    'tests/images/rgb/Lena.raw',
    'tests/images/grey_level/Baboon.raw',
    'tests/images/grey_level/Lena.raw'
)

qualities = (90, 80, 50, 20, 10, 5)


def main():
    (_, axarr) = plt.subplots(len(filenames), len(qualities) + 1)

    for i, fn in enumerate(filenames):
        logging.getLogger(__name__).info('-- FN: %s ------------' % fn)
        original = read_img(fn).reshape(
            (512, 512) if ('grey_level' in fn) else (512, 512, 3)
        )
        axarr[i][0].imshow(original, cmap='gray', vmin=0, vmax=255)
        for j, q in enumerate(qualities):
            logging.getLogger(__name__).info('------- QF: %s -------' % q)
            extracted = compress_and_extract({
                'fn': fn,
                'size': (512, 512),
                'grey_level': ('grey_level' in fn),
                'quality': q,
                'subsampling_mode': 1
            }).reshape(
                (512, 512) if ('grey_level' in fn) else (512, 512, 3)
            )
            logging.getLogger(__name__).info(
                '---------- PSNR: %.8f ----' % psnr(original, extracted)
            )
            axarr[i][j + 1].imshow(extracted, cmap='gray', vmin=0, vmax=255)

    plt.show()


if __name__ == '__main__':
    main()
