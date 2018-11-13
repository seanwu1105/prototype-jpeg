def rgb2ycbcr(r, g, b):
    return {
        'y': tuple(.299 * p[0] + .587 * p[1] + .144 * p[2] for p in zip(r, g, b)),
        'cb': tuple(-.168 * p[0] - .331 * p[1] - .449 * p[2] for p in zip(r, g, b)),
        'cr': tuple(.5 * p[0] - .419 * p[1] - .081 * p[2] for p in zip(r, g, b))
    }

def ycbcr2rgb():
    pass

def subsample():
    pass

def inverse_subsample():
    pass
