# Prototype JPEG

[![pipeline status](https://gitlab.com/GLaDOS1105/prototype-jpeg/badges/master/pipeline.svg)](https://gitlab.com/GLaDOS1105/prototype-jpeg/commits/master)
[![coverage report](https://gitlab.com/GLaDOS1105/prototype-jpeg/badges/master/coverage.svg)](https://gitlab.com/GLaDOS1105/prototype-jpeg/commits/master)

A prototype JPEG compressor in Python.

JPEG HUFFMAN Table source: http://dirac.epucfe.eu/projets/wakka.php?wiki=P14AB08/download&file=P14AB08_JPEG_ALGORITH_BASELINE_ON_EMBEDDED_SYSTEMS.pdf

TODO:
unittest for grey level encoding and decoding.
name 'y' 'cb' 'cr' to CONST

NOTE: Large file might overflow the memory.