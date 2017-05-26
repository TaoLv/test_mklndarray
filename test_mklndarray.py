import sys
import theano
import numpy
import unittest

sys.path[0:0] = theano.config.compiledir
import mkl_ndarray   # noqa
from mkl_ndarray import mkl_ndarray as mkl   # noqa


class TestMKLNdarray(unittest.TestCase):

    def test_dtype_def(self):
        assert mkl.MKLNdarray.float32 == 11
        assert mkl.MKLNdarray.float64 == 12

    def test_zeros(self):
        a = mkl.MKLNdarray.zeros((4, 4), mkl.MKLNdarray.float32)

        assert a.shape == (4, 4)
        assert a.size == 16
        assert a.dtype == 'float32'
        assert a.ndim == 2

    def test_to_ndarray(self):
        a = mkl.MKLNdarray.zeros((4, 4), mkl.MKLNdarray.float32)
        b = numpy.zeros((4, 4)).astype(numpy.float32)
        c = numpy.asarray(a)   # a.__array__()

        assert numpy.allclose(b, c)

    def test_from_ndrray(self):
        a = numpy.asarray([[2.0, 3.0], [4.0, 5.0]]).astype(numpy.float32)
        b = mkl.MKLNdarray(a)

        assert b.shape == (2, 2)
        assert b.size == 4
        assert b.dtype == 'float32'
        assert b.ndim == 2

        c = numpy.asarray(b)

        assert numpy.allclose(a, c)

    def test_copy(self):
        a = mkl.MKLNdarray.zeros((4, 4), mkl.MKLNdarray.float32)
        b = numpy.zeros((4, 4)).astype(numpy.float32)

        c = a.__copy__()

        assert c.shape == (4, 4)
        assert c.size == 16
        assert c.dtype == 'float32'
        assert c.ndim == 2

        assert a.base is None
        assert c.base is a

        d = numpy.asarray(c)
        assert numpy.allclose(b, d)


if __name__ == '__main__':
    unittest.main()
