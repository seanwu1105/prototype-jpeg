import unittest
import unittest.mock

from example import example


class TestExample(unittest.TestCase):
    @unittest.mock.patch('example.show_raw_images')
    def test_main(self, mock_main_show_raw_image):
        assert example() is None
