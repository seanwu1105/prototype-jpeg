import unittest
import unittest.mock

from main import main

class TestMain(unittest.TestCase):
    @unittest.mock.patch('main.show_raw_images')
    def test_main(self, mock_main_show_raw_image):
        assert main() is None
