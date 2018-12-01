import unittest
import unittest.mock

from main import main

class TestMain(unittest.TestCase):
    @unittest.mock.patch('main.show_raw_images')
    def test_main(self):
        assert main() is None
