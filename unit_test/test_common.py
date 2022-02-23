import unittest
import easydl
import torch


class MyTestCase(unittest.TestCase):

    def test_get_cpu_count(self):
        import multiprocessing
        print(multiprocessing.cpu_count())


if __name__ == '__main__':
    unittest.main()

