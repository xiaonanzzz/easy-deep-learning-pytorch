import sys

import unittest
import easydl
import torch


class MyTestCase(unittest.TestCase):

    def test_image_classification(self):
        from easydl_example.cifar import train_cifar
        from easydl.config import change_cmd_arguments
        change_cmd_arguments('project_name', 'debug')
        change_cmd_arguments('debug', 'True')
        train_cifar()

if __name__ == '__main__':
    unittest.main()