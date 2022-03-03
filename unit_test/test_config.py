import sys

import unittest
import easydl
import torch


class MyTestCase(unittest.TestCase):

    def test_change_config(self):
        from easydl.config import change_cmd_arguments
        from easydl.config import get_config_from_cmd
        change_cmd_arguments('debug', '1')

        v = get_config_from_cmd('debug', False)
        assert v == True

if __name__ == '__main__':
    unittest.main()