import unittest

import torch

from easydl.trainer.metrics import recall_in_k_pytorch, recall_in_k_sklearn


class MyTestCase(unittest.TestCase):

    def test_self_retrieval(self):
        qn, gn, d = 12, 12, 64
        qx = torch.randn((qn, d))
        qy = torch.randint(0, 3, (qn,) )

        ret1 = recall_in_k_pytorch(qx, qy, qx, qy, k_list=[1, 2, 3, 4, 5])
        ret2 = recall_in_k_sklearn(qx, qy, qx, qy, ks=[1, 2, 3, 4, 5])

        for k in range(1, 6):
            self.assertEqual(ret1[k], 1.0)  # add assertion here
            self.assertEqual(ret2[k], 1.0)

    def test_something(self):

        qn, gn, d = 3, 12, 64
        qx = torch.randn((qn, d))
        qy = torch.randint(0, 3, (qn,) )
        ix, iy = torch.randn((gn, d)), torch.randint(0, 3, (gn,) )

        ret1 = recall_in_k_pytorch(qx, qy, ix, iy, k_list=[1, 2, 3, 4, 5])
        ret2 = recall_in_k_sklearn(qx, qy, ix, iy, ks=[1, 2, 3, 4, 5])

        print(ret1)
        print(ret2)
        for k in range(1, 6):
            self.assertEqual(ret1[k], ret2[k])  # add assertion here



if __name__ == '__main__':
    unittest.main()
