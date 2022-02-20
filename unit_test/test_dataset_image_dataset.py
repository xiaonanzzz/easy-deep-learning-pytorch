import unittest

class MyTestCase(unittest.TestCase):

    def test_cub_dataset(self):
        from easydl.datasets.cub import Cub2011MetricLearningDS, CubMetricLearningExperiment
        from easydl.config import get_config_from_cmd
        data_path = get_config_from_cmd('data_path', '~/data/CUB_200_2011')

        trainds = Cub2011MetricLearningDS(data_path, item_schema=('image', 'label_code', 'label', 'name'))

        for x in range(len(trainds)):
            print(trainds[x][1:])



if __name__ == '__main__':
    unittest.main()
