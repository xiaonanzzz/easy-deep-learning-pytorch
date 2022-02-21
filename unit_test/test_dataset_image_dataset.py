import unittest

class MyTestCase(unittest.TestCase):

    def test_cub_dataset(self):
        from easydl.datasets.cub import Cub2011MetricLearningDS, CubMetricLearningExperiment
        from easydl.config import get_config_from_cmd
        data_path = get_config_from_cmd('data_path', '~/data/CUB_200_2011')

        trainds = Cub2011MetricLearningDS(data_path, item_schema=('image', 'label_code', 'label', 'name'))


    def test_cub_classification(self):
        from easydl.datasets.cub import CubClassificationDS
        from easydl.config import get_config_from_cmd
        data_path = get_config_from_cmd('data_path', '~/data/CUB_200_2011')

        trainds = CubClassificationDS(data_path, split='train', item_schema=('image', 'label_code', 'label', 'name'))
        print(trainds[0])
        print(len(trainds.dataset.label_map))

        testds = CubClassificationDS(data_path, split='test', item_schema=('image', 'label_code', 'label', 'name'))

        testds.show_profile()


if __name__ == '__main__':
    unittest.main()
