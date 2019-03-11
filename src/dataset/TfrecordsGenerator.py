from fire import Fire

from src.dataset.core.ConfigReader import ConfigReader
from src.dataset.core.DatasetPreprocessor import DatasetPreprocessor


class TfrecordsGenerator:

    def generate(self, config_path=None):
        try:
            config = ConfigReader(config_path)
            dataset_preprocessor = DatasetPreprocessor(config)
            dataset_preprocessor.transform_dataset()
        except Exception as ex:
            print('Failed to convert dataset. {}'.format(ex))


if __name__ == '__main__':
    Fire(TfrecordsGenerator)
