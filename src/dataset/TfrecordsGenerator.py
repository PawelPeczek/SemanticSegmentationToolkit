from fire import Fire
from typing import Union
from src.dataset.preprocessing.DataPreProcessingConfigReader import DataPreProcessingConfigReader
from src.dataset.preprocessing.DatasetPreprocessor import DatasetPreprocessor


class TfrecordsGenerator:

    def generate(self, config_path: Union[None, str] = None) -> None:
        try:
            config = DataPreProcessingConfigReader(config_path)
            dataset_preprocessor = DatasetPreprocessor(config)
            dataset_preprocessor.transform_dataset()
        except Exception as ex:
            print('Failed to convert images. {}'.format(ex))


if __name__ == '__main__':
    Fire(TfrecordsGenerator)
