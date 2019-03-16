from fire import Fire

from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.train_eval.core.GraphExecutor import GraphExecutor
from src.train_eval.core.TrainValConfigReader import TrainValConfigReader


class TrainingSupervisor:

    def full_training(self, descriptive_name, config_path=None):
        try:
            config = TrainValConfigReader(config_path)
            executor = GraphExecutor(descriptive_name, config, IteratorType.TRAINING_ITERATOR)
            executor.train()
        except Exception as ex:
            print('Failed to proceed full training. {}'.format(ex))

    def overfit_training(self, descriptive_name, config_path=None):
        try:
            config = TrainValConfigReader(config_path)
            executor = GraphExecutor(descriptive_name, config, IteratorType.DUMMY_ITERATOR)
            executor.train()
        except Exception as ex:
            print('Failed to proceed overfitting training. {}'.format(ex))


if __name__ == '__main__':
    Fire(TrainingSupervisor)
