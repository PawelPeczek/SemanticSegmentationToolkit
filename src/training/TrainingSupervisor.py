from fire import Fire

from src.training.core.Trainer import Trainer
from src.training.core.TrainingConfigReader import TrainingConfigReader


class TrainingSupervisor:

    def full_training(self, descriptive_name, config_path=None):
        try:
            config = TrainingConfigReader(config_path)
            trainer = Trainer(descriptive_name, config)
            trainer.train()
        except Exception as ex:
            print('Failed to convert dataset. {}'.format(ex))


if __name__ == '__main__':
    Fire(TrainingSupervisor)
