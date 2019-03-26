import tensorflow as tf
from typing import Dict

from src.train_eval.core.optimizer_wrappers.OptimizerWrapper import OptimizerWrapper


class AdamWrapper(OptimizerWrapper):

    def __init__(self, config_dict: Dict):
        super().__init__(config_dict)

    def get_optimizer(self) -> tf.train.Optimizer:
        learning_rate = self._config_dict['learning_rate']
        beta1 = self._get_parameter_value('beta1', 0.9)
        beta2 = self._get_parameter_value('beta1', 0.999)
        epsilon = self._get_parameter_value('epsilon', 1e-08)
        use_locking = self._get_parameter_value('use_locking', False)
        name = self._get_parameter_value('name', 'Adam')
        return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=beta1,
                                      beta2=beta2,
                                      epsilon=epsilon,
                                      use_locking=use_locking,
                                      name=name)

