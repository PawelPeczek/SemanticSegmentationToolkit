from abc import ABC, abstractmethod
from typing import Dict, Any
import tensorflow as tf


class OptimizerWrapper(ABC):

    def __init__(self, config_dict: Dict):
        self._config_dict = config_dict

    @abstractmethod
    def get_optimizer(self) -> tf.train.Optimizer:
        raise NotImplementedError('This method must be implemented.')

    def _get_parameter_value(self,
                             parameter_name: str,
                             default_value: Any = None) -> Any:
        if parameter_name in self._config_dict:
            return self._config_dict[parameter_name]
        else:
            return default_value


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


class AdamWWrapper(OptimizerWrapper):

    def __init__(self, config_dict: Dict):
        super().__init__(config_dict)

    def get_optimizer(self) -> tf.train.Optimizer:
        weight_decay = self._config_dict['weight_decay']
        learning_rate = self._config_dict['learning_rate']
        beta1 = self._get_parameter_value('beta1', 0.9)
        beta2 = self._get_parameter_value('beta1', 0.999)
        epsilon = self._get_parameter_value('epsilon', 1e-08)
        use_locking = self._get_parameter_value('use_locking', False)
        name = self._get_parameter_value('name', 'AdamW')
        return tf.contrib.opt.AdamWOptimizer(weight_decay,
                                             learning_rate=learning_rate,
                                             beta1=beta1,
                                             beta2=beta2,
                                             epsilon=epsilon,
                                             use_locking=use_locking,
                                             name=name)


NAME_TO_OPTIMIZER = {
    'adam': AdamWrapper,
    'adamw': AdamWWrapper
}


class OptimizerWrapperError(Exception):
    pass


class OptimizerWrapperFactory:

    __NON_EXISTING_WRAPPER_ERROR_MSG = 'Optimizer wrapper with given name ' \
                                       'not exists.'

    def assembly(self, config_dict: Dict) -> OptimizerWrapper:
        optimizer_name = config_dict['optimizer_name'].lower()
        if optimizer_name not in NAME_TO_OPTIMIZER:
            raise OptimizerWrapperError(
                OptimizerWrapperFactory.__NON_EXISTING_WRAPPER_ERROR_MSG)
        return NAME_TO_OPTIMIZER[optimizer_name](config_dict)

