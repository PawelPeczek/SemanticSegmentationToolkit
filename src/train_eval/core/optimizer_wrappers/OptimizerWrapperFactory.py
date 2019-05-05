from typing import Dict

from src.train_eval.core.optimizer_wrappers.AdamWWrapper import AdamWWrapper
from src.train_eval.core.optimizer_wrappers.AdamWrapper import AdamWrapper
from src.train_eval.core.optimizer_wrappers.OptimizerWrapper import OptimizerWrapper


class OptimizerWrapperFactory:

    def assembly(self, config_dict: Dict) -> OptimizerWrapper:
        if config_dict['optimizer_name'].lower() == 'adam':
            return AdamWrapper(config_dict)
        else:
            return AdamWWrapper(config_dict)
