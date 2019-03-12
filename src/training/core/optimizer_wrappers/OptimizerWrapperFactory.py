from src.training.core.optimizer_wrappers.AdamWrapper import AdamWrapper


class OptimizerWrapperFactory:

    def assembly(self, config_dict):
        if config_dict['optimizer_name'].lower() == 'adam':
            return AdamWrapper(config_dict)
