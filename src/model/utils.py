from typing import Dict, Optional, List

from src.common.models_register import NAME_TO_MODEL
from src.model.network import Network


class ModelFactory:

    __MODEL_NOT_PRESENT_ERROR_MSG = 'Requested model not exists ' \
                                    'or is not registered.'

    @staticmethod
    def assembly(model_name: str,
                 output_classes: int,
                 image_mean: Optional[List[float]] = None,
                 ignore_labels: Optional[List[int]] = None,
                 model_config: Optional[dict] = None) -> Network:
        if model_name not in NAME_TO_MODEL:
            raise RuntimeError(
                ModelFactory.__MODEL_NOT_PRESENT_ERROR_MSG)
        return NAME_TO_MODEL[model_name](
            output_classes=output_classes,
            image_mean=image_mean,
            ignore_labels=ignore_labels,
            config=model_config)

