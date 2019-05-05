from typing import Dict

from src.model.SemanticSegmentationModel import SemanticSegmentationModel
from src.model.normal_input_model.ICNetV2 import ICNetV2
from src.model.normal_input_model.ICNetV3 import ICNetV3
from src.model.normal_input_model.ICNetV4 import ICNetV4
from src.model.normal_input_model.ICNetV5 import ICNetV5
from src.model.normal_input_model.ICNetV6 import ICNetV6
from src.model.small_input_model.ThickModel import ThickModel
from src.model.small_input_model.ThinModelV1 import ThinModelV1
from src.model.small_input_model.ThinModelV2 import ThinModelV2
from src.model.small_input_model.ThinModelV3 import ThinModelV3
from src.model.small_input_model.ThinModelV4 import ThinModelV4
from src.model.small_input_model.UltraSlimModel import UltraSlimModel


class SegmentationModelFactory:

    def __init__(self):
        self.__model_gallery = self.__prepare_model_gallery()

    def assembly(self, model_name: str) -> SemanticSegmentationModel:
        if model_name not in self.__model_gallery:
            raise RuntimeError('Requested model not exists or is not registered.')
        return self.__model_gallery[model_name]

    def __prepare_model_gallery(self) -> Dict[str, SemanticSegmentationModel]:
        gallery = {
            'ThickModel': ThickModel(),
            'ThinModelV1': ThinModelV1(),
            'ThinModelV2': ThinModelV2(),
            'ThinModelV3': ThinModelV3(),
            'ThinModelV4': ThinModelV4(),
            'UltraSlimModel': UltraSlimModel(),
            'ICNetV2': ICNetV2(),
            'ICNetV3': ICNetV3(),
            'ICNetV4': ICNetV4(),
            'ICNetV5': ICNetV5(),
            'ICNetV6': ICNetV6()
        }
        return gallery

