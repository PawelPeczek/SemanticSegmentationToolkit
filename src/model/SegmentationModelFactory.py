from typing import Dict

from src.model.SemanticSegmentationModel import SemanticSegmentationModel
from src.model.normal_input_model.ICNetV10 import ICNetV10
from src.model.normal_input_model.ICNetV11 import ICNetV11
from src.model.normal_input_model.ICNetV12 import ICNetV12
from src.model.normal_input_model.ICNetV13 import ICNetV13
from src.model.normal_input_model.ICNetV14 import ICNetV14
from src.model.normal_input_model.ICNetV15 import ICNetV15
from src.model.normal_input_model.ICNetV2 import ICNetV2
from src.model.normal_input_model.ICNetV3 import ICNetV3
from src.model.normal_input_model.ICNetV4 import ICNetV4
from src.model.normal_input_model.ICNetV5 import ICNetV5
from src.model.normal_input_model.ICNetV6 import ICNetV6
from src.model.normal_input_model.ICNetV7 import ICNetV7
from src.model.normal_input_model.ICNetV8 import ICNetV8
from src.model.normal_input_model.ICNetV9 import ICNetV9
from src.model.normal_input_model.MPPNet import MPPNet
from src.model.normal_input_model.OriginalICNet import OriginalICNet
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
            'ICNetV6': ICNetV6(),
            'ICNetV7': ICNetV7(),
            'ICNetV8': ICNetV8(),
            'OriginalICNet': OriginalICNet(),
            'ICNetV9': ICNetV9(),
            'ICNetV10': ICNetV10(),
            'MPPNet': MPPNet(),
            'ICNetV11': ICNetV11(),
            'ICNetV12': ICNetV12(),
            'ICNetV13': ICNetV13(),
            'ICNetV14': ICNetV14(),
            'ICNetV15': ICNetV15()
        }
        return gallery

