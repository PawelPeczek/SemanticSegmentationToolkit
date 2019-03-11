from src.model.small_input_model.ThickModel import ThickModel
from src.model.small_input_model.ThinModelV1 import ThinModelV1
from src.model.small_input_model.ThinModelV2 import ThinModelV2
from src.model.small_input_model.UltraSlimModel import UltraSlimModel


class SegmentationModelBuilder:

    def __init__(self):
        self.__model_gallery = self.__prepare_model_gallery()

    def build(self, model_name):
        if model_name not in self.__model_gallery:
            raise RuntimeError('Requested model not exists or is not registered.')
        return self.__model_gallery[model_name]

    def __prepare_model_gallery(self):
        gallery = {
            'ThickModel': ThickModel(),
            'ThinModelV1': ThinModelV1(),
            'ThinModelV2': ThinModelV2(),
            'UltraSlimModel': UltraSlimModel()
        }
        return gallery

