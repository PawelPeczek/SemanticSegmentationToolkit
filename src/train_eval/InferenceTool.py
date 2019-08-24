from typing import Union

from fire import Fire

from src.common.config_utils import InferenceConfigReader
from src.train_eval.core.inference_utils.StreamInferenceUtil import StreamInferenceUtil


class InferenceTool:

    def infer_on_video_stream(self, config_path: Union[str, None] = None) -> None:
        try:
            config = InferenceConfigReader(config_path)
            inference_util = StreamInferenceUtil(config)
            inference_util.infer_on_video_stream()
        except Exception as ex:
            print('Failed to proceed model evaluation. {}'.format(ex))


if __name__ == '__main__':
    Fire(InferenceTool)
