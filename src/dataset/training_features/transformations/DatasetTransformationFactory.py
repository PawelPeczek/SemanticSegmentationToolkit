from src.dataset.training_features.transformations.AdjustBrightness import AdjustBrightness
from src.dataset.training_features.transformations.AdjustContrast import AdjustContrast
from src.dataset.training_features.transformations.CropAndScale import CropAndScale
from src.dataset.training_features.transformations.DatasetTransformation import DatasetTransformation
from src.dataset.training_features.transformations.GaussianNoiseAdder import GaussianNoiseAdder
from src.dataset.training_features.transformations.HorizontalFlip import HorizontalFlip
from src.dataset.training_features.transformations.Rotation import Rotation
from src.dataset.training_features.transformations.TransformationType import TransformationType


class DatasetTransformationFactory:

    def assembly_transformation(self, transformation_type: TransformationType) -> DatasetTransformation:
        if transformation_type is TransformationType.CROP_AND_SCALE:
            return CropAndScale(transformation_type)
        elif transformation_type is TransformationType.ROTATION:
            return Rotation(transformation_type)
        elif transformation_type is TransformationType.HORIZONTAL_FLIP:
            return HorizontalFlip(transformation_type)
        elif transformation_type is TransformationType.GAUSSIAN_NOISE:
            return GaussianNoiseAdder(transformation_type)
        elif transformation_type is TransformationType.ADJUST_BRIGHTNESS:
            return AdjustBrightness(transformation_type)
        elif transformation_type is TransformationType.ADJUST_CONTRAST:
            return AdjustContrast(transformation_type)
