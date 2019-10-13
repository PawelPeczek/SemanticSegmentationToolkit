from enum import Enum


class TransformationType(Enum):

    CROP_AND_SCALE = 'crop_and_scale'
    ROTATION = 'rotation'
    HORIZONTAL_FLIP = 'horizontal_flip'
    GAUSSIAN_NOISE = 'gaussian_noise'
    ADJUST_BRIGHTNESS = 'adjust_brightness'
    ADJUST_CONTRAST = 'adjust_contrast'


