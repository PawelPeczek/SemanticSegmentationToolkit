ATROUS_ENCODER_PYRAMID_CFG_ERROR_MSG = \
    'Lists of pyramid_heads_dilation_rate and pyramid_heads_kernels must be ' \
    'equally long.'
ATROUS_ENCODER_RESIDUAL_ERROR_MSG = \
    'Input (or reducted input) do not fit the pyramid output in terms of ' \
    'filters number.'


class AtrousPyramidEncoderError(Exception):
    pass
