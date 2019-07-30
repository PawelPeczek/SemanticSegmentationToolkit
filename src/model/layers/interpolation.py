from typing import Union

import tensorflow as tf


def upsample_bilinear(x: tf.Tensor,
                      zoom_factor: int) -> tf.Tensor:
    return _interpolate(
        x=x,
        resize_factor=zoom_factor,
        resize_method=tf.image.ResizeMethod.BILINEAR)


def upsample_nn(x: tf.Tensor,
                zoom_factor: int) -> tf.Tensor:
    return _interpolate(
        x=x,
        resize_factor=zoom_factor,
        resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def downsample_bilinear(x: tf.Tensor,
                        shrink_factor: int) -> tf.Tensor:
    resize_factor = 1 / shrink_factor
    return _interpolate(
        x=x,
        resize_factor=resize_factor,
        resize_method=tf.image.ResizeMethod.BILINEAR)


def downsample_nn(x: tf.Tensor,
                  shrink_factor: int) -> tf.Tensor:
    resize_factor = 1 / shrink_factor
    return _interpolate(
        x=x,
        resize_factor=resize_factor,
        resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def _interpolate(x: tf.Tensor,
                 resize_factor: float,
                 resize_method: tf.image.ResizeMethod) -> tf.Tensor:
    if resize_factor >= 1:
        resize_factor = int(round(resize_factor))
        new_shape = x.shape[1] * resize_factor, x.shape[2] * resize_factor
    else:
        resize_factor = 1 / resize_factor
        resize_factor = int(round(resize_factor))
        new_shape = x.shape[1] // resize_factor, x.shape[2] // resize_factor
    return tf.image.resize_images(
        images=x,
        size=new_shape,
        method=resize_method,
        align_corners=True)


def resize_bilinear(x: tf.Tensor,
                    height: Union[int, tf.Dimension],
                    width: Union[int, tf.Dimension]) -> tf.Tensor:
    return _resize(
        x=x,
        height=height,
        width=width,
        resize_method=tf.image.ResizeMethod.BILINEAR)


def resize_nn(x: tf.Tensor,
              height: Union[int, tf.Dimension],
              width: Union[int, tf.Dimension]) -> tf.Tensor:
    return _resize(
        x=x,
        height=height,
        width=width,
        resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def _resize(x: tf.Tensor,
            height: Union[int, tf.Dimension],
            width: Union[int, tf.Dimension],
            resize_method: tf.image.ResizeMethod) -> tf.Tensor:
    return tf.image.resize_images(
        images=x,
        size=(height, width),
        method=resize_method,
        align_corners=True)
