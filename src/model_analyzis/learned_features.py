import math
from typing import List
import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.dataset.analysis.core import ListSplitter
from src.utils.filesystem_utils import create_directory


def get_all_graph_nodes_names(graph: tf.Graph) -> List[str]:
    return [n.name for n in graph.as_graph_def().node]


def visualize_features(session: tf.Session,
                       graph: tf.Graph,
                       nodes: List[str],
                       input_images: List[str],
                       x_placeholder: tf.Tensor,
                       target_dir: str
                       ) -> None:
    tensors_to_evaluate = _get_tensors_by_name(graph=graph, names=nodes)
    create_directory(target_dir)
    images = list(map(cv.imread, input_images))
    images = np.stack(images, axis=0)
    inference_results = session.run(
        tensors_to_evaluate,
        feed_dict={x_placeholder: images}
    )
    _persist_results(
        images=images,
        inference_results=inference_results,
        nodes=nodes,
        target_dir=target_dir
    )


def _get_tensors_by_name(graph: tf.Graph, names: List[str]) -> List[tf.Tensor]:
    return [
        graph.get_tensor_by_name(f'{name}:0') for name in names
    ]


def _persist_results(images: np.ndarray,
                     inference_results: List[np.ndarray],
                     nodes: List[str],
                     target_dir: str
                     ) -> None:
    for node_name, node_output in tqdm(zip(nodes, inference_results)):
        _persist_single_node_output(
            node_name=node_name,
            node_output=node_output,
            images=images,
            target_dir=target_dir
        )


def _persist_single_node_output(node_name: str,
                                node_output: np.ndarray,
                                images: np.ndarray,
                                target_dir: str):
    node_name = node_name.replace('/', '_')
    target_dir = os.path.join(target_dir, node_name)
    create_directory(target_dir)
    for channel_idx in range(node_output.shape[-1]):
        channel_output = node_output[..., channel_idx]
        target_file_name = os.path.join(target_dir, f'{channel_idx}.jpeg')
        _persist_single_output_slice(
            images=images,
            channel_output=channel_output,
            target_file_name=target_file_name
        )


def _persist_single_output_slice(images: np.ndarray,
                                 channel_output: np.ndarray,
                                 target_file_name: str
                                 ) -> None:
    channel_output = _standardize_channel_output(channel_output)
    overlayed_images = _overlay_results(images, channel_output)
    target_grid = _prepare_target_grid(images=overlayed_images)
    cv.imwrite(target_file_name, target_grid)


def _standardize_channel_output(channel_output: np.ndarray) -> np.ndarray:
    result = []
    for element_idx in range(channel_output.shape[0]):
        standardized_element = _standardize_output_element(
            channel_output[element_idx]
        )
        result.append(standardized_element)
    return np.stack(result, axis=0)


def _standardize_output_element(output_element: np.ndarray) -> np.ndarray:
    min_val, max_val = np.min(output_element), np.max(output_element)
    val_span = max_val - min_val
    if min_val < 0:
        output_element = output_element + min_val
    else:
        output_element = output_element - min_val
    output_element = (output_element / val_span) * 255
    output_element = output_element.astype(np.uint8)
    padding_shape = (output_element.shape[0], output_element.shape[1], 2)
    output_element = np.expand_dims(output_element, axis=-1)
    padding = np.zeros(padding_shape, dtype=np.uint8)
    return np.concatenate([padding, output_element], axis=-1)


def _overlay_results(images: np.ndarray,
                     channel_output: np.ndarray
                     ) -> List[np.ndarray]:
    results = []
    for element_idx in range(images.shape[0]):
        image = images[element_idx]
        output = channel_output[element_idx]
        output = cv.resize(output, image.shape[:2][::-1])
        result = cv.addWeighted(image, 0.35, output, 0.65, 0.0)
        results.append(result)
    return results


def _prepare_target_grid(images: List[np.ndarray]):
    grid_size = math.ceil(math.sqrt(len(images)))
    images_splitted = ListSplitter.split_list(
        to_split=images,
        chunks_number=grid_size
    )
    if len(images_splitted[-1]) < grid_size:
        padding_len = grid_size - len(images_splitted[-1])
        padding = [np.zeros_like(images[0]) for _ in range(padding_len)]
        images_splitted[-1] += padding
    result = []
    for grid_row in images_splitted:
        row = np.concatenate(grid_row, axis=1)
        result.append(row)
    return np.concatenate(result, axis=0)
