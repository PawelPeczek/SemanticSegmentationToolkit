from typing import List

import cv2 as cv
import numpy as np

from src.dataset.analysis.primitives import PreprocessedGroundTruth, \
    GroundTruthClass, ContourDetail
from src.dataset.utils.mapping_utils import Color2IdMapping, Color


class GroundTruthPreprocessor:

    def __init__(self, color2id: Color2IdMapping):
        self.__color2id = color2id

    def preprocess(self, image: np.ndarray) -> PreprocessedGroundTruth:
        classes_info = {}
        for color in self.__color2id:
            color_id = self.__color2id[color]
            ground_truth_class_info = self.__extract_class_info(
                image=image,
                color=color
            )
            classes_info[color_id] = ground_truth_class_info
        return PreprocessedGroundTruth(
            image=image,
            classes=classes_info
        )

    def __extract_class_info(self,
                             image: np.ndarray,
                             color: Color) -> GroundTruthClass:
        mask_extractor = _ImageMaskExtractor(color_to_search=color)
        mask = mask_extractor.extract_mask(image=image)
        contours_extractor = _ContoursExtractor()
        contours = contours_extractor.extract_contours(mask=mask)
        return GroundTruthClass(segmentation_mask=mask, contours=contours)


class _ImageMaskExtractor:

    def __init__(self, color_to_search: Color):
        self.__color_to_search = np.array(color_to_search)

    def extract_mask(self, image: np.ndarray) -> np.ndarray:
        mask = image == self.__color_to_search
        mask = mask[:, :, 1]
        return np.array(mask, dtype=np.uint8)


class _ContoursExtractor:

    def extract_contours(self, mask: np.ndarray) -> List[ContourDetail]:
        _, contours, _ = cv.findContours(
            image=mask,
            mode=cv.RETR_TREE,
            method=cv.CHAIN_APPROX_SIMPLE
        )
        return list(map(lambda c: ContourDetail(c), contours))

