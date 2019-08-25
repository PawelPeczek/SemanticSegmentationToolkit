import os
from functools import cmp_to_key
from typing import Tuple, List
from glob import glob

import cv2 as cv
from tqdm import tqdm

from src.utils.filesystem_utils import create_directory


class VideoPreprocessor:

    def __init__(self,
                 frames_directory_path: str,
                 output_path: str,
                 destination_size: Tuple[int, int] = (2048, 1024)):
        self.__frames_directory_path = frames_directory_path
        self.__output_path = output_path
        self.__destination_size = destination_size

    def produce_output_video(self) -> None:
        self.__prepare_storage()
        frames_collector = _PNGFramesCollector(
            frames_directory_path=self.__frames_directory_path
        )
        frames_images_paths = frames_collector.get_frames()
        video_encoder = _VideoEncoder(
            output_path=self.__output_path,
            target_size=self.__destination_size
        )
        video_encoder.encode_frames(frames_paths=frames_images_paths)

    def __prepare_storage(self) -> None:
        output_directory_path = os.path.dirname(self.__output_path)
        create_directory(output_directory_path)


class _PNGFramesCollector:

    def __init__(self, frames_directory_path: str):
        self.__frames_directory_path = frames_directory_path

    def get_frames(self) -> List[str]:
        frame_paths = glob(os.path.join(self.__frames_directory_path, '*.png'))
        frame_paths = sorted(frame_paths,
                             key=cmp_to_key(self.__frame_path_comparator))
        return frame_paths

    def __frame_path_comparator(self, path_one: str, path_two: str) -> int:
        first_frame_number = self.__extract_frame_number(path_one)
        second_frame_number = self.__extract_frame_number(path_two)
        return first_frame_number - second_frame_number

    def __extract_frame_number(self, path: str) -> int:
        frame_number = path.split('_')[-2]
        return int(frame_number)


class _VideoEncoder:

    def __init__(self,
                 output_path: str,
                 target_size: Tuple[int, int]):
        self.__output_path = output_path
        self.__target_size = target_size

    def encode_frames(self, frames_paths: List[str]) -> None:
        writer = cv.VideoWriter(
            self.__output_path,
            cv.VideoWriter_fourcc(*'DIVX'),
            25,
            self.__target_size
        )
        for frame_path in tqdm(frames_paths):
            self.__encode_frame(frame_path=frame_path, writer=writer)

    def __encode_frame(self,
                       frame_path: str,
                       writer: cv.VideoWriter) -> None:
        frame = cv.imread(frame_path)
        height, width = frame.shape[:2]
        if (width, height) != self.__target_size:
            frame = cv.resize(
                frame,
                self.__target_size,
                interpolation=cv.INTER_LINEAR
            )
        writer.write(frame)
