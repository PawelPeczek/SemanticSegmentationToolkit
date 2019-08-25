import os
import sys
from functools import cmp_to_key
from typing import Tuple, List
from glob import glob
import cv2 as cv

from src.utils.filesystem_utils import create_directory


class VideoPreprocessor:

    def __init__(self, frames_directory_path: str, output_path: str, destination_size: Tuple[int, int] = (2048, 1024)):
        self.__frames_directory_path = frames_directory_path
        self.__output_path = output_path
        self.__destination_size = destination_size

    def produce_output_video(self) -> None:
        self.__prepare_storage()
        frames_images_paths = self.__get_frames()
        self.__change_frames_into_video(frames_images_paths)

    def __get_frames(self) -> List[str]:
        frame_paths = glob(os.path.join(self.__frames_directory_path, '*.png'))
        frame_paths = sorted(frame_paths, key=cmp_to_key(self.__frame_path_comparator))
        return frame_paths

    def __frame_path_comparator(self, path_one: str, path_two: str) -> int:
        first_frame_number = self.__extract_frame_number(path_one)
        second_frame_number = self.__extract_frame_number(path_two)
        return first_frame_number - second_frame_number

    def __extract_frame_number(self, path: str) -> int:
        frame_number = path.split('_')[-2]
        return int(frame_number)

    def __change_frames_into_video(self, frames_images_paths: List[str]) -> None:
        video = cv.VideoWriter(self.__output_path, cv.VideoWriter_fourcc(*'DIVX'), 25, self.__destination_size)
        processed_frames = 0
        all_frames_number = len(frames_images_paths)
        for frame_path in frames_images_paths:
            processed_frames += 1
            frame = cv.imread(frame_path)
            height, width = frame.shape[:2]
            if (width, height) != self.__destination_size:
                frame = cv.resize(frame, self.__destination_size, interpolation=cv.INTER_CUBIC)
            video.write(frame)
            self.__print_progress(processed_frames, all_frames_number)
        video.release()

    def __print_progress(self, processed_elements: int, total_elements_num: int):
        porgress = processed_elements / total_elements_num * 100
        sys.stdout.write("\rProgress: {}\t%\t".format(porgress))
        sys.stdout.flush()

    def __prepare_storage(self) -> None:
        output_directory_path = os.path.dirname(self.__output_path)
        create_directory(output_directory_path)

