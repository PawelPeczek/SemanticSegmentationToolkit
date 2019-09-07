import math
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from typing import List, Tuple, Dict, Optional
import os
import re
import logging

import tensorflow as tf
import cv2 as cv
import numpy as np

from src.common.config_utils import DataPreProcessingConfigReader
from src.common.utils.progress import ProgressReporter
from src.dataset.utils.mapping_utils import Color2IdMapping, \
    get_color_to_id_mapping
from src.utils.filesystem_utils import create_directory, dump_content_to_csv


class DatasetPart(Enum):
    TRAINING_SET = 'train'
    TEST_SET = 'test'
    VALIDATION_SET = 'val'


class DatasetPreprocessor:

    def __init__(self, config: DataPreProcessingConfigReader):
        self.__config = config

    def transform_dataset(self) -> None:
        create_directory(self.__config.output_tfrecords_dir)
        self.__transform_dataset_part(DatasetPart.TRAINING_SET)
        self.__transform_dataset_part(DatasetPart.VALIDATION_SET)

    def __transform_dataset_part(self, dataset_part: DatasetPart) -> None:
        dataset_dir = self.__config.dataset_dir
        extractors = (_ExamplesExtractor, _GroundTruthsExtractor)
        extractors = tuple(
            map(lambda e: e(dataset_dir, dataset_part), extractors)
        )
        samples = self.__prepare_samples(*extractors)
        dataset_converter = _TFRecordsConverter(
            dataset_part=dataset_part,
            config=self.__config
        )
        dataset_converter.convert(samples=samples)
        self.__persist_samples_list(samples=samples, dataset_part=dataset_part)

    def __prepare_samples(self,
                          examples_extractor: '_ExamplesExtractor',
                          gt_extractor: '_GroundTruthsExtractor'
                          ) -> List['Sample']:
        examples = examples_extractor.extract_input()
        gt = gt_extractor.extract_input()
        consolidator = _SamplesConsolidator()
        return consolidator.connect_examples_with_ground_truths(
            examples=examples,
            ground_truths=gt
        )

    def __persist_samples_list(self,
                               dataset_part: DatasetPart,
                               samples: List['Sample']) -> None:
        samples = list(map(lambda s: s.to_compact_form(), samples))
        target_path = os.path.join(
            self.__config.dataset_dir,
            self.__config.output_tfrecords_dir,
            f'{dataset_part.name}_samples_mapping.csv'
        )
        dump_content_to_csv(
            file_path=target_path,
            content=samples
        )


class _DataSetExtractor(ABC):

    def __init__(self,
                 dataset_dir: str,
                 dataset_part: DatasetPart):
        self._dataset_part = dataset_part
        self.__content_dir = dataset_dir

    def extract_input(self) -> List[str]:
        result = []
        for directory, _, content in os.walk(self.__content_dir):
            prepare_path = partial(os.path.join, directory)
            content_paths = list(map(prepare_path, content))
            to_extract = list(filter(self._file_is_valid, content_paths))
            result += to_extract
        return result

    @abstractmethod
    def _file_is_valid(self, file_path: str) -> bool:
        raise RuntimeError('This method must be implemented by derived class.')


class _GroundTruthsExtractor(_DataSetExtractor):

    def _file_is_valid(self, file_path: str) -> bool:
        return (
                       file_path.endswith('_gtFine_color.png') or
                       file_path.endswith('_gtCoarse_color.png')
                ) and \
               self._dataset_part.value in file_path


class _ExamplesExtractor(_DataSetExtractor):

    def _file_is_valid(self, file_path: str) -> bool:
        return file_path.endswith('_leftImg8bit.png') and \
               self._dataset_part.value in file_path


class _NameRootExtractor:

    @staticmethod
    def extract_name_root(path: str) -> str:
        file_name = path.split('/')[-1]
        pattern = "^([a-zA-Z-]+_[0-9]{6}_[0-9]{6})_.*$"
        return re.match(pattern, file_name).group(1)


class Sample:

    def __init__(self,
                 sample_id: int,
                 example_path: str,
                 ground_truth_path: str):
        self.__sample_id = sample_id
        self.__example_path = example_path
        self.__ground_truth_path = ground_truth_path

    @property
    def id(self) -> int:
        return self.__sample_id

    @property
    def example(self) -> str:
        return self.__example_path

    @property
    def ground_truth(self) -> str:
        return self.__ground_truth_path

    def to_compact_form(self) -> list:
        return [
            self.__sample_id, self.__example_path, self.__ground_truth_path
        ]


class _SamplesConsolidator:

    def connect_examples_with_ground_truths(self,
                                            examples: List[str],
                                            ground_truths: List[str]
                                            ) -> List['Sample']:
        examples = dict(
            map(self.__connect_path_and_file_name_root, examples)
        )
        ground_truths = dict(
            map(self.__connect_path_and_file_name_root, ground_truths)
        )
        return self.__proceed_consolidation(
            examples=examples,
            ground_truths=ground_truths
        )

    def __connect_path_and_file_name_root(self, path: str) -> Tuple[str, str]:
        name_root = _NameRootExtractor.extract_name_root(path)
        return name_root, path

    def __proceed_consolidation(self,
                                examples: Dict[str, str],
                                ground_truths: Dict[str, str]
                                ) -> List['Sample']:
        samples = []
        for sample_id, example_root in enumerate(examples):
            if example_root not in ground_truths:
                logging.warning(f'Example {example_root} without ground truth.')
                continue
            example_path = examples[example_root]
            gt_path = ground_truths[example_root]
            sample = Sample(
                sample_id=sample_id,
                example_path=example_path,
                ground_truth_path=gt_path
            )
            samples.append(sample)
        return samples


class _TFRecordsConverter:

    def __init__(self,
                 dataset_part: DatasetPart,
                 config: DataPreProcessingConfigReader):
        self.__dataset_part = dataset_part
        self.__config = config

    def convert(self, samples: List[Sample]) -> None:
        logging.info(f'Start creating *.tfrecords - '
                     f'{self.__dataset_part.value} set')
        self.__prepare_storage()
        manager = self.__initialize_inter_process_communication_manager()
        progress_reporter = manager.ProgressReporter(
            elements_to_process=len(samples)
        )
        batches = self.__prepare_sample_batches(samples=samples)
        workers = self.__prepare_workers(
            batches=batches,
            progress_reporter=progress_reporter
        )
        execution_supervisor = _ExecutionSupervisor(
            max_parallel_workers=self.__config.max_workers
        )
        execution_supervisor.run_workers(workers)
        manager.shutdown()

    def __initialize_inter_process_communication_manager(self) -> BaseManager:
        BaseManager.register('ProgressReporter', ProgressReporter)
        manager = BaseManager()
        manager.start()
        return manager

    def __prepare_sample_batches(self,
                                 samples: List[Sample]) -> List[List[Sample]]:
        batch_splitter = _Splitter(self.__config.binary_batch_size)
        return  batch_splitter.split_into_batches(samples)

    def __prepare_storage(self) -> None:
        targer_dir_path = os.path.join(
            self.__config.output_tfrecords_dir,
            self.__dataset_part.value
        )
        create_directory(targer_dir_path)

    def __prepare_workers(self,
                          batches: List[List[Sample]],
                          progress_reporter: ProgressReporter
                          ) -> List['_TFRecordsConversionWorker']:
        color2id = get_color_to_id_mapping(self.__config.mapping_file)
        workers = []
        for batch_id, batch in enumerate(batches):
            target_path = self.__prepare_target_path(batch_id)
            worker = _TFRecordsConversionWorker(
                target_path=target_path,
                samples=batch,
                color2id=color2id,
                destination_size=self.__config.destination_size,
                progress_reporter=progress_reporter
            )
            workers.append(worker)
        return workers

    def __prepare_target_path(self, batch_id: int) -> str:
        tfrecords_file_name = f'{self.__config.tfrecords_base_name}-{batch_id}'
        return os.path.join(
            self.__config.output_tfrecords_dir,
            self.__dataset_part.value,
            tfrecords_file_name
        )


class _TFRecordsConversionWorker(Process):

    def __init__(self,
                 target_path: str,
                 samples: List[Sample],
                 color2id: Color2IdMapping,
                 destination_size: Tuple[int, int],
                 progress_reporter: Optional[ProgressReporter] = None):
        super().__init__()
        self.__target_path = target_path
        self.__samples = samples
        self.__color2id = color2id
        self.__destination_size = destination_size
        self.__progress_reporter = progress_reporter

    def run(self) -> None:
        with tf.python_io.TFRecordWriter(self.__target_path) as writer:
            self.__convert_samples(writer)

    def __convert_samples(self, writer: tf.python_io.TFRecordWriter) -> None:
        for sample in self.__samples:
            self.__convert_sample(sample=sample, writer=writer)

    def __convert_sample(self,
                         sample: Sample,
                         writer: tf.python_io.TFRecordWriter) -> None:
        example = self.__prepare_example(sample.example)
        gt = self.__prepare_gt(sample.ground_truth)
        data = {
            'id': self.__wrap_int64(sample.id),
            'example': self.__wrap_bytes(example),
            'gt': self.__wrap_bytes(gt)
        }
        feature = tf.train.Features(feature=data)
        example = tf.train.Example(features=feature)
        serialized = example.SerializeToString()
        writer.write(serialized)
        if self.__progress_reporter is not None:
            self.__progress_reporter.report_processed_element()

    def __prepare_example(self, example_path: str) -> bytes:
        example = self.__load_image(
            image_path=example_path,
            interpolation=cv.INTER_LINEAR
        )
        return example.tostring()

    def __prepare_gt(self, gt_path: str) -> bytes:
        gt = self.__load_image(
            image_path=gt_path,
            interpolation=cv.INTER_NEAREST
        )
        class_mapper = _GroundTruthClassMapper(color2id=self.__color2id)
        gt = class_mapper.map_classes_id(ground_truth=gt)
        gt = gt.astype(np.uint8)
        return gt.tostring()

    def __load_image(self,
                     image_path: str,
                     interpolation: int) -> np.ndarray:
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        target_width, target_height = self.__destination_size
        if image_height != target_height or image_width != target_width:
            image = cv.resize(
                image,
                self.__destination_size,
                interpolation=interpolation
            )
        return image

    def __wrap_bytes(self, value: bytes) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __wrap_int64(self, value: int) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class _GroundTruthClassMapper:

    def __init__(self,
                 color2id: Color2IdMapping):
        self.__color2id = color2id

    def map_classes_id(self, ground_truth: np.ndarray) -> np.ndarray:

        def class_mapper(colour: List[int]) -> int:
            colour = colour[0], colour[1], colour[2]
            if colour in self.__color2id:
                return self.__color2id[colour]
            else:
                return 0

        return np.apply_along_axis(class_mapper, 2, ground_truth)


class _Splitter:

    def __init__(self, batch_size: int):
        self.__batch_size = batch_size

    def split_into_batches(self, samples: list) -> List[list]:
        batches_raw_number = len(samples) / self.__batch_size
        full_batches_number = int(math.floor(batches_raw_number))
        not_full_batch_exists = len(samples) % self.__batch_size
        batches = []
        for i in range(full_batches_number):
            start_idx = i * self.__batch_size
            end_idx = (i + 1) * self.__batch_size
            batch = samples[start_idx:end_idx]
            batches.append(batch)
        if not_full_batch_exists:
            start_idx = full_batches_number * self.__batch_size
            batch = samples[start_idx:]
            batches.append(batch)
        return batches


class _ExecutionSupervisor:

    def __init__(self, max_parallel_workers: int):
        self.__max_parallel_workers = max_parallel_workers

    def run_workers(self, workers: List[_TFRecordsConversionWorker]) -> None:
        splitter = _Splitter(self.__max_parallel_workers)
        workers_batches = splitter.split_into_batches(workers)
        for workers_batch in workers_batches:
            self.__run_workers_batch(workers_batch)

    def __run_workers_batch(self,
                            workers_batch: List[_TFRecordsConversionWorker]
                            ) -> None:
        for worker in workers_batch:
            worker.start()
        for worker in workers_batch:
            worker.join()

