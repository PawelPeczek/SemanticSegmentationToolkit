import math
from functools import partial, reduce
from multiprocessing import Process, Queue
from multiprocessing.pool import Pool
from typing import List, Optional
import logging
from datetime import datetime
import os

import numpy as np
import cv2 as cv
from tqdm import tqdm

from src.common.config_utils import DataAnalysisConfigReader
from src.dataset.analysis.analyzers import AnalysisCouple, GroundTruthAnalyzer, \
    GroundTruthAnalysisConsolidator
from src.dataset.analysis.analyzers_register import ANALYZERS, CONSOLIDATORS
from src.dataset.analysis.primitives import AnalysisResult, \
    PreprocessedGroundTruth
from src.dataset.analysis.utils import GroundTruthPreprocessor
from src.dataset.utils.mapping_utils import get_color_to_id_mapping
from src.utils.filesystem_utils import read_text_file_lines, dump_text_file


class AnalysisManager:

    def __init__(self, config: DataAnalysisConfigReader):
        self.__config = config
        self.__chain_assembler = _ProcessingChainAssembler(
            config=config.analysis_config
        )
        self.__analysis_processor = _AnalysisProcessor(
            mapping_path=config.mapping_path,
            mapping_workers_number=config.workers
        )
        self.__saver = _ResultsSaver()

    def run_analysis(self) -> None:
        files_to_process = read_text_file_lines(self.__config.images_path)
        images = self.__prepare_images(files_to_process=files_to_process)
        analysis_couples = self.__chain_assembler.assembly_processing_chain()
        analysis_results = self.__analysis_processor.proceed_analysis(
            images=images,
            analysis_couples=analysis_couples
        )
        self.__saver.persist_results(
            target_dir=self.__config.results_dir,
            analysis_results=analysis_results,
            analyzed_files=files_to_process
        )

    def __prepare_images(self,
                         files_to_process: List[str]
                         ) -> List[np.ndarray]:
        logging.info('Preparing images.')
        return list(map(self.__prepare_image, tqdm(files_to_process)))

    def __prepare_image(self, file_path: str) -> np.ndarray:
        image = cv.imread(file_path)
        target_width, target_height = self.__config.target_size
        if image.shape[0] != target_height or image.shape[1] != target_width:
            image = cv.resize(image, (target_width, target_height))
        return image


class _ProcessingChainAssembler:

    def __init__(self, config: dict):
        self.__config = config

    def assembly_processing_chain(self) -> List[AnalysisCouple]:
        return list(map(
            self.__assembly_analysis_couple,
            self.__config['to_be_used']
        ))

    def __assembly_analysis_couple(self, name: str) -> AnalysisCouple:
        self.__validate_name(name=name)
        kwargs = self.__get_configurable_kwargs(name=name)
        analyzer_cls, consolidator_cls = ANALYZERS[name], CONSOLIDATORS[name]
        analyzer = self.__instantiate_analyzer(
            analyzer_cls=analyzer_cls,
            kwargs=kwargs
        )
        consolidator = self.__instantiate_consolidator(
            analyzer_cls=consolidator_cls,
            kwargs=kwargs
        )
        return AnalysisCouple(
            analyzer=analyzer,
            consolidator=consolidator
        )

    def __validate_name(self, name: str) -> None:
        if name not in ANALYZERS or name not in CONSOLIDATORS:
            raise RuntimeError('Analyzer name not registered.')

    def __get_configurable_kwargs(self, name: str) -> dict:
        if name in self.__config['configurable']:
            kwargs = self.__config['configurable'][name]
        else:
            kwargs = None
        return kwargs

    def __instantiate_analyzer(self,
                               analyzer_cls: type,
                               kwargs: Optional[dict]) -> GroundTruthAnalyzer:
        kwargs = self.__extract_dict_key(
            dictionary=kwargs,
            key='analyzer'
        )
        return self.__instantiate(cls=analyzer_cls, kwargs=kwargs)

    def __instantiate_consolidator(self,
                                   analyzer_cls: type,
                                   kwargs: Optional[dict]
                                   ) -> GroundTruthAnalysisConsolidator:
        kwargs = self.__extract_dict_key(
            dictionary=kwargs,
            key='consolidator'
        )
        return self.__instantiate(cls=analyzer_cls, kwargs=kwargs)

    def __extract_dict_key(self, dictionary: dict, key: str) -> Optional[dict]:
        if dictionary is not None and key in dictionary:
            result = dictionary[key]
        else:
            result = None
        return result

    def __instantiate(self, cls: type, kwargs: Optional[dict]) -> object:
        if kwargs is not None:
            return cls(**kwargs)
        else:
            return cls()


class _AnalysisProcessor:

    def __init__(self,
                 mapping_path: str,
                 mapping_workers_number: int
                 ):
        mapping = get_color_to_id_mapping(mapping_path=mapping_path)
        self.__preprocessor = GroundTruthPreprocessor(
            color2id=mapping
        )
        self.__mapping_workers_number = mapping_workers_number

    def proceed_analysis(self,
                         images: List[np.ndarray],
                         analysis_couples: List[AnalysisCouple]
                         ) -> List[AnalysisResult]:
        analyzed_images = self.__run_analysis(
            images=images,
            analysis_couples=analysis_couples
        )
        return self.__summarize_results(
            analysis_couples=analysis_couples,
            analyzed_images=analyzed_images
        )

    def __run_analysis(self,
                       images: List[np.ndarray],
                       analysis_couples: List[AnalysisCouple]
                       ) -> List[List[AnalysisResult]]:
        logging.info('Analysis in progress...')
        images_for_workers = ListSplitter.split_list(
            to_split=images,
            chunks_number=self.__mapping_workers_number
        )
        result_queue = Queue()
        workers = self.__initialize_workers(
            images_for_workers=images_for_workers,
            analysis_couples=analysis_couples,
            result_queue=result_queue
        )
        analysis_results = self.__fetch_results(
            queue=result_queue,
            expected_elements_number=len(images)
        )
        self.__stop_workers(workers)
        return analysis_results

    def __initialize_workers(self,
                             images_for_workers: List[List[np.ndarray]],
                             analysis_couples: List[AnalysisCouple],
                             result_queue: Queue
                             ) -> List[Process]:
        worker_init = partial(
            _AnalysisWorker,
            analysis_couples=analysis_couples,
            preprocessor=self.__preprocessor,
            result_queue=result_queue
        )
        workers = list(map(worker_init, images_for_workers))
        for worker in workers:
            worker.start()
        return workers

    def __fetch_results(self,
                        queue: Queue,
                        expected_elements_number: int,
                        ) -> List[AnalysisResult]:
        results = []
        for _ in tqdm(range(expected_elements_number)):
            result = queue.get()
            results.append(result)
        return results

    def __stop_workers(self, workers: List[Process]) -> None:
        for worker in workers:
            worker.join()

    def __summarize_results(self,
                            analysis_couples: List[AnalysisCouple],
                            analyzed_images: List[List[AnalysisResult]]
                            ) -> List[AnalysisResult]:
        logging.info('Consolidation in progress...')
        consolidators = list(map(lambda c: c.consolidator, analysis_couples))
        analysis_reductor = partial(
            self.__analysis_reductor,
            consolidators=consolidators
        )
        return reduce(analysis_reductor, tqdm(analyzed_images))

    def __analysis_reductor(self,
                            acc: List[AnalysisResult],
                            next_elements: List[AnalysisResult],
                            consolidators: List[GroundTruthAnalysisConsolidator]
                            ) -> List[AnalysisResult]:
        results = []
        iterator = zip(consolidators, acc, next_elements)
        for consolidator, already_consolidated, to_consolidate in iterator:
            consolidated = consolidator.consolidate(
                already_consolidated=already_consolidated,
                to_consolidate=to_consolidate
            )
            results.append(consolidated)
        return results


class ListSplitter:

    @staticmethod
    def split_list(to_split: list, chunks_number: int) -> List[list]:
        chunk_size = int(math.ceil(len(to_split) / chunks_number))
        result = []
        for i in range(chunks_number):
            start_idx, end_idx = i * chunk_size, (i + 1) * chunk_size
            chunk = to_split[start_idx: end_idx]
            result.append(chunk)
        return result


class _AnalysisWorker(Process):

    def __init__(self,
                 images: List[np.ndarray],
                 analysis_couples: List[AnalysisCouple],
                 preprocessor: GroundTruthPreprocessor,
                 result_queue: Queue):
        super().__init__()
        self.__images = images
        self.__analysis_couples = analysis_couples
        self.__preprocessor = preprocessor
        self.__result_queue = result_queue

    def run(self) -> None:
        for image in self.__images:
            analysis_result = self.__analyze_image(image=image)
            self.__result_queue.put(analysis_result)
        self.__result_queue.close()
        self.__result_queue.join_thread()

    def __analyze_image(self,
                        image: np.ndarray,
                        ) -> List[AnalysisResult]:
        preprocessed_ground_truth = self.__preprocessor.preprocess(image=image)
        return self.__analyze_current_ground_truth(
            ground_truth=preprocessed_ground_truth,
        )

    def __analyze_current_ground_truth(self,
                                       ground_truth: PreprocessedGroundTruth,
                                       ) -> List[AnalysisResult]:
        analyzers = map(lambda c: c.analyzer, self.__analysis_couples)
        proceed_single_analysis = partial(
            self.__proceed_single_analysis,
            ground_truth=ground_truth
        )
        return list(map(proceed_single_analysis, analyzers))

    def __proceed_single_analysis(self,
                                  analyzer: GroundTruthAnalyzer,
                                  ground_truth: PreprocessedGroundTruth
                                  ) -> AnalysisResult:
        return analyzer.analyze(ground_truth)


class _ResultsSaver:

    def persist_results(self,
                        target_dir: str,
                        analysis_results: List[AnalysisResult],
                        analyzed_files: List[str]
                        ) -> None:
        target_path = self.__generate_target_file_path(target_dir=target_dir)
        analysis_summary = self.__create_analysis_summary(analysis_results)
        analyzed_files = '\n'.join(analyzed_files)
        summary_text = f'{analysis_summary}\n\n' \
            f'Analyzed files:\n{analyzed_files}'
        dump_text_file(target_path, summary_text)
        logging.info(f'Results were saved under {target_path}')

    def __generate_target_file_path(self, target_dir: str) -> str:
        now = datetime.now()
        timestamp = now.strftime("%m_%d_%Y_%H_%M_%S")
        return os.path.join(
            target_dir,
            f'{timestamp}.log'
        )

    def __create_analysis_summary(self,
                                  analysis_results: List[AnalysisResult]
                                  ) -> str:
        summaries = []
        for result in analysis_results:
            single_summary = f'Name: {result.name}, value: {result.value}'
            summaries.append(single_summary)
        return '\n'.join(summaries)


