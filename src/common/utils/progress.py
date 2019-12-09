import sys
import multiprocessing


class ProgressReporter:

    def __init__(self,
                 elements_to_process: int):
        self.__elements_to_process = elements_to_process
        self.__already_processed = 0
        self.__lock = multiprocessing.Lock()

    def print_progress(self, clear_previous_line: bool = True) -> None:
        if self.__elements_to_process > 0:
            progress = self.__already_processed / self.__elements_to_process
            progress = f'{progress:.5f}%'
        else:
            progress = 'N/A'
        cr_placeholder = '\r' if clear_previous_line else ''
        sys.stdout.write(f'{cr_placeholder}Progress: {progress}\t\t')
        sys.stdout.flush()

    def report_processed_element(self) -> None:
        self.__lock.acquire()
        if self.__already_processed < self.__elements_to_process:
            self.__already_processed += 1
        self.print_progress()
        self.__lock.release()

