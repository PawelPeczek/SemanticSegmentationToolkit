from typing import Optional

from fire import Fire

from src.common.config_utils import DataAnalysisConfigReader
from src.dataset.analysis.core import AnalysisManager


class AnalysisCLI:

    def analyze_dataset(self, config_path: Optional[str] = None) -> None:
        try:
            config = DataAnalysisConfigReader(config_path)
            analysis_manager = AnalysisManager(config=config)
            analysis_manager.run_analysis()
        except Exception as ex:
            print('Failed to analyze dataset. {}'.format(ex))


if __name__ == '__main__':
    Fire(AnalysisCLI)
