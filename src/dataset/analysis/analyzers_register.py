from src.dataset.analysis.analyzers import ClassOccupationAnalyzer, \
    ClassAverageConsolidator, PolyLineComplexityAnalyzer, InstancesAnalyzer, \
    ReceptiveFieldAnalyzer, AverageConsolidator

ANALYZERS = {
    'class_occupation': ClassOccupationAnalyzer,
    'class_average': ClassAverageConsolidator,
    'poly_line_complexity': PolyLineComplexityAnalyzer,
    'instances_analyzer': InstancesAnalyzer,
    'receptive_field': ReceptiveFieldAnalyzer

}

CONSOLIDATORS = {
    'class_occupation': AverageConsolidator,
    'class_average': AverageConsolidator,
    'poly_line_complexity': AverageConsolidator,
    'instances_analyzer': AverageConsolidator,
    'receptive_field': AverageConsolidator
}