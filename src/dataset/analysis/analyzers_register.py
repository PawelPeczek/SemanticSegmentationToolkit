from src.dataset.analysis.analyzers import ClassOccupationAnalyzer, \
    ClassAverageConsolidator, PolyLineComplexityAnalyzer, InstancesAnalyzer, \
    ReceptiveFieldAnalyzer, AverageConsolidator

ANALYZERS = {
    'class_occupation': ClassOccupationAnalyzer,
    'poly_line_complexity': PolyLineComplexityAnalyzer,
    'instances_analyzer': InstancesAnalyzer,
    'receptive_field': ReceptiveFieldAnalyzer

}

CONSOLIDATORS = {
    'class_occupation': ClassAverageConsolidator,
    'poly_line_complexity': ClassAverageConsolidator,
    'instances_analyzer': ClassAverageConsolidator,
    'receptive_field': AverageConsolidator
}