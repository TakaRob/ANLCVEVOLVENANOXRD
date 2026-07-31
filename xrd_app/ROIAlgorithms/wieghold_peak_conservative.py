"""Conservative image-only Wieghold peak classifier.

LOSO: F1=0.7975, precision=0.9917, recall=0.6871. Default choice when false
positive boxes cost more than missed weak peaks.
"""
from wieghold_peak_features import classify

WEIGHTS = [-5.240733376045919, 0.6365299768284921, 0.1266633865067956,
           -0.11241323634373161, 0.1632655588716248, 0.34766908203042807,
           -0.4779818473429938, -0.27595587846033776, -1.5022654866548464,
           3.0961203644662336]


def detect_rois(image, sensitivity=0.65, min_distance=4, max_rois=15):
    return classify(image, WEIGHTS, sensitivity, min_distance, max_rois)
