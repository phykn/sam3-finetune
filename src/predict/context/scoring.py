from math import exp, log


def area_ratio_score(candidate_ratio: float, reference_ratio: float) -> float:
    if candidate_ratio <= 0.0 or reference_ratio <= 0.0:
        return 0.0
    return float(exp(-abs(log(candidate_ratio / reference_ratio))))
