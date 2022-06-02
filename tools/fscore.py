def FScore(precision: float, recall: float, beta = 1.) -> float:
    return ((1 + beta**2) * precision * recall) / ((beta**2) * precision + recall)