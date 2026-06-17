from enum import Enum


class ModelType(str, Enum):
    LINEAR        = "Regressão Linear"
    RIDGE         = "Ridge (L2)"
    RANDOM_FOREST = "Random Forest"


class Priority(str, Enum):
    CRITICAL = "CRITICA"
    HIGH     = "ALTA"
    MEDIUM   = "MEDIA"
    OK       = "OK"
