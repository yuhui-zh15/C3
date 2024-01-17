from enum import Enum


class MappingType(str, Enum):
    MLP = "mlp"
    Transformer = "transformer"
    Linear = "linear"


class Modality(str, Enum):
    Vision = "vision"
    Language = "language"
