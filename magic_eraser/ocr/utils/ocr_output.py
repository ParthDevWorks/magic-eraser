from dataclasses import dataclass


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class OCRResult:
    bounding_box: BoundingBox
    word: str
    confidence: float
