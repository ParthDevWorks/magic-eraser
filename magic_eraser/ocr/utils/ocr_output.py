from dataclasses import dataclass, astuple


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def to_tuple(self):
        return astuple(self)


@dataclass
class OCRResult:
    bounding_box: BoundingBox
    word: str
    confidence: float
