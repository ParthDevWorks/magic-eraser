import os
import logging

import torch
import numpy as np
from doctr.models import ocr_predictor, fast_base, crnn_vgg16_bn
from doctr.io import Document, Word, Page

from magic_eraser.ocr.base import OCRModel
from magic_eraser.ocr.utils.ocr_output import OCRResult, BoundingBox


class Doctr(OCRModel):
    """
    Loads the pretrained Doctr model weights.

    Args:
        detector (str): The detector model to use. Defaults to "fast_base".
        recognizer (str): The recognizer model to use. Defaults to "crnn_vgg16_bn".
        detect_language (bool): Whether to detect the language. Defaults to False.

    Raises:
        ValueError:  If the path to the model weights is not found.
    """

    def __init__(
        self,
        detector: str = "fast_base",
        recognizer: str = "crnn_vgg16_bn",
        detect_language: bool = False,
    ) -> None:

        OCR_MODEL_PATH = os.environ.get("OCR_MODEL_PATH")
        if OCR_MODEL_PATH is None:
            raise ValueError("OCR_MODEL_PATH environment variable is not set")

        doctr_path = os.path.join(OCR_MODEL_PATH, "doctr")

        doctr_model_detector_path = os.path.join(
            doctr_path,
            "detection",
            f"{detector}-688a8b34.pt",
        )
        if not os.path.exists(doctr_model_detector_path):
            raise ValueError(
                f"Doctr detector's Model Weights not located inside {doctr_model_detector_path}"
            )

        detector_model = fast_base(pretrained=False, pretrained_backbone=False)
        detector_model.load_state_dict(
            torch.load(doctr_model_detector_path, map_location="cpu")
        )

        doctr_model_recognizer_path = os.path.join(
            doctr_path, "recognition", f"{recognizer}-9762b0b0.pt"
        )

        if not os.path.exists(doctr_model_recognizer_path):
            raise ValueError(
                f"Doctr recognizer's Model Weights not located inside {doctr_model_recognizer_path}"
            )

        recognizer_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
        recognizer_model.load_state_dict(
            torch.load(doctr_model_recognizer_path, map_location="cpu")
        )

        logging.info("Loading Doctr model weights")

        self.model = ocr_predictor(
            det_arch=detector_model,
            reco_arch=recognizer_model,
            pretrained=False,
            pretrained_backbone=False,
            detect_language=detect_language,
        )

    def get_model_id(self) -> str:
        """Returns the identifier for the segmentation model.

        Returns:
            str: A string representing the model ID.
        """
        return "doctr"

    def _preprocess(self, image_tensor: torch.tensor) -> np.ndarray:
        """Preprocesses the image tensor.

        Args:
            image_tensor (torch.Tensor): A tensor representing the image.

        Returns:
            np.ndarray: A NumPy array representing the image. (H x W x C format)
        """
        if image_tensor.shape[0] > 3:
            image_tensor = image_tensor[:3, :, :]
        image_tensor = image_tensor.numpy().transpose(1, 2, 0)
        return image_tensor

    def inference(self, image_tensor: torch.Tensor) -> list[OCRResult]:
        """Performs inference on a image.

        Args:
            image_tensors (torch.Tensor): A tensors representing the image.

        Returns:
            List[OCRResult]: A structured representation of the OCR results.
        """

        if self.model is None:
            raise ValueError("Doctr model has not been initialized")

        image_np = self._preprocess(image_tensor)
        prediction = self.model([image_np])

        post_processing_output = self._post_processing(prediction)

        return post_processing_output

    def _post_processing(self, prediction: Document) -> list[OCRResult]:
        """Processes the raw model predictions into a structured format.

        Args:
            predictions (Document): The raw predictions from the Doctr model.

        Returns:
            list[OCRResult]: A structured representation of the Doctr results.
        """
        output = []

        for page in prediction.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        if word.value:
                            output.append(
                                OCRResult(
                                    bounding_box=self._word_to_bbox(word, page),
                                    word=word.value,
                                    confidence=word.confidence,
                                )
                            )

        return output

    def shutdown(self) -> None:
        """Cleans up resources by setting the model to None."""
        self.model = None

    def _word_to_bbox(self, word: Word, page: Page) -> BoundingBox:
        geometry = np.array(word.geometry)
        geometry[1] -= geometry[
            0
        ]  # Convert from [(xmin, ymin) , (xmax, ymax)] to [(x, y), (w, h)]

        geometry *= np.array(page.dimensions)[None::-1]  # Convert to Pixel Co-ordinates
        x, y, w, h = geometry.flatten()
        return BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h))


if __name__ == "__main__":
    from magic_eraser.utils.image import load_image

    model = Doctr()
    image_tensor = load_image(
        "/Users/parthrathod/Documents/Projects/MAGIC_ERASER/magic-eraser/sample_data/ocr/sample_4.png"
    )
    out = model.inference(image_tensor)
    print(out)
