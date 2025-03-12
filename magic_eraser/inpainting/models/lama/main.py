import logging
import os
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
from magic_eraser.inpainting.base import InpaintingModel
from magic_eraser.utils.image import resize_image


class LamOnnx(InpaintingModel):
    """
    Loads the pretrained Lama model weights.

    Raises:
        ValueError:  If the path to the model weights is not found.
    """

    def __init__(self) -> None:

        INPAINTING_MODEL_PATH = os.environ.get("INPAINTING_MODEL_PATH")
        if INPAINTING_MODEL_PATH is None:
            raise ValueError("INPAINTING_MODEL_PATH environment variable is not set")

        self.lamaonnx_path = os.path.join(
            INPAINTING_MODEL_PATH, "lama_onnx/lama_fp32.onnx"
        )

        if not os.path.exists(self.lamaonnx_path):
            raise ValueError(
                f"Lama Model Weights not located inside {INPAINTING_MODEL_PATH}"
            )

        logging.info("Loading Lama model weights")

        self.model = ort.InferenceSession(self.lamaonnx_path)

    def get_model_id(self) -> str:
        """Returns the identifier for the inpainting model.

        Returns:
            str: A string representing the model ID.
        """
        return "lama_onnx"

    def _pre_process(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Resize the tensors to (512, 512)
        resized_image_tensor = resize_image(image=image, target_size=(512, 512))
        resized_mask_tensor = resize_image(image=mask, target_size=(512, 512))

        image_resized = resized_image_tensor.unsqueeze(0)
        mask_resized = resized_mask_tensor[0:1].unsqueeze(0)

        # Convert to NumPy (ONNX requires NumPy arrays)
        image_numpy = image_resized.numpy()
        mask_numpy = mask_resized.numpy().astype(np.float32)

        return image_numpy, mask_numpy

    def _post_processing(
        self, og_image: torch.Tensor, output: np.ndarray
    ) -> torch.Tensor:
        output_t = Image.fromarray(output.transpose(1, 2, 0).astype(np.uint8))
        output_tensor = T.ToTensor()(output_t)

        resized_image = resize_image(
            image=output_tensor, target_size=(og_image.shape[1], og_image.shape[2])
        )

        return resized_image

    def inference(self, image: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            image torch.Tensor: The input image tensor.
            mask torch.Tensor: The mask tensor indicating areas to be inpainted.

        Returns:
            torch.Tensor: The inpainted image tensor.

        """

        if self.model is None:
            raise ValueError(
                "Model is not initialized. Initialize the model before calling inference."
            )

        resized_image, resized_mask = self._pre_process(image, mask)

        # Run the model
        np_array = self.model.run(None, {"image": resized_image, "mask": resized_mask})[
            0
        ][0]

        output = self._post_processing(image, np_array)

        return output

    def shutdown(self) -> None:
        """Cleans up resources by setting the model to None."""
        self.model = None
