from typing import Tuple
import logging
import os

import torch
from PIL import Image
import torchvision.transforms as T
from magic_eraser.utils.image import resize_image
from magic_eraser.models.inpainting.base import InpaintingModel
from diffusers import AutoPipelineForInpainting


class RunWayML(InpaintingModel):
    """
     Loads the pretrained RunWay AI model weights.

    Raises:
        ValueError:  If the path to the model weights is not found.
    """

    def __init__(
        self,
        prompt: str = "Blend with surroundings. Fill seamlessly. Extend the texture. Match the surrounding area",
    ) -> None:

        INPAINTING_MODEL_PATH = os.environ.get("INPAINTING_MODEL_PATH")
        if INPAINTING_MODEL_PATH is None:
            raise ValueError("INPAINTING_MODEL_PATH environment variable is not set")

        self.runwayml_path = os.path.join(
            INPAINTING_MODEL_PATH,
            "runwayml",
            "models--runwayml--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
        )

        if not os.path.exists(self.runwayml_path):
            raise ValueError(
                f"Runway AI Model Weights not located inside {INPAINTING_MODEL_PATH}"
            )

        logging.info("Loading RunwayML weights")

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.prompt = prompt
        self.model = AutoPipelineForInpainting.from_pretrained(
            self.runwayml_path,
        ).to(self.device)

    def get_model_id(self) -> str:
        """Returns the identifier for the inpainting model.

        Returns:
            str: A string representing the model ID.
        """
        return "runwayml"

    def _pre_process(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[Image.Image, Image.Image]:

        resized_image_tensor = resize_image(image=image, target_size=(512, 512))
        resized_mask_tensor = resize_image(image=mask, target_size=(512, 512))

        image_pil = T.ToPILImage()(resized_image_tensor)
        msk_pil = T.ToPILImage()(resized_mask_tensor.to(torch.uint8) * 255)
        return image_pil, msk_pil

    def _post_processing(
        self, og_image: torch.Tensor, output: Image.Image
    ) -> torch.Tensor:
        output_tensor = T.ToTensor()(output)

        resized_image = resize_image(
            image=output_tensor, target_size=(og_image.shape[1], og_image.shape[2])
        )

        return resized_image

    def inference(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

        image_pil, mask_pil = self._pre_process(image, mask)

        output = self.model(
            prompt=self.prompt,
            image=image_pil,
            mask_image=mask_pil,
            generator=self.generator,
            num_inference_steps=10,
            guidance_sacle=10,
        ).images[0]

        final_output = self._post_processing(image, output)

        return final_output

    def shutdown(self) -> None:
        """Cleans up resources by setting the model to None."""
        self.model = None
        self.device = None
